from argparse import ArgumentParser
from typing import List, Dict, Tuple
import os

import numpy as np
from PIL import Image

from slam_utils import get_publisher, cv2_to_pil
from src.keyframes.frame_overlap import FrameTracker
from src.keyframes.bluriness import tenengrad
from src.models import get_model, Prediction
from src.storage import(
    FIFOCache,
    FrameRepository,
    NdarrayRepository
)
from src.transforms.estimate import vggtlong_est_scenes_transform
from src.transforms.graphs import Sim3Graph
from src.transforms.utils import to_pointcloud


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('video_path')
    parser.add_argument('--model', required=True, choices=('vggt', 'da3', 'mapanything'))
    parser.add_argument('--min_disparity', type=float, default=20, help="Minimum disparity to generate a new keyframe")
    parser.add_argument('--group-len', type=int, default=16)
    parser.add_argument('--num-overlap', type=int, default=2)
    parser.add_argument('--viz', action='store_true')
    return parser.parse_args()


class FramesMemory:
    def __init__(self) -> None:
        self._ids_dict = {}
        self._cache = FIFOCache(max_size=32)
        self._repo = FrameRepository('output/.tmp/frames', clear=True)
    
    def append(self, frame: Image.Image, img_cnt: int, sel_img_cnt: int) -> None:
        self._ids_dict[img_cnt] = sel_img_cnt
        self._cache.append(img_cnt, frame)
        repo_inner_id = self._repo.append(frame, img_cnt)
        assert repo_inner_id == sel_img_cnt

    def get(self, frame_id: int) -> Image.Image:
        if frame_id in self._cache:
            return self._cache.get(frame_id)
        repo_innner_id = self._ids_dict[frame_id]
        frame, repo_img_cnt = self._repo.get(repo_innner_id)
        assert frame_id == repo_img_cnt
        return frame


class LoopDetector:
    def __init__(self) -> None:
        self._descriptors_groups = []
        self._descriptors_groups_ids = []
    
    def append(self, descs: np.ndarray, descs_ids: List[int]) -> None:
        self._descriptors_groups.append(descs)
        self._descriptors_groups_ids.append(descs_ids)

    def __call__(self, descs: np.ndarray, descs_ids: List[int], min_similarity: float=0.8) -> Dict[str, tuple]:
        """
        Appends descriptors of new group of imgaes.
        And matches another group finding most similar pair of images.
        """
        dst_group = len(self._descriptors_groups)
        max_sim = -1
        max_idx = None
        for i, ref_descs in enumerate(self._descriptors_groups[:-2]):
            sim, idcs = LoopDetector.match_desc_groups(ref_descs, descs)
            if sim > max_sim:
                max_sim = sim
                max_idx = (i, *idcs)
        
        self.append(descs, descs_ids)
    
        
        if max_sim < min_similarity:
            print(f"Max sim is not enough: {max_sim}")
            return {}
        ref_group, ref_img, dst_img = max_idx
        return {
            'src': (ref_group, self._descriptors_groups_ids[ref_group][ref_img]),
            'dst': (dst_group, descs_ids[dst_img]),
            'sim': max_sim
        }

    @staticmethod
    def match_desc_groups(descs_a: np.ndarray, descs_b: np.ndarray) -> tuple:
        sim_mat = descs_a @ descs_b.T
        max_sim = np.max(sim_mat)
        row_idx, col_idx = np.unravel_index(
            np.argmax(sim_mat),
            sim_mat.shape
        )
        return (max_sim, (row_idx, col_idx))


class PredsMemory:
    def __init__(self, namespace: str) -> None:
        self.root = os.path.join('output', namespace)
        self._preds_paths: Dict[int, str] = {}
        self._preds_cache: FIFOCache = FIFOCache(4)
        self.__total_preds = 0
    
    def __len__(self) -> int:
        return self.__total_preds
    
    def get_path(self, pred_id: int) -> str:
        return os.path.join(self.root, f'{pred_id}.npz')
    
    def append(self, pred: Prediction) -> Tuple[Prediction, Prediction]:
        """
        Appends newest prediction and returns (prev, new) if prev exit
        """
        pred_id = self.__total_preds 

        self._preds_cache.append(pred_id, pred)
        npz_path = self.get_path(pred_id)
        self._preds_paths[pred_id] = npz_path
        np.savez(npz_path, **pred.asdict())

        self.__total_preds += 1
        
        if pred_id < 1:
            return ()
        return (
            self._preds_cache.get(pred_id - 1),
            pred
        )

    def get(self, pred_id: int) -> Prediction:
        if pred_id in self._preds_cache:
            return self._preds_cache.get(pred_id)
        npz_path = self.get_path(pred_id)
        return Prediction.from_npz_file(npz_path)


def get_kf_window(closed_loop_side, preds_memory):
    group_id, match_kf = closed_loop_side
    preds = preds_memory.get(group_id)

    idx = np.argmax(preds.ids == match_kf)
    start = max(0, idx - 1)
    end = min(len(preds.ids), idx + 2)

    return group_id, preds, preds.ids[start:end]


def main():
    args = parse_args()

    #Load model
    VPR_REPO = '/home/emmanuel/Desktop/tesis/Visual_Place_Recognition'
    if args.model == 'vggt':
        model = get_model('vggt', VPR_REPO)
    elif args.model == 'da3':
        model = get_model('da3-giant', VPR_REPO)
    elif args.model == 'mapanything':
        model = get_model('mapanything', VPR_REPO)

    keyframes_memory = FramesMemory() #Fast access to 
    video_path = args.video_path
    video = get_publisher(video_path)
    unaligned_preds_memory = PredsMemory('unaligned_preds')
    
    video_tracker = FrameTracker() #KeyFrame detector
    loop_detector = LoopDetector()
    graph = Sim3Graph()
    graph.add_anchor_prior(0)

    num_total_imgs = 0
    num_selected_imgs = 0

    prev_img_list = []
    new_img_list = []
    while True:
        #1. Get frames
        frame = video.read()
        if frame is None:
            break
        num_total_imgs += 1
        
        #FUTURE. Do something with this.
        #tenengrad(frame)

        #2. Filter keyframes
        enough_disparity = video_tracker.compute_disparity(
            frame,
            args.min_disparity
        )
        if not enough_disparity:
            continue
        num_selected_imgs += 1

        #3 Convert to PIL Images
        frame = cv2_to_pil(frame)
        keyframes_memory.append(frame, num_total_imgs, num_selected_imgs)
        
        #4 Stack images to list of images
        new_img_list.append(num_total_imgs)
        img_list = prev_img_list[-args.num_overlap:] + new_img_list
        if len(img_list) < args.group_len:
            continue
        
        #5. 3D reconstruction of submap and global descriptors
        print(f"Processing group of images: {img_list}")
        view_preds = model.views_encoding([
            keyframes_memory.get(i) for i in img_list
        ]) #DINO tokens, processed images,, and global descriptors
        preds = model.chunk_prediction(view_preds) #3D reconstruction and camera properties
        preds.ids = np.asarray(img_list) #Per-image identifies

        #5a. Store Prediction output
        adjacent_preds = unaligned_preds_memory.append(preds) #Appends new preds. Returns {prev, new} if prev exists

        #5b. Store global descriptors
        descs = view_preds.descriptors[-len(new_img_list):].numpy()
        descs_ids = new_img_list
        closed_loop = loop_detector(descs, descs_ids) #Appends global descriptors, but also returns 

        prev_img_list = new_img_list
        new_img_list = []
        if not adjacent_preds:
            continue

        #6 Local aligning
        prev_pred, curr_pred = adjacent_preds
        meas = vggtlong_est_scenes_transform(curr_pred, prev_pred)
        child_id = len(unaligned_preds_memory) - 1
        parent_id = child_id - 1
        graph.add_measurement(parent_id, child_id, meas)

        if not closed_loop:
            continue

        print("Found loop closure")
        print(closed_loop)
        #7 Loop closure
        src_group_id, src_preds, src_kf_ids = get_kf_window(closed_loop['src'], unaligned_preds_memory)
        dst_group_id, dst_preds, dst_kf_ids = get_kf_window(closed_loop['dst'], unaligned_preds_memory)

        connecting_kf_ids = np.concatenate([src_kf_ids, dst_kf_ids])
        view_preds = model.views_encoding([
            keyframes_memory.get(i) 
            for i in connecting_kf_ids
        ])
        aux_preds = model.chunk_prediction(view_preds)
        aux_preds.ids = connecting_kf_ids

        #FIXME. By the moment we are not going to store this preds becuase it
        #makes the numbering differ (from closed-loop detector to memory). Nonethelles,
        #Graph will be aware of it.
        meas_dst_aux = vggtlong_est_scenes_transform(
            aux_preds, dst_preds
        )
        meas_aux_src = vggtlong_est_scenes_transform(
            src_preds, aux_preds
        )
        graph.add_measurement(
            dst_group_id, 
            src_group_id, 
            meas_dst_aux @ meas_aux_src
        )

        _, new_est = graph.optimize(verbose=True)
        graph.update_estimation(new_est)

    np.save(
        os.path.join('output', 'estimations.npy'),
        np.asarray([
            est._matrix for est in new_est.values()]
        )
    )

    if args.viz:
        import open3d as o3d
        pcds = [
            to_pointcloud(
                list(new_est.values())[i](unaligned_preds_memory.get(i)),
                lower_p=60,
                min_conf=1.05
            )
            for i in range(len(unaligned_preds_memory))
        ]
        o3d.visualization.draw_geometries(pcds)

if __name__ == '__main__':
    main()
