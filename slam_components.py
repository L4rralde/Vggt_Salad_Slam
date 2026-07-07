from argparse import ArgumentParser
from typing import List, Dict

import numpy as np
from PIL import Image

from slam_utils import get_publisher, cv2_to_pil
from src.keyframes.frame_overlap import FrameTracker
from src.keyframes.bluriness import tenengrad
from src.models import get_model
from src.storage.fifo_cache import FIFOCache
from src.storage.frames import FrameRepository


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('video-path')
    parser.add_argument('--model', required=True, choices=('vggt', 'da3', 'mapanything'))
    parser.add_argument('--min_disparity', type=float, default=50, help="Minimum disparity to generate a new keyframe")
    parser.add_argument('--group-len', type=int, default=16)
    parser.add_argument('--num-overlap', type=int, default=2)
    return parser.parse_args()


class FramesStorage:
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

    def __call__(self, descs: np.ndarray, descs_ids: List[int], **kwargs) -> Dict[str, tuple]:
        dst_group = len(self._descriptors_groups)
        max_sim = -1
        max_idx = None
        for i, ref_descs in enumerate(self._descriptors_groups[:-2]):
            sim, idcs = LoopDetector.match_desc_groups(ref_descs, descs)
            if sim > max_sim:
                max_sim = sim
                max_idx = (i, *idcs)
        
        self._descriptors_groups.append(descs)
        self._descriptors_groups_ids.append(descs_ids)

        if max_sim < kwargs.get('min_similarity', 0.75):
            return {}
        ref_group, ref_img, dst_img = max_idx
        return {
            'src': (ref_group, self._descriptors_groups_ids[ref_group][ref_img]),
            'dst': (dst_group, descs_ids[dst_img])
        }

    @staticmethod
    def match_desc_groups(descs_a: np.ndarray, descs_b: np.ndarray) -> tuple:
        sim_mat = descs_a @ descs_b.T
        max_sim = np.max(sim_mat)
        row_idx, col_idx = np.unravel_index(max_sim, sim_mat.shape)
        return (max_sim, (row_idx, col_idx))



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

    keyframes_memory = FramesStorage() #Fast access to 
    video_path = args.video_path
    video = get_publisher(video_path)
    
    video_tracker = FrameTracker() #KeyFrame detector
    loop_detector = LoopDetector()

    num_total_imgs = 0
    num_selected_imgs = 0

    preds_cnt = 0
    preds_cache = FIFOCache(2)
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

        #3 Convert to PIL Image
        frame = cv2_to_pil(frame)
        keyframes_memory.append(frame, num_total_imgs, num_selected_imgs)
        
        #4 Stack images to list of images
        new_img_list.append(num_total_imgs)
        img_list = prev_img_list[-args.num_overlap:] + new_img_list
        if len(img_list) < args.group_len:
            continue
        
        #5. 3D reconstruction of submap and global description
        view_preds = model.views_encoding([
            keyframes_memory.get(i) for i in img_list
        ])
        preds = model.chunk_prediction(view_preds)
        preds.ids = np.asarray(img_list)

        descs = view_preds.descriptors[-len(new_img_list):]
        descs_ids = new_img_list

        loop_info = loop_detector(descs, descs_ids)
        


if __name__ == '__main__':
    main()
