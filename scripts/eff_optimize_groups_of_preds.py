from pathlib import Path
import re
import argparse
import sys,os
from typing import List
from time import perf_counter
import json

import numpy as np
from tqdm import tqdm
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.models.dtypes import Prediction
from src.transforms.graphs import Sim3Graph, SL4Graph
from src.transforms.estimate import (
    vggtlong_est_scenes_transform,
    estimate_affine_from_extrinsics,
    estimate_sim3_from_extrinsics
)
from src.transforms.utils import to_pointcloud
from src.transforms.matransforms import MatrixTransform


def natural_key(s: str):
    """Sort helper: img2.jpg < img10.jpg"""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", s)
    ]

def find_npz_files(root_dir):
    root = Path(root_dir)
    result = []

    exts = {".npz"}

    for f in root.iterdir():
        if not f.is_file():
            continue
    
        if not f.suffix.lower() in exts:
            continue

        if 'cheat' in str(f):
            continue

        result.append(str(f))
    
    result = sorted(result, key=natural_key)

    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str)
    parser.add_argument('--show-old', action='store_true')
    parser.add_argument('--sl4', action='store_true')

    args = parser.parse_args()
    return args


class PredictionsDir:
    def __init__(self, root_dir):
        self.preds_paths = find_npz_files(root_dir)
    
    def __getitem__(self, key) -> Prediction:
        path = self.preds_paths[key]
        return (os.path.basename(path), Prediction.from_npz_file(path))

    def __len__(self) -> int:
        return len(self.preds_paths)

    @property
    def file_names(self) -> str:
        return [os.path.basename(p) for p in self.preds_paths]


def are_connected(a_ids: Prediction, b_ids: Prediction) -> bool:
    shared_ids = list(set(a_ids) & set(b_ids))
    no_shared = not shared_ids
    return not no_shared


def get_cameras_centers(
    preds: PredictionsDir,
    estimations: List[MatrixTransform],
    variances: List[np.ndarray]
):
    assert len(preds) == len(estimations) == len(variances)
    img_poses_dict = {}

    for fname, pred in preds:
        est = estimations[fname]
        var = variances[fname]

        for img_id, extr in zip(pred.ids, pred.extrinsic):
            full_extr = np.eye(4, dtype=extr.dtype)
            full_extr[:3] = extr[:3]
            pose = np.linalg.inv(full_extr)

            camera_center = pose[:3, 3]
            new_camera_center = est(camera_center)

            if not img_id in img_poses_dict:
                img_poses_dict[img_id] = (new_camera_center, var)
                continue

            _, reg_var = img_poses_dict[img_id]
            if var.sum() < reg_var.sum():
                img_poses_dict[img_id] = (new_camera_center, var)
    
    return {img_id: v[0] for img_id, v in img_poses_dict.items()}


def main():
    args = parse_args()

    preds = PredictionsDir(args.root_dir)

    preds_ids = [p.ids for _, p in preds]
    
    edges = []
    for parent in range(len(preds_ids)):
        for child in range(parent+1, len(preds_ids)):
            parent_preds = preds_ids[parent]
            child_preds = preds_ids[child]
            if not are_connected(parent_preds, child_preds):
                continue
            edges.append((parent, child))
    
    if args.sl4:
        graph = SL4Graph()
        rel_trans_est_function = estimate_affine_from_extrinsics
    else:
        graph = Sim3Graph()
        rel_trans_est_function = vggtlong_est_scenes_transform

    graph.add_anchor_prior(preds[0][0])

    print("Computing local aligners")
    for parent, child in tqdm(edges):
        parent_path, parent_preds = preds[parent]
        child_path, child_preds = preds[child]
        meas = rel_trans_est_function(
            child_preds, parent_preds
        )
        graph.add_measurement(parent_path, child_path, meas)
    
    start = perf_counter()
    prev_est, new_est = graph.optimize(verbose=True)
    end = perf_counter()

    assert list(prev_est.keys()) == list(new_est.keys()) == preds.file_names

    print(f"Pose graph optimization took {end - start:.4f} seconds")

    print("Saving results")
    np.save(
        os.path.join(args.root_dir, 'estimations.npy'),
        np.asarray([
            est._matrix for est in prev_est.values()]
        )
    )


    np.save(
        os.path.join(args.root_dir, 'post_optimization_estimations.npy'),
        np.asarray([
            est._matrix for est in new_est.values()]
        )
    )


    prev_opt_est_variances = graph.eval(prev_est)['variances']
    post_opt_est_variances = graph.eval(new_est)['variances']

    assert prev_est.keys() == post_opt_est_variances.keys()


    prev_optim_cam_centers = get_cameras_centers(
        preds,
        prev_est,
        prev_opt_est_variances
    )

    post_optim_cam_centers = get_cameras_centers(
        preds,
        new_est,
        post_opt_est_variances
    )

    assert prev_optim_cam_centers.keys () == post_optim_cam_centers.keys()

    print("Saving new poses")
    img_list_dump_path = os.path.join(args.root_dir, "image_list.txt")
    with open(img_list_dump_path, 'w') as f:
        f.write('\n'.join(post_optim_cam_centers.keys()))

    prev_optim_cam_centers = np.concatenate(
        [pose[None, ...] for pose in prev_optim_cam_centers.values()],
        axis=0
    )
    np.save(
        os.path.join(args.root_dir, 'pre_optim_cam_centers.npy'),
        prev_optim_cam_centers
    )

    post_optim_cam_centers = np.concatenate(
        [pose[None, ...] for pose in post_optim_cam_centers.values()],
        axis=0
    )
    np.save(
        os.path.join(args.root_dir, 'post_optim_cam_centers.npy'),
        post_optim_cam_centers
    )


    estimators = (
        prev_est
        if args.show_old
        else new_est
    )
    print("Generating pointclouds")

    pcds = [
        to_pointcloud(
            list(estimators.values())[i](preds[i][1]),
            lower_p=60,
            min_conf=1.05
        )
        for i in tqdm(range(len(preds)))
    ]

    o3d.visualization.draw_geometries(pcds)


if __name__ == '__main__':
    main()
