import os
from pathlib import Path
import re

import numpy as np
from tqdm import tqdm

from src.models import Prediction
from src.transforms.sgraph.scale_graph import ScaleGraph
from src.transforms.estimate import(
    vggtlong_est_scenes_transform,
    EstimateScaleAnchorIntrinsic
)

root_path = '/media/emmanuel/hdd_storage/kitti_odom_groups/08/da3_preds/'


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


def main():
    preds = PredictionsDir(root_path)

    preds_ids = [p.ids for _, p in preds]

    edges = []
    for parent in range(len(preds_ids)):
        for child in range(parent+1, len(preds_ids)):
            parent_preds = preds_ids[parent]
            child_preds = preds_ids[child]
            if not are_connected(parent_preds, child_preds):
                continue
            edges.append((parent, child))

    graph = ScaleGraph()
    values = {0: 1.0}

    #Adjacent estimations:
    for i, j in tqdm(edges):
        parent_path, parent_preds = preds[i]
        child_path, child_preds = preds[j]
        meas = vggtlong_est_scenes_transform(
            child_preds, parent_preds
        )
        s_ij = meas.s
        graph.add_measurement(i, j, s_ij, weight=1.0)

        if not j in values:
            values[j] = values[i] * s_ij
    
    #Final global scaleing factor estimation
    root_preds = preds[0]
    anchor_k = root_preds.intrinsic[0]
    
    g_scale_estimator = EstimateScaleAnchorIntrinsic(anchor_k)
    
    for j in range(1, len(preds_ids)):
        graph.add_measurement(
            0,
            j,
            g_scale_estimator(preds[j]),
            weight=0.1
        )


    initial_scales = [values[i] for i in range(len(preds_ids))]
    new_scales = graph.optimize(initial_scales)

    print(initial_scales)
    print(new_scales)

