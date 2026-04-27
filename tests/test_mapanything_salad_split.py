import sys, os
from typing import Dict, Iterable, List
from argparse import ArgumentParser
from dataclasses import asdict
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from test_utils import ImgDirDataset
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.models.mapanything_salad import MapAnythingSaladSplit
from src.models import Prediction


def parse_args() -> dict:
    parser = ArgumentParser()
    parser.add_argument('img_dir', type=str)
    parser.add_argument('vpr_repo_path', type=str)
    parser.add_argument('--num-seeds', type=int, default=10)
    args = parser.parse_args()

    return args


def compare_predictions(
    ref_preds: Dict[str, np.ndarray],
    preds: Prediction
) -> float:
    renamed_ref_preds = {
        'depth': ref_preds['depth_z'].squeeze(-1).numpy(),
        'depth_conf': ref_preds['conf'].numpy(),
        'extrinsic': torch.linalg.inv(ref_preds['camera_poses'])[:, :3].numpy(),
        'intrinsic': ref_preds['intrinsics'].numpy(),
        'images': ref_preds['img_no_norm'].permute(0, 3, 1, 2).numpy(),
        'mask': ref_preds['mask'].squeeze(-1).numpy()
    }
    preds = asdict(preds)
    acc_diff = 0
    for k in renamed_ref_preds.keys():
        try:
            diff = np.abs(renamed_ref_preds[k] - preds[k]).mean()
        except:
            diff = (renamed_ref_preds[k] ^ preds[k]).astype(np.float32).sum()
        if diff > 1e-6:
            raise RuntimeError(f"{k} mismatch: {diff}")
        acc_diff += diff
    return acc_diff


def preds_dict_list_to_dict(dict_list: List[Dict]) -> Dict:
    single_dict = defaultdict(list)
    for d in dict_list:
        for k, v in d.items():
            single_dict[k].append(v)
    return {k: torch.cat(v) for k, v in single_dict.items()}


def compare_pipelines(
    split_model: MapAnythingSaladSplit,
    dataset: Iterable,
    max_batch_size: int=10,
    num_seeds: int=1
) -> None:
    for i in range(num_seeds):
        seed = i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        curr_max = min(max_batch_size, len(dataset))
        batch_size = random.randint(1, curr_max)
        selected_paths = [
            dataset[j][1]
            for j in random.sample(range(len(dataset)), batch_size)
        ]

        ref_preds = split_model.model.inference(selected_paths)
        ref_preds = preds_dict_list_to_dict(ref_preds)
        ref_preds = {k: v.cpu() for k, v in ref_preds.items()}

        imgs = [Image.open(p) for p in selected_paths]
        perview_preds = split_model.views_encoding(imgs)
        preds = split_model.chunk_prediction(perview_preds)

        acc_diff = compare_predictions(ref_preds, preds)
        view_descriptors = perview_preds.descriptors.cpu().numpy()
        acc_diff += abs(ref_preds['descriptor'] - view_descriptors).sum()
        status = "PASS" if acc_diff < 1e-6 else "FAIL"
        assert status == "PASS"

        print(f"seed: {seed}. diff: {acc_diff: .2f}. {status}")


def main() -> None:
    args = parse_args()
    dataset = ImgDirDataset(args.img_dir)

    print("loading mapanything")
    if not torch.cuda.is_available():
        raise RuntimeError("Only works with cuda")
    
    split_model = MapAnythingSaladSplit(args.vpr_repo_path)
    compare_pipelines(split_model, dataset, num_seeds=args.num_seeds)


if __name__ == '__main__':
    main()
        