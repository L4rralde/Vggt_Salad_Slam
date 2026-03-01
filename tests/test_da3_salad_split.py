import sys, os
from typing import Iterable, Dict
from argparse import ArgumentParser
import random

import numpy as np
import torch
from PIL import Image

from test_utils import ImgDirDataset
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.models.da3_salad import Da3SaladSplit, Prediction


def parse_args() -> dict:
    parser = ArgumentParser()
    parser.add_argument('img_dir', type=str)
    parser.add_argument('vpr_repo_path', type=str)
    parser.add_argument('--num-seeds', type=int, default=10)
    args = parser.parse_args()

    return args


def compare_predictions(
    pred_dict: Dict[str, np.ndarray],
    pred_obj: Prediction,
) -> float:
    dict_keys = ('depth', 'conf', 'extrinsics', 'intrinsics', 'processed_images')
    obj_keys = ('depth', 'depth_conf', 'extrinsic', 'intrinsic', 'images')
    acc_diff = 0.0
    for dict_k, obj_k in zip(dict_keys, obj_keys):
        if not dict_k in pred_dict:
            raise KeyError(dict_k)
        diff = np.abs(pred_dict[dict_k] - getattr(pred_obj, obj_k)).sum()
        if diff > 1e-6:
            raise RuntimeError(f"{dict_k} mismatch: {acc_diff}")
        acc_diff += diff
    return acc_diff


def compare_pipelines(
    da3_salad_split:  Da3SaladSplit,
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

        da3_preds = da3_salad_split.model.inference(selected_paths, process_res=504)
        imgs = [Image.open(p) for p in selected_paths]
        perview_preds = da3_salad_split.views_encoding(imgs)
        perseq_latent = da3_salad_split.sequence_encoding(perview_preds)
        preds = da3_salad_split.heads_prediction(perseq_latent)
        chunk_preds = da3_salad_split.chunk_prediction(perview_preds) #This are the two las stages aggregated.

        acc_diff = compare_predictions(da3_preds, preds)
        acc_diff += compare_predictions(da3_preds, chunk_preds)

        view_descriptors = perview_preds.descriptors.numpy()
        diff = abs(da3_preds['descriptor'] - view_descriptors).sum()
        if diff > 1e-6:
            raise RuntimeError(f"descriptor mismatch: {diff}")
        acc_diff += diff

        status = "PASS" if acc_diff < 1e-6 else "FAIL"
        assert status == "PASS"

        print(f"seed: {seed}. diff: {acc_diff: .2f}. {status}")


def main() -> None:
    args = parse_args()
    dataset = ImgDirDataset(args.img_dir)

    print("loading vggt")
    if not torch.cuda.is_available():
        raise RuntimeError("Only works with cuda")
    configs = ['small', 'base', 'large', 'giant']
    for conf in configs:
        print(f"Testing on Da3_Salad_{conf.capitalize()}")
        da3_salad_split = Da3SaladSplit(args.vpr_repo_path, conf)
        compare_pipelines(da3_salad_split, dataset, num_seeds=args.num_seeds)


if __name__ == '__main__':
    main()
