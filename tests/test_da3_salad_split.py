import sys, os
from typing import Iterable
from argparse import ArgumentParser
import random

import numpy as np
import torch
from PIL import Image

from test_utils import ImgDirDataset
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.models.da3_salad import Da3SaladSplit


def parse_args() -> dict:
    parser = ArgumentParser()
    parser.add_argument('img_dir', type=str)
    parser.add_argument('vpr_repo_path', type=str)
    parser.add_argument('--num-seeds', type=int, default=10)
    args = parser.parse_args()

    return args


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

        shared_keys = set([
            k
            for k in da3_preds.keys()
            if k in preds
        ])

        acc_diff = 0.0
        for key in shared_keys:
            diff = np.abs(da3_preds[key] - preds[key]).sum()
            if diff > 1e-6:
                raise RuntimeError(f"{key} mismatch: {diff}")
            acc_diff += diff
            diff = np.abs(da3_preds[key] - chunk_preds[key]).sum()
            if diff > 1e-6:
                raise RuntimeError(f"{key} mismatch: {diff}")
            acc_diff += diff

        view_descriptors = perview_preds['descriptor'].numpy()
        desc_diff = abs(da3_preds['descriptor'] - view_descriptors).sum()
        #acc_diff += desc_diff
        print(desc_diff)
        #if diff > 1e-6:
        #    raise RuntimeError(f"{key} mismatch: {diff}")

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
