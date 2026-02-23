import sys, os
from typing import Iterable
from argparse import ArgumentParser
import random

import numpy as np
import torch
from PIL import Image

from test_utils import ImgDirDataset
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.models.vggt_salad import VggtSaladSplit


def parse_args() -> dict:
    parser = ArgumentParser()
    parser.add_argument('img_dir', type=str)
    parser.add_argument('vpr_repo_path', type=str)
    parser.add_argument('--num-seeds', type=int, default=10)
    args = parser.parse_args()

    return args


def compare_pipelines(
    vggt_salad_split:  VggtSaladSplit,
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

        vggt_preds = vggt_salad_split.model.inference(selected_paths)
        imgs = [Image.open(p) for p in selected_paths]
        perview_preds = vggt_salad_split.per_view_encoding(imgs)
        perseq_latent = vggt_salad_split.per_sequence_encoding(perview_preds)
        preds = vggt_salad_split.heads_prediction(perseq_latent)
        chunk_preds = vggt_salad_split.views_chunk_predicton(perview_preds) #This are the two las stages aggregated.

        shared_keys = set([
            k
            for k in vggt_preds.keys()
            if k in preds
        ])

        acc_diff = 0.0
        for key in shared_keys:
            diff = np.abs(vggt_preds[key] - preds[key]).sum()
            if diff > 1e-6:
                raise RuntimeError(f"{key} mismatch: {diff}")
            acc_diff += diff
            diff = np.abs(vggt_preds[key] - chunk_preds[key]).sum()
            if diff > 1e-6:
                raise RuntimeError(f"{key} mismatch: {diff}")
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
    
    vggt_salad_split = VggtSaladSplit(args.vpr_repo_path)
    compare_pipelines(vggt_salad_split, dataset, num_seeds=args.num_seeds)


if __name__ == '__main__':
    main()
