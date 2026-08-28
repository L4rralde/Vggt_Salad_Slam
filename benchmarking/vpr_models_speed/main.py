import os
import argparse
from pathlib import Path
from typing import Optional, Callable

from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, input_dir: os.PathLike, transform: Optional[Callable] = None) -> None:
        self.input_dir = Path(input_dir)
        self.paths = sorted(
            p for p in self.input_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        self.transform = transform or T.ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self.transform(img)

        return img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('--batch-size', type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = os.path.realpath(args.input_dir)
    
    transform = T.Compose([
        T.Resize((322, 322), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDataset(input_dir, transform)
    
    # Cap workers to avoid memory thrashing on high-core machines
    num_workers = min(os.cpu_count()//2, 8)
    
    print(f"Dataset size: {len(dataset)} images")
    print(f"Batch size: {args.batch_size}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True # Ensures consistent batch sizes for accurate timing
    )

    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
    model = model.eval().to('cuda')

    # Enable benchmark since input size is static (322x322)
    torch.backends.cudnn.benchmark = True
    
    total_batches = len(dataloader)
    warmup_batches = total_batches // 10
    print("warmup_batches", warmup_batches)
    print("total batches", total_batches)
    print("batch size", args.batch_size)
    
    starters = []
    enders = []
    
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
        for i, imgs in tqdm(enumerate(dataloader), total=total_batches, desc="Calculating descriptors"):
            imgs = imgs.to('cuda', non_blocking=True)

            if i < warmup_batches:
                output = model(imgs)
                continue

            if i == warmup_batches:
                torch.cuda.synchronize()
            
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            
            starter.record()
            output = model(imgs)
            ender.record()
            
            starters.append(starter)
            enders.append(ender)

        # Synchronize ONCE at the end so the CPU/DataLoader isn't blocked during the loop
        torch.cuda.synchronize()
    
    if not starters:
        print("Not enough batches to calculate timing after warmup.")
        return

    times = np.array([s.elapsed_time(e) for s, e in zip(starters, enders)])
    
    avg_time_ms = times.mean()
    avg_time_sec = avg_time_ms / 1000
    
    print(f"Average batch model inference time: {avg_time_sec:.6f} seconds")
    print(f"Number of batches per second: {1 / avg_time_sec:.2f}")


if __name__ == '__main__':
    main()