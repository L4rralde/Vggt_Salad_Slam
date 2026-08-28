import re
from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image


def natural_key(s: str):
    """Sort helper: img2.jpg < img10.jpg"""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", s)
    ]


def find_images(root_dir: Path):
    root_dir = Path(root_dir)
    images = []

    jpeg_exts = {".jpg", ".jpeg", ".png"}

    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in jpeg_exts:
            images.append(path)

    images.sort(key=lambda p: natural_key(str(p.relative_to(root_dir))))
    return images


def find_image_dirs(root_dir):
    root = Path(root_dir)
    result = {}

    jpeg_exts = {".jpg", ".jpeg", ".png"}

    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue

        images = [
            str(p)
            for p in subdir.iterdir()
            if p.is_file() and p.suffix.lower() in jpeg_exts
        ]

        if images:
            result[str(subdir.name)] = sorted(
                images,
                key=lambda p: natural_key(Path(p).name)
            )
    
    return result


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, return_path=False):
        self.transform = transform
        self.files = sorted(
            [p for p in Path(root_dir).iterdir() if p.is_file()],
            key=lambda p: natural_key(Path(p).name),
        )
        self.return_path = return_path

        if not self.files:
            raise RuntimeError(f"No files found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.return_path:
            return str(self.files[idx]), image
        return image


def build_groups(items, n, m):
    """
    Create overlapping groups.

    Example:
        n=3, m=1

        [0,1,2]
        [2,3,4]
        [4,5,6]
    """
    if n <= 0:
        raise ValueError("n must be > 0")

    if m >= n:
        raise ValueError("m must be smaller than n")

    step = n - m
    groups = []

    for start in range(0, len(items), step):
        group = items[start:start + n]

        if not group:
            break

        groups.append(group)

        if len(group) < n:
            break

    return groups
