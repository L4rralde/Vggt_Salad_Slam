
import os
import glob
from typing import List, Callable

from PIL import Image
from torch.utils.data import Dataset


class ImgDirDataset(Dataset):
    def __init__(self, img_dir: str, transform: Callable|None = None):        
        self.img_pahts = ImgDirDataset.scan_dir(img_dir)
        assert len(self.img_pahts) > 0, "Found no valid image"
        self.transform = transform

    @staticmethod
    def scan_dir(img_dir: str) -> List[str]:
        valid_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        all_img_paths = []
        for ext in valid_exts:
            all_img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        
        return sorted(all_img_paths)

    def __len__(self) -> int:
        return len(self.img_pahts)

    def __getitem__(self, index) -> Image.Image:
        img_path = self.img_pahts[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_path
