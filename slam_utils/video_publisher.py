import os
from typing import List
from glob import glob
from abc import ABC, abstractmethod

from PIL import Image
import numpy as np
import cv2

from .image_dtypes import pil_to_cv2


def sort_files_numerically(file_list):
    return sorted(
        file_list,
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )


class VideoPublisher(ABC):
    @abstractmethod
    def __init__(self, path: str, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_fps(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def read(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def release(self) -> None:
        raise NotImplementedError()


class FolderPublisher(VideoPublisher):
    def __init__(self, path: str, fps: float=-1):
        self.fpaths = self.load_files(path)
        if not self.fpaths:
            raise ValueError(f"Found no image in folder: {path}")
        self.fps = fps
        self.idx_to_publish: int = 0

    def load_files(self, dir: str) -> List[str]:
        fpaths = glob(os.path.join(dir, '*.jpg')) + glob(os.path.join(dir, '*.png'))
        fpaths = sort_files_numerically(fpaths)
        return fpaths

    def get_fps(self) -> float:
        return self.fps

    def read(self) -> np.ndarray:
        if self.idx_to_publish == len(self.fpaths):
            return None
        fpath = self.fpaths[self.idx_to_publish]
        img = pil_to_cv2(Image.open(fpath))
        self.idx_to_publish += 1
        return img

    def release(self) -> None:
        self.idx_to_publish = 0


class VideoCapturePublisher(VideoPublisher):
    def __init__(self, path: str):
        self.video = cv2.VideoCapture(path)

    def get_fps(self) -> int:
        fps = self.video.get(cv2.CAP_PROP_FPS)
        if not fps:
            return -1
        return fps

    def read(self) -> np.ndarray:
        ret, frame = self.video.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        self.video.release()


def get_publisher(path: str, **kwargs) -> VideoPublisher:
    if os.path.isdir(path):
        return FolderPublisher(path, **kwargs)
    if path.endswith('.mp4'):
        return VideoCapturePublisher(path, **kwargs)
    raise ValueError(f"Unrecognize type of path: {path}")
