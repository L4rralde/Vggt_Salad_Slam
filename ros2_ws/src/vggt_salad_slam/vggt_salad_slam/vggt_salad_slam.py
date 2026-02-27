import os
from typing import List, Tuple
from collections import deque
from dataclasses import dataclass
from time import perf_counter

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision.utils import save_image
import torch
from addict import Dict

from models import get_model, VggtLikeSaladSplit


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    cv_frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(cv_frame_rgb)
    return pil_frame


@dataclass
class Frame:
    img: Image.Image
    id: int
    timestamp: int


@dataclass
class KeyFrame:
    img: Tensor
    patch_tokens: Tensor
    descriptor: Tensor
    id: int


class KeyFramesBank:
    def __init__(self):
        self.key_frames: Dict[int, KeyFrame] = {}
        self.processed: List[int] = []
        self.not_yet_processed: List[int] = []
        self.last_id: int|None = None

    @property
    def last_descriptor(self) -> Tensor:
        return self.key_frames[self.last_id].descriptor

    def append(self, keyframe: KeyFrame) -> None:
        self.last_id = keyframe.id
        self.key_frames[keyframe.id] = keyframe
        self.not_yet_processed.append(keyframe.id)
        print(f"Appending key frame: {keyframe.id}")

    def find_keyframes(self, view_preds: Dict[str, Tensor], th: float=0.75) -> None:
        current_ref = self.last_descriptor
        new_keyframes = []
        for i, descriptor in enumerate(view_preds['descriptor']):
            sim = (current_ref @ descriptor).item()
            if sim < th:
                current_ref = descriptor
                new_keyframes.append(i)

        return new_keyframes

    def update(self, view_preds: Dict[str, Tensor], frames_buffer: List[Frame]) -> None:
        if not self.key_frames:
            first_kf = KeyFrame(
                view_preds['images'][0],
                view_preds['patch_tokens'][0],
                view_preds['descriptor'][0],
                frames_buffer[0].id    
            )
            self.append(first_kf)
        
        new_kf_idcs = self.find_keyframes(view_preds)
        for i in new_kf_idcs:
            new_kf = KeyFrame(
                view_preds['images'][i],
                view_preds['patch_tokens'][i],
                view_preds['descriptor'][i],
                frames_buffer[i].id
            )
            self.append(new_kf)


    def fetch(self, shared: int) -> Tuple[List[KeyFrame]]:
        return (
            [self.key_frames[id] for id in self.processed[-shared:]],
            [self.key_frames[id] for id in self.not_yet_processed]
        )

    def register(self, model: VggtLikeSaladSplit, shared: int, chunk_size: int) -> None:
        registered, not_registered = self.fetch(shared)
        n = chunk_size - shared
        to_register = not_registered[:n]
        registered_ids = [f.id for f in registered]
        to_register_ids = [f.id for f in to_register]
        all_frames = registered + to_register
        assert len(all_frames) > 0, "No keyframes for 3D preds"
        images = torch.stack([f.img for f in all_frames])
        patch_tokens = torch.stack([f.patch_tokens for f in all_frames])

        view_preds = Dict()
        view_preds.images = images
        view_preds.patch_tokens = patch_tokens

        start = perf_counter()
        print(f"Processing scene of {len(all_frames)} views with idcs ({registered_ids}), ({to_register_ids})")
        model.chunk_prediction(view_preds)
        end = perf_counter()
        print(f"Took {end - start} seconds.")
        self.not_yet_processed = self.not_yet_processed[n:]
        self.processed += to_register_ids


class FramesQ:
    def __init__(self, max_size: int=20):
        self.frames = deque()
        self.max_size = max_size
        self.frame_cnt = 0

    def ready(self) -> bool:
        return len(self.frames) == self.max_size

    def append(self, img_msg: CompressedImage) -> None:
        if self.ready():
            raise RuntimeError("Queue is full and ready for vggt")
        np_arr = np.frombuffer(img_msg.data, np.uint8)
        cv_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        pil_frame = cv2_to_pil(cv_frame)
        frame = Frame(
            pil_frame,
            self.frame_cnt,
            img_msg.header.stamp.sec
        )
        self.frames.append(frame)
        self.frame_cnt += 1

    def clear(self) -> None:
        self.frames.clear()

    def frame_list(self) -> List[Image.Image]:
        return [frame.img for frame in self.frames]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> Frame:
        return self.frames[i]


class VggtSaladSlamNode(Node):
    def __init__(self, model: VggtLikeSaladSplit):
        super().__init__("vggt_salad_slam")
        
        self.model: VggtLikeSaladSplit = model

        self.bridge = CvBridge()
        self.video_subscription = self.create_subscription(
            CompressedImage,
            'camera/image/compressed',
            self.video_callback,
            10
        )

        self.frames_buffer = FramesQ(12)
        self.keyframes = KeyFramesBank()
    
    def video_callback(self, msg: CompressedImage):
        self.get_logger().debug(f"Received frame at {msg.header.stamp}")
        self.frames_buffer.append(msg)

        if self.frames_buffer.ready():
            self.get_logger().info(f"Calling views encoding")
            start = perf_counter()
            view_preds = self.model.views_encoding(self.frames_buffer.frame_list())
            end = perf_counter()
            self.get_logger().info(f"Took {end - start: .2f} s")

            self.keyframes.update(view_preds, self.frames_buffer)
            self.frames_buffer.clear()

            if len(self.keyframes.not_yet_processed) > 5:
                self.keyframes.register(self.model, 5, 10)


def main(args=None):
    rclpy.init(args=args)
    logger = rclpy.logging.get_logger('main')

    logger.info("Loading model")
    model = get_model(
        backbone_arch='vggt',
        vpr_repo='/home/emmanuel/Desktop/tesis/Visual_Place_Recognition'
    )
    logger.info("Finished loading model")
    
    node = VggtSaladSlamNode(model)
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
