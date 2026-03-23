from typing import Any, Optional, List, Dict
from dataclasses import dataclass, asdict

import torch
import numpy as np


@dataclass
class ViewPrediction:
    images: torch.Tensor
    patch_tokens: torch.Tensor
    descriptors: torch.Tensor


@dataclass
class Prediction:
    depth: np.ndarray
    depth_conf: np.ndarray
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    images: np.ndarray
    ids: Optional[List[int]] = None
    mask: Optional[np.ndarray] = None #For mapanything

    @classmethod
    def from_da3(cls, preds: Dict[str, np.ndarray]):
        return cls(
            preds['depth'],
            preds['conf'],
            preds['extrinsics'],
            preds['intrinsics'],
            preds['processed_images'],
        )

    @classmethod
    def from_vggt(cls, preds: Dict[str, np.ndarray]):
        return cls(**preds)

    @classmethod
    def from_dict(cls, preds: Dict[str, np.ndarray]):
        return cls(**preds)

    def asdict(self) -> Dict[str, np.ndarray]:
        return asdict(self)
