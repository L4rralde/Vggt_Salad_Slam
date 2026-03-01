from typing import Any, Optional, List
from dataclasses import dataclass

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
