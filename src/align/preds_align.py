from typing import Dict, Any, Callable, Type
from abc import ABC

import numpy as np
import numpy.typing as npt

from .homography import transforms, estimate
from .align_utils import(
    get_conf_mask,
    get_ids,
    get_pointmap,
    get_shared_preds,
    to_homogeneous
)


class FitAlign(ABC):
    def __init__(self) -> None:
        self._transform: transforms.Transform = None
        self._transform_type: Type[transforms.Transform] = transforms.Transform
        self._estimate_fn: Callable = None

    def fit(
        self,
        src: Dict[str, npt.ArrayLike],
        tgt: Dict[str, npt.ArrayLike],
        **kwargs
    ) -> "FitAlign":        
        src, tgt = get_shared_preds(src,  tgt)
        src_mask = get_conf_mask(src)
        tgt_mask = get_conf_mask(tgt)
        mask = src_mask & tgt_mask
        
        src_conf = src['conf'][mask]
        tgt_conf = tgt['conf'][mask]
        
        weights = np.min(
            np.vstack((src_conf, tgt_conf)),
            axis=0
        )
        src_points = get_pointmap(src)[mask]
        tgt_points = get_pointmap(tgt)[mask]
        self._transform = self._transform_type.from_matrix(
            self._estimate_fn(
                src_points,
                tgt_points,
                weights
            )
        )

        return self

    def transform(self, src: Dict[str, npt.ArrayLike]) -> npt.ArrayLike:
        pointmap = get_pointmap(src)
        original_shape = pointmap.shape
        pointmap = pointmap.reshape((-1, 3))
        return self._transform(pointmap).reshape(original_shape)

    def fit_transform(
        self,
        src: Dict[str, npt.ArrayLike],
        tgt: Dict[str, npt.ArrayLike],
        **kwargs
    ) -> npt.ArrayLike:
        self.fit(src, tgt, **kwargs)
        return self.transform(src)

    def __matmul__(self, other: "FitAlign") -> "FitAlign":
        new_aligner = self.__class__()
        new_aligner._transform = self._transform @ other._transform
        return new_aligner

#FUTURE
#class FitSE3(FitAlign)
        
class FitSim3(FitAlign):
    def __init__(self) -> None:
        self._transform: transforms.Sim3 = None
        self._estimate_fn: Callable = estimate.estimate_sim3
        self._transform_type: Type[transforms.Sim3] = transforms.Sim3


class FitAffine(FitAlign):
    def __init__(self) -> None:
        self._transform: transforms.Affine = None
        self._estimate_fn: Callable = estimate.estimate_affine
        self._transform_type: Type[transforms.Affine] = transforms.Affine

    def fit(
        self, src: Dict[str, Any], tgt: Dict[str, Any], **kwargs
    ) -> "FitAffine":
        src, tgt = get_shared_preds(src,  tgt)
        assert (get_ids(src) == get_ids(tgt))

        src_mask = get_conf_mask(src)
        tgt_mask = get_conf_mask(tgt)
        #In the case the shared image is of different shape in different groups
        # we cannot make a pixel-to-pixel comparison.
        #This may happen if cropping, padding or resizing is performed in a 
        #group in order to make all (tensor) images of the same shape
        if src_mask.shape == tgt_mask.shape:
            mask = src_mask & tgt_mask

            src_conf = src['conf'][mask]
            tgt_conf = tgt['conf'][mask]
            
            weights = np.min(
                np.vstack((src_conf, tgt_conf)),
                axis=0
            )

            s = estimate.estimate_scale(
                src['depth'][mask],
                tgt['depth'][mask],
                weights
            )
        elif len(get_ids(src)) >= 2:
            #We can use the total length of the trajectory to estimate s
            src_t = src['extrinsic'][:, :3, 3]
            src_distance = 0
            for t, next_t in zip(src_t[:-1], src_t[1:]):
                src_distance += np.linalg.norm(next_t - t)

            tgt_t = tgt['extrinsic'][:, :3, 3]
            tgt_distance = 0
            for t, next_t in zip(tgt_t[:-1], tgt[1:]):
                tgt_distance += np.linalg.norm(next_t - t)
            
            s = tgt_distance/src_distance
        else:
            raise RuntimeError("Not enough information to compute scale factor")
            scale = 1.0 #We could set the scale factor to 1 with caveats.

        shared_idx = np.random.randint(len(src['depth']))

        tgt_extrinsic = to_homogeneous(tgt['extrinsic'][shared_idx][None, :])
        tgt_extrinsic = tgt_extrinsic[0]
        tgt_intrinsic = tgt['intrinsic'][shared_idx]
        src_intrinsic = src['intrinsic'][shared_idx]
        src_extrinsic = to_homogeneous(src['extrinsic'][shared_idx][None, :])
        src_extrinsic = src_extrinsic[0]
        A_initial = np.eye(4)
        A_initial[:3, :3] = s * np.linalg.inv(tgt_intrinsic) @ src_intrinsic
        A_initial = tgt_extrinsic @ A_initial @ np.linalg.inv(src_extrinsic)

        self._transform = transforms.Affine.from_matrix(A_initial)
        
        return self


class FitHomography(FitAlign):
    def __init__(self) -> None:
        self._transform: transforms.Homography = None
        self._estimate_fn: Callable = estimate.estimate_homography_ransac
        self._transform_type: Type[transforms.Homography] = transforms.Homography
