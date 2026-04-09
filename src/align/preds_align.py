from typing import Dict, Tuple, List, Any, Callable
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from src.align.homography import transforms, estimate


def get_conf_mask(
    preds: Dict[str, npt.ArrayLike],
    lower_p: float=40.0,
    min_conf: float =1.05,
    upper_p: float=80.0
) -> npt.ArrayLike:
    conf = preds['conf']
    lower = np.percentile(conf, lower_p)
    upper = np.percentile(conf, upper_p)
    conf_thresh = min(max(min_conf, lower), upper)

    if not 'mask' in preds.keys():
        return conf > conf_thresh

    return preds['mask'] & (conf > conf_thresh)


def get_ids(preds: Dict[str, npt.ArrayLike]) -> List:
    if 'ids' in preds:
        return list(preds['ids'])
    elif 'image_names' in preds:
        return list(preds['image_names'])
    raise ValueError("Input preds has no ids")


def get_shared_preds(
    preds_a: Dict[str, npt.ArrayLike],
    preds_b: Dict[str, npt.ArrayLike]
) -> Tuple[Dict[str, npt.ArrayLike], Dict[str, npt.ArrayLike]]:
    a_ids = get_ids(preds_a)
    b_ids = get_ids(preds_b)
    shared_ids = list(set(a_ids) & set(b_ids))
    if not shared_ids:
        raise ValueError("Preds do not share any view")

    preds_a_shared = {k: v[shared_ids] for k, v in preds_a.items()}
    preds_b_shared = {k: v[shared_ids] for k, v in preds_b.items()}
    if len(shared_ids) == 1:
        preds_a_shared = {
            k: v[shared_ids][None, :]
            for k, v in preds_a_shared.items()
        }
        preds_b_shared = {
            k: v[shared_ids][None, :]
            for k, v in preds_b_shared.items()
        }
    return preds_a_shared, preds_b_shared


def as_homogeneous(extrinsic: npt.ArrayLike) -> npt.ArrayLike:
    homo = np.eye(4, dtype=extrinsic.dtype)
    homo[:3, :] = extrinsic[:3, :] #Copy.
    return homo


def depth_to_pointmap(
    depth: npt.ArrayLike,
    intrinsic: npt.ArrayLike,
    extrinsic: npt.ArrayLike,
    scale: float=1.0
) -> npt.ArrayLike:
    #Extrinsic world to cam. Camera pose
    ext_w2c = as_homogeneous(extrinsic) #Copy
    K = intrinsic
    K_inv = np.linalg.inv(K)

    H, W = depth.shape[:2]

    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)

    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3) # (H*W, 3)
    scaled_depth = scale * depth #Complete copy
    d_flat = scaled_depth.reshape(-1)

    rays = K_inv @ pix.T

    Xc = rays * d_flat[None, :]
    Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))]) #Homogeneus vector of each ray
    c2w = np.linalg.inv(ext_w2c)
    Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M,3)
    Xw = Xw.reshape(H, W, 3)

    return Xw


def get_pointmap(preds: Dict[str, npt.ArrayLike]) -> npt.ArrayLike:
    if 'world_points' in preds.keys():
        return preds['world_points']
    pointmap = depth_to_pointmap(
        preds['depth'],
        preds['intrinsic'],
        preds['extrinsic']
    )
    return pointmap


class FitAlign(ABC):
    def __init__(self) -> None:
        self._transform: transforms.Transform = None
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
            np.vstack((src_conf[mask], tgt_conf[mask])),
            axis=0
        )
        src_points = get_pointmap(src)[mask]
        tgt_points = get_pointmap(tgt)[mask]
        self._transform = self._estimate_fn(
            src_points,
            tgt_points,
            weights
        )

        return self

    def transform(self, src: Dict[str, npt.ArrayLike]) -> npt.ArrayLike:
        pointmap = get_pointmap(src)
        return self._transform(pointmap)

    def fit_transform(
        self,
        src: Dict[str, npt.ArrayLike],
        tgt: Dict[str, npt.ArrayLike],
        **kwargs
    ) -> npt.ArrayLike:
        self.fit(src, tgt, **kwargs)
        self.transform(src)

#FUTURE
#class FitSE3(FitAlign)
        
class FitSim3(FitAlign):
    def __init__(self) -> None:
        self._transform: transforms.Sim3 = None
        self._estimate_fn: Callable = estimate.estimate_sim3


class FitAffine(FitAlign):
    def __init__(self) -> None:
        self._transform: transforms.Affine = None
        self._estimate_fn: Callable = estimate.estimate_affine

    def fit(
        self, src: Dict[str, Any], tgt: Dict[str, Any], **kwargs
    ) -> "FitAffine":
        src, tgt = get_shared_preds(src,  tgt)
        src_mask = get_conf_mask(src)
        tgt_mask = get_conf_mask(tgt)
        mask = src_mask & tgt_mask

        src_conf = src['conf'][mask]
        tgt_conf = tgt['conf'][mask]
        
        weights = np.min(
            np.vstack((src_conf[mask], tgt_conf[mask])),
            axis=0
        )

        s = estimate.estimate_scale(
            src['depth'][mask],
            tgt['depth'][mask],
            weights
        )

        shared_idx = np.random.randint(len(src['depth']))

        tgt_extrinsic = tgt['extrinsic'][shared_idx]
        tgt_R = tgt_extrinsic[:3, :3]
        tgt_t = tgt_extrinsic[:3, 3]
        tgt_intrinsic = tgt['intrinsic'][shared_idx]
        src_intrinsic = src['intrinsic'][shared_idx]
        src_extrinsic = as_homogeneous(src['extrinsic'])
        A_initial = np.eye(4)
        A_initial[:3, :3] = s * (
            tgt_R.T @ np.linalg.inv(tgt_intrinsic) @ src_intrinsic
        )
        A_initial[:3, 3] = -1.0 * tgt_R.T @ tgt_t
        A_initial @= src_extrinsic

        src_points = get_pointmap(src)[mask]
        tgt_points = get_pointmap(tgt)[mask]

        alpha = kwargs.get('alpha', 1.0)
        self._transform = estimate.estimate_affine(
            src_points,
            tgt_points,
            weights,
            A_initial,
            alpha
        )
        
        return self


class FitHomography(FitAlign):
    def __init__(self) -> None:
        self._transform: transforms.Homography = None
        self._estimate_fn: Callable = estimate.estimate_homography