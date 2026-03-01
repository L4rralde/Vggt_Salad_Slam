from typing import Tuple
from abc import abstractmethod

import numpy as np
from addict import Dict

from .utils import(
    get_conf_mask,
    depth_to_pointmap,
    closed_form_se3_inv,
    as_homogeneous,
    sim3_transform_mat,
    unproject_depth_map_to_point_map
)
from .third_party.vggt_long.sim3_utils import robust_weighted_estimate_sim3


def vggtlong_est_scenes_transform(
    src_preds: Dict[str, np.ndarray],
    tgt_preds: Dict[str, np.ndarray]
) -> Tuple[float, np.ndarray, np.ndarray]:
    src_ids = list(src_preds.ids)
    dst_ids = list(src_preds.ids)

    common_ids = set(src_ids).intersection(set(dst_ids))
    if not common_ids:
        raise ValueError("No overlapping views")
    
    src_idcs = [src_ids.index(id) for id in common_ids]
    dst_idcs = [dst_ids.index(id) for id in common_ids]

    src_tgt = src_preds.depth_conf[src_idcs]
    tgt_conf = tgt_preds.depth_conf[dst_idcs]
    common_mask = get_conf_mask(src_tgt) & get_conf_mask(tgt_conf)

    src_point = unproject_depth_map_to_point_map(
        src_preds.depth,
        src_preds.intrinsics,
        src_preds.extrinsics
    )
    tgt_point = unproject_depth_map_to_point_map(
        tgt_preds.depth,
        tgt_preds.intrinsics,
        tgt_preds.extrinsics
    )

    src_point = src_point[src_idcs][common_mask]
    tgt_point = tgt_point[dst_idcs][common_mask]

    initial_weights = np.min(
        np.vstack((src_tgt[common_mask], tgt_conf[common_mask])),
        axis=0
    )
    s, R, t = robust_weighted_estimate_sim3(
        src_point,
        tgt_point,
        initial_weights
    )

    return s, R, t


class Sim3Align:
    def __init__(self):
        self.sim3: Tuple[float, np.ndarray, np.ndarray] = ()

    @abstractmethod
    def fit(
        self,
        tgt_preds: Dict[str, np.ndarray],
        src_preds: Dict[str, np.ndarray],
    ) -> "Sim3Align":
        raise NotImplementedError()

    def fit_transform(
        self,
        tgt_preds: Dict[str, np.ndarray],
        src_preds: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        self.fit(tgt_preds, src_preds)
        return self.transform(src_preds)

    def transform(self, preds: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        s, R, t = self.sim3
        new_preds = Dict(preds)
        new_preds.depth = s*preds.depth
        
        se3_t = sim3_transform_mat(1.0, R, t)
        se3_t_inv = closed_form_se3_inv(se3_t)
        
        new_extrinsics = []
        for extrinsic in preds.extrinsics:
            extrinsic = as_homogeneous(extrinsic).copy()
            extrinsic[:3, 3] *= s
            new_extrinsic = (extrinsic @ se3_t_inv)[:3]
            new_extrinsics.append(new_extrinsic[None, ...])
        new_preds.extrinsics = np.vstack(new_extrinsics)

        return new_preds


class VggtlongAlign(Sim3Align):
    def fit(
        self,
        tgt_preds: Dict[str, np.ndarray],
        src_preds: Dict[str, np.ndarray],
    ) -> "VggtlongAlign":
        self.sim3 = vggtlong_est_scenes_transform(tgt_preds, src_preds)
        return self
