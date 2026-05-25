import numpy as np
from scipy.special import huber
from scipy.optimize import minimize

from src.models.dtypes import Prediction
from . import matransforms as mt
from . import  vggt_long_sim3_utils as vggt_long
from .utils import (
    get_pointmap,
    depth_to_pointmap,
    get_conf_mask,
    extr_to_homogeneous,
    get_shared_preds,
)


def estimate_scale_from_depth_maps(
    src_depth: np.ndarray,
    tgt_depth: np.ndarray,
    weights: np.ndarray=None
) -> float:
    if not src_depth.shape == tgt_depth.shape:
        raise ValueError(
            f"Input depths are of different shapes: {src_depth.shape}, {tgt_depth.shape}"
        )
    N = src_depth.shape[0]
    if weights is None:
        weights = np.ones((N))
    else:
        weights = np.sqrt(np.asarray(weights).reshape(N,))

    src_depth = src_depth.ravel()
    tgt_depth = tgt_depth.ravel()

    loss = lambda s: huber(1e-3, weights*(tgt_depth - s*src_depth)).sum()
    scale = minimize(loss, 1.0).x[0]

    return scale


def estimate_scale_from_extrinsics(
    src_extrinsic: np.ndarray,
    tgt_extrinsic: np.ndarray
) -> float:
    #src_extrinsic.shape = (n, 3, 4) or (n, 4, 4)
    if not src_extrinsic.shape == tgt_extrinsic.shape:
        raise ValueError(
            f"Input are of different shapes: {src_extrinsic.shape}, {tgt_extrinsic.shape}"
        )

    n, _, _ = src_extrinsic.shape
    if n == 1:
        raise ValueError("At least two shared images are required for this method")
    src_c2w = np.linalg.inv(extr_to_homogeneous(src_extrinsic))
    tgt_c2w = np.linalg.inv(extr_to_homogeneous(tgt_extrinsic))

    src_origins = src_c2w[..., :3, 3]
    tgt_origins = tgt_c2w[..., :3, 3]
    src_traj_len = np.linalg.norm(
        src_origins[1:] - src_origins[:-1],
        axis=-1
    ).sum()
    tgt_traj_len = np.linalg.norm(
        tgt_origins[1:] - tgt_origins[:-1],
        axis=-1
    ).sum()

    if src_traj_len < 1e-6:
        raise RuntimeError("All camera poses are almost identical")
    
    return tgt_traj_len/src_traj_len

def estimate_scale(
    src_preds: Prediction,
    tgt_preds: Prediction  
) -> float:
    src_preds, tgt_preds = get_shared_preds(src_preds, tgt_preds)
    if src_preds.depth.shape == tgt_preds.depth.shape:

        weights = np.min(
            np.vstack((src_preds.depth_conf[None], tgt_preds.depth_conf[None])),
            axis=0,
        )
        mask = get_conf_mask(src_preds) & get_conf_mask(tgt_preds)
        
        s = estimate_scale_from_depth_maps(
            src_preds.depth[mask],
            tgt_preds.depth[mask],
            weights[mask]
        )
    elif len(src_preds.ids) > 1:
        s = estimate_scale_from_extrinsics(
            src_preds.extrinsic,
            tgt_preds.extrinsic
        )
    else:
        raise RuntimeError("Unable to determine relative scale factor")
    return s


def vggtlong_est_scenes_transform(
    src_preds: Prediction,
    tgt_preds: Prediction
) -> mt.Sim3:
    src_preds, tgt_preds = get_shared_preds(src_preds, tgt_preds)
    common_mask = get_conf_mask(src_preds) & get_conf_mask(tgt_preds)

    src_point = get_pointmap(src_preds)
    tgt_point = get_pointmap(tgt_preds)

    src_point = src_point[common_mask]
    tgt_point = tgt_point[common_mask]

    initial_weights = np.min(
        np.vstack((src_preds.depth_conf[common_mask], tgt_preds.depth_conf[common_mask])),
        axis=0
    )
    s, R, t = vggt_long.robust_weighted_estimate_sim3(
        src_point,
        tgt_point,
        initial_weights
    )

    matrix = np.eye(4)
    matrix[:3, :3] = s*R
    matrix[:3, 3] = t
    return mt.Sim3(matrix)

#FUTURE. Swift-VGGT estimation


def estimate_sim3_from_extrinsics(
    src_preds: Prediction,
    tgt_preds: Prediction,
) -> mt.Sim3:
    src_preds, tgt_preds = get_shared_preds(src_preds, tgt_preds)
    s = estimate_scale(src_preds, tgt_preds)
    src_extrinsic = extr_to_homogeneous(src_preds.extrinsic)
    tgt_extrinsic = extr_to_homogeneous(tgt_preds.extrinsic)
    view_sim3_mat = np.linalg.inv(tgt_extrinsic) @ (s * src_extrinsic)
    if len(view_sim3_mat) == 1:
        return view_sim3_mat[0]

    sim3_estimates = [mt.Sim3(mat) for mat in view_sim3_mat]
    #Future average all estimations
    return sim3_estimates[0]


def estimate_affine_from_extrinsics(
    src_preds: Prediction,
    tgt_preds: Prediction
) -> mt.Affine:
    src_preds, tgt_preds = get_shared_preds(src_preds, tgt_preds)
    s = estimate_scale(src_preds, tgt_preds)
    src_extrinsic = extr_to_homogeneous(src_preds.extrinsic)
    tgt_extrinsic = extr_to_homogeneous(tgt_preds.extrinsic)

    src_intrinsic = np.zeros_like(src_extrinsic)
    src_intrinsic[..., :3, :3] = src_preds.intrinsic
    src_intrinsic[..., 3, 3] = 1.0
    tgt_intrinsic = np.zeros_like(tgt_extrinsic)
    tgt_intrinsic[..., :3, :3] = tgt_preds.intrinsic
    tgt_intrinsic[..., 3, 3] = 1.0

    matrix_estimates = (
        np.linalg.inv(tgt_extrinsic) @
        (s * np.linalg.inv(tgt_intrinsic) @ src_intrinsic) @
        src_extrinsic
    )
    #Future average all estimations
    return mt.Affine(matrix_estimates[0])

#TODO. estimate Homographies with VGGT-SLAM 1.0 method

#TODO. estimate homographies with VGGT-SLAM 2.0 method. Constrained to only one shared image.