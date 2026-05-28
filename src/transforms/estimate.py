import numpy as np
from scipy.special import huber
from scipy.optimize import minimize, minimize_scalar

from src.models.dtypes import Prediction
from . import matransforms as mt
from . import  vggt_long_sim3_utils as vggt_long
from .graphs import average_transforms
from .utils import (
    get_pointmap,
    depth_to_pointmap,
    get_conf_mask,
    extr_to_homogeneous,
    get_shared_preds,
)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    # Normalize weights
    weights /= weights.sum()

    # Sort by values
    sorted_idx = np.argsort(values)
    values = values[sorted_idx]
    weights = weights[sorted_idx]

    # Find where cumulative weight crosses 0.5
    cumulative = np.cumsum(weights)
    median_idx = np.searchsorted(cumulative, 0.5)

    return values[median_idx]


def merge_depth_conf_min(src_preds: Prediction, tgt_preds: Prediction) -> np.ndarray:
    weights = np.min(
        np.vstack((src_preds.depth_conf[None], tgt_preds.depth_conf[None])),
        axis=0,
    )
    return weights


def merge_depth_conf_mult(src_preds: Prediction, tgt_preds: Prediction) -> np.ndarray:
    return src_preds.depth_conf * tgt_preds.depth_conf


def estimate_scale_wls(src_d: np.ndarray, tgt_d: np.ndarray, w: np.ndarray) -> float:
    src_d = src_d.ravel()
    tgt_d = tgt_d.ravel()
    w = w.ravel()

    return np.sum(w * src_d * tgt_d) / np.sum(w * src_d**2)


def estimate_scale_wls_log(src_d: np.ndarray, tgt_d: np.ndarray, w: np.ndarray) -> float:
    src_d = src_d.ravel()
    tgt_d = tgt_d.ravel()
    w = w.ravel()

    log_ratios = np.log(tgt_d / src_d)
    return np.exp(np.sum(w * log_ratios) / np.sum(w))


def estimate_scale_hubber_loss(src_d, tgt_d, w, delta=1e-3):
    def huber_loss(s):
        residuals = tgt_d - s * src_d
        return np.sum(w * huber(delta, residuals))

    result = minimize_scalar(huber_loss, bounds=(1e-6, 1e6), method='bounded')
    return result.x

def estimate_scale_ransac(src_preds: Prediction, tgt_preds: Prediction) -> np.ndarray:
    mask = get_conf_mask(src_preds) & get_conf_mask(tgt_preds)

    src_depth = src_preds.depth[mask].ravel()
    tgt_depth = tgt_preds.depth[mask].ravel()
    
    weights = merge_depth_conf_mult(src_preds, tgt_preds)[mask].ravel()

    best_s, best_inliers = None, 0
    for _ in range(100):
        idx = np.random.choice(100)
        s_candidate = tgt_depth[idx] / src_depth[idx]
        residuals = np.abs(tgt_depth - s_candidate * src_depth)
        inliers = np.sum((residuals < 1e-4) * weights)
        if inliers > best_inliers:
            best_s, best_inliers = s_candidate, inliers
        
    residuals = np.abs(tgt_depth - best_s * src_depth)
    inliers_idcs = residuals < 0.01
    src_depth = src_depth[inliers_idcs]
    tgt_depth = tgt_depth[inliers_idcs]
    weights = weights[inliers_idcs]

    return estimate_scale_wls_log(src_depth, tgt_depth, weights)


def estimate_scale_weighted_median(src_preds: Prediction, tgt_preds: Prediction) -> float:
    mask = get_conf_mask(src_preds) & get_conf_mask(tgt_preds)

    src_depth = src_preds.depth[mask].ravel()
    tgt_depth = tgt_preds.depth[mask].ravel()
    
    weights = merge_depth_conf_mult(src_preds, tgt_preds)[mask].ravel()

    ratios = tgt_depth/src_depth
    return weighted_median(ratios, weights)


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

        #weights = np.min(
        #    np.vstack((src_preds.depth_conf[None], tgt_preds.depth_conf[None])),
        #    axis=0,
        #)

        weights = src_preds.depth_conf.ravel() * tgt_preds.depth_conf.ravel()
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
) -> mt.Affine|mt.Homography:
    src_preds, tgt_preds = get_shared_preds(src_preds, tgt_preds)
    assert list(src_preds.ids) == list(tgt_preds.ids)

    s = estimate_scale_ransac(src_preds, tgt_preds)
    shared_idx = np.random.randint(len(src_preds.depth))

    tgt_extrinsic = extr_to_homogeneous(tgt_preds.extrinsic)
    tgt_intrinsic = tgt_preds.intrinsic

    src_extrinsic = extr_to_homogeneous(src_preds.extrinsic)
    src_intrinsic = src_preds.intrinsic

    A = np.zeros_like(tgt_extrinsic)
    A[..., 3, 3] = 1.0
    A[..., :3, :3] = s * np.linalg.inv(tgt_intrinsic) @ src_intrinsic
    A = np.linalg.inv(tgt_extrinsic) @ A @ src_extrinsic

    return average_transforms([mt.Affine(mat) for mat in A])

#TODO. estimate Homographies with VGGT-SLAM 1.0 method

#TODO. estimate homographies with VGGT-SLAM 2.0 method. Constrained to only one shared image.