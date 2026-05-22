from typing import Dict, Tuple

import numpy.typing as npt
import numpy as np

from src.models.dtypes import Prediction


def extr_to_homogeneous(extr_matrices):
    """
    Convert an array of shape (n, 3, 4) into homogeneous 4x4 extr_matrices (n, 4, 4).

    Parameters:
        extr_matrices (np.ndarray): Input array of shape (n, 3, 4)

    Returns:
        np.ndarray: Output array of shape (n, 4, 4)
    """
    n, r, c = extr_matrices.shape
    if r == 4:
        assert np.abs(extr_matrices[:, 3, :3]).sum() < 1e-6
        return extr_matrices
    homogeneous = np.zeros((n, 4, 4), dtype=extr_matrices.dtype)
    homogeneous[:, :3, :4] = extr_matrices
    homogeneous[:, 3, 3] = 1.0
    return homogeneous


def get_conf_mask(
    preds: Prediction,
    lower_p: float=40.0,
    min_conf: float =1.05,
    upper_p: float=80.0
) -> npt.ArrayLike:
    conf = preds.depth_conf
    lower = np.percentile(conf, lower_p)
    upper = np.percentile(conf, upper_p)
    conf_thresh = min(max(min_conf, lower), upper)

    if not 'mask' in preds.keys():
        return conf > conf_thresh

    return preds.mask & (conf > conf_thresh)


def get_shared_preds(
    preds_a: Prediction,
    preds_b: Prediction
) -> Tuple[Prediction, Prediction]:
    a_ids = preds_a.ids
    b_ids = preds_b.ids
    shared_ids = list(set(a_ids) & set(b_ids))
    if not shared_ids:
        raise ValueError("Preds do not share any view")

    preds_a_idcs = np.asarray(
        [a_ids.index(id) for id in shared_ids]
    )
    preds_b_idcs = np.asarray(
        [b_ids.index(id) for id in shared_ids]
    )

    preds_a_shared = {
        k: v[preds_a_idcs] 
        for k, v in preds_a.asdict().items()
    }
    preds_b_shared = {
        k: v[preds_b_idcs]
        for k, v in preds_b.asdict().items()
    }

    if len(shared_ids) == 1:
        preds_a_shared = {
            k: v[preds_a_idcs][None, :]
            for k, v in preds_a_shared.items()
        }
        preds_b_shared = {
            k: v[preds_b_idcs][None, :]
            for k, v in preds_b_shared.items()
        }
    return (
        Prediction.from_dict(preds_a_shared),
        Prediction.from_dict(preds_b_shared)
    )


def depth_to_pointmap(
    depth: npt.ArrayLike,
    intrinsic: npt.ArrayLike,
    extrinsic: npt.ArrayLike,
    scale: float=1.0
) -> npt.ArrayLike:

    #Extrinsic world to cam. Camera pose
    ext_w2c = extr_to_homogeneous(extrinsic) #Copy
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


def get_pointmap(preds: Prediction) -> npt.ArrayLike:
    if preds.pointmap is not None:
        return preds.pointmap
    pointmap = depth_to_pointmap(
        preds.pointmap,
        preds.intrinsic,
        preds.extrinsic
    )
    return pointmap


def to_pointcloud(
    preds: Prediction,
    lower_p: float=40,
    min_conf: float=1.02,
    upper_p: float=80
):
    import open3d as o3d

    mask = get_conf_mask(preds, lower_p, min_conf, upper_p)

    points = get_pointmap(preds)[mask].reshape(-1, 3)
    colors = preds.images[mask].reshape(-1, 3)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud
