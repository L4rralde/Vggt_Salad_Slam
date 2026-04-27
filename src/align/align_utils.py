from typing import Dict, List, Tuple

import numpy.typing as npt
import numpy as np

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

    preds_a_idcs = np.asarray(
        [a_ids.index(id) for id in shared_ids]
    )
    preds_b_idcs = np.asarray(
        [b_ids.index(id) for id in shared_ids]
    )

    preds_a_shared = {k: v[preds_a_idcs] for k, v in preds_a.items()}
    preds_b_shared = {k: v[preds_b_idcs] for k, v in preds_b.items()}
    if len(shared_ids) == 1:
        preds_a_shared = {
            k: v[preds_a_idcs][None, :]
            for k, v in preds_a_shared.items()
        }
        preds_b_shared = {
            k: v[preds_b_idcs][None, :]
            for k, v in preds_b_shared.items()
        }
    return preds_a_shared, preds_b_shared


def to_homogeneous(matrices):
    """
    Convert an array of shape (n, 3, 4) into homogeneous 4x4 matrices (n, 4, 4).

    Parameters:
        matrices (np.ndarray): Input array of shape (n, 3, 4)

    Returns:
        np.ndarray: Output array of shape (n, 4, 4)
    """
    n, r, c = matrices.shape
    if r == 4:
        assert np.abs(matrices[:, 3, :3]).sum() < 1e-6
        return matrices
    homogeneous = np.zeros((n, 4, 4), dtype=matrices.dtype)
    homogeneous[:, :3, :4] = matrices
    homogeneous[:, 3, 3] = 1.0
    return homogeneous


def depth_to_pointmap(
    depth: npt.ArrayLike,
    intrinsic: npt.ArrayLike,
    extrinsic: npt.ArrayLike,
    scale: float=1.0
) -> npt.ArrayLike:
    #Extrinsic world to cam. Camera pose
    ext_w2c = to_homogeneous(extrinsic) #Copy
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
