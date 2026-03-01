import numpy as np
from scipy.optimize import minimize
from scipy.special import huber


def as_homogeneous(extrinsic: np.ndarray) -> np.ndarray:
    homo = np.eye(4, dtype=extrinsic.dtype)
    homo[:3, :] = extrinsic[:3, :] #Copy.
    return homo


def sim3_transform_mat(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=R.dtype)
    mat[:3, :3] = s*R
    mat[:3, 3] = t
    return mat


def closed_form_se3_inv(se3_mat):
    inv = np.zeros_like(se3_mat)
    R = se3_mat[:3, :3]
    t = se3_mat[:3, 3]
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    inv[3, 3] = 1
    return inv


def get_conf_mask(conf):
    lower = np.percentile(conf, 40)
    upper = np.percentile(conf, 80)
    conf_thresh = min(max(1.05, lower), upper)
    mask = (conf > conf_thresh)

    return mask


def est_scale_factor(
    src_depth: np.ndarray,
    src_conf: np.ndarray,
    dst_depth: np.ndarray,
    dst_conf: np.ndarray
) -> float:
    src_mask = get_conf_mask(src_conf)
    dst_mask = get_conf_mask(dst_conf)
    
    common_mask = src_mask & dst_mask

    dst_conf_depth = dst_depth[common_mask]
    src_conf_depth = src_depth[common_mask]

    loss = lambda s: huber(1e-3, dst_conf_depth - s*src_conf_depth).mean()
    scale = minimize(loss, 1.0).x[0]

    return scale


def depth_to_pointmap(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    scale: float=1.0
) -> np.ndarray:
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


def unproject_depth_map_to_point_map(
    depthmaps: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> np.ndarray:
    iterator = zip(depthmaps, intrinsics, extrinsics)
    point_maps = ([
        depth_to_pointmap(depth, intrinsic, extrinsic)[None, ...]
        for depth, intrinsic, extrinsic in iterator
    ])

    return np.vstack(point_maps)
