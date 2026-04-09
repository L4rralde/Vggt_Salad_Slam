#Code from VGGT-LONG with minor modifications.
#VGGT-Long: https://github.com/DengKaiCQ/VGGT-Long/
#@misc{deng2025vggtlongchunkitloop,
#      title={VGGT-Long: Chunk it, Loop it, Align it -- Pushing VGGT's Limits on Kilometer-scale Long RGB Sequences}, 
#      author={Kai Deng and Zexin Ti and Jiawei Xu and Jian Yang and Jin Xie},
#      year={2025},
#      eprint={2507.16443},
#      archivePrefix={arXiv},
#      primaryClass={cs.CV},
#      url={https://arxiv.org/abs/2507.16443}, 
#}


from typing import List, Tuple

import numpy as np


def apply_sim3(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (s * (R @ points.T)).T + t


def estimate_sim3(source_points: np.ndarray, target_points) -> Tuple[float, np.ndarray, np.ndarray]:
    mu_src = np.mean(source_points, axis=0)
    mu_tgt = np.mean(target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt((src_centered ** 2).sum(axis=1).mean())
    scale_tgt = np.sqrt((tgt_centered ** 2).sum(axis=1).mean())
    s = scale_tgt / scale_src

    src_scaled = src_centered * s

    H = src_scaled.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_tgt - s * R @ mu_src
    return s, R, t


def compute_sim3_ab(S_a, S_b):

    s_a, R_a, T_a = S_a
    s_b, R_b, T_b = S_b

    s_ab = s_b / s_a
    R_ab = R_b @ R_a.T
    T_ab = T_b - s_ab * (R_ab @ T_a)

    return (s_ab, R_ab, T_ab)


def weighted_estimate_se3(
    source_points: np.ndarray,
    target_points: np.ndarray,
    weights: List[float]
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    source_points:  (Nx3)
    target_points:  (Nx3)
    :weights:  (N,) [0,1]
    """
    total_weight = np.sum(weights)
    if total_weight < 1e-6:
        raise ValueError("Total weight too small for meaningful estimation")
    
    normalized_weights = weights / total_weight

    mu_src = np.sum(normalized_weights[:, None] * source_points, axis=0)
    mu_tgt = np.sum(normalized_weights[:, None] * target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    weighted_src = src_centered * np.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * np.sqrt(normalized_weights)[:, None]
    
    H = weighted_src.T @ weighted_tgt

    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_tgt - R @ mu_src
    
    return 1.0, R, t


def weighted_estimate_sim3(
    source_points: np.ndarray,
    target_points: np.ndarray,
    weights: float
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    source_points:  (Nx3)
    target_points:  (Nx3)
    :weights:  (N,) [0,1]
    """
    total_weight = np.sum(weights)
    if total_weight < 1e-6:
        raise ValueError("Total weight too small for meaningful estimation")
    
    normalized_weights = weights / total_weight

    mu_src = np.sum(normalized_weights[:, None] * source_points, axis=0)
    mu_tgt = np.sum(normalized_weights[:, None] * target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt(np.sum(normalized_weights * np.sum(src_centered**2, axis=1)))
    scale_tgt = np.sqrt(np.sum(normalized_weights * np.sum(tgt_centered**2, axis=1)))
    s = scale_tgt / scale_src

    weighted_src = (s * src_centered) * np.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * np.sqrt(normalized_weights)[:, None]
    
    H = weighted_src.T @ weighted_tgt

    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_tgt - s * R @ mu_src
    return (s, R, t)


def huber_loss(residuals: np.ndarray, delta: float) -> np.ndarray:
    abs_residuals = np.abs(residuals)
    return np.where(
        abs_residuals <= delta,
        0.5 * residuals**2,
        delta * (abs_residuals - 0.5 * delta)
    )


def robust_weighted_estimate_sim3(
    src: np.ndarray,
    tgt: np.ndarray,
    init_weights: List[float],
    delta: float=0.1,
    max_iters: int=20,
    tol: float=1e-9,
    using_sim3: bool=True
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    src:  (Nx3)
    tgt:  (Nx3)
    init_weights:  (N,)
    delta: (1). Huber loss delta
    """
    if using_sim3:
        s, R, t = weighted_estimate_sim3(src, tgt, init_weights)
    else:
        s, R, t = weighted_estimate_se3(src, tgt, init_weights)

    prev_error = float('inf')
    
    for iters in range(max_iters):

        transformed = s * (src @ R.T) + t
        residuals = np.linalg.norm(tgt - transformed, axis=1)  # (N,)
        #print(f'[{iters}/{max_iters}] Residuals: {np.mean(residuals)}')
        
        abs_res = np.abs(residuals)
        huber_weights = np.ones_like(residuals)
        large_res_mask = abs_res > delta
        huber_weights[large_res_mask] = delta / abs_res[large_res_mask]
        
        combined_weights = init_weights * huber_weights
        
        combined_weights /= (np.sum(combined_weights) + 1e-12)
        
        if using_sim3:
            s_new, R_new, t_new = weighted_estimate_sim3(src, tgt, combined_weights)
        else:
            s_new, R_new, t_new = weighted_estimate_se3(src, tgt, combined_weights)

        param_change = np.abs(s_new - s) + np.linalg.norm(t_new - t)
        rot_angle = np.arccos(min(1.0, max(-1.0, (np.trace(R_new @ R.T) - 1)/2)))
        current_error = np.sum(huber_loss(residuals, delta) * init_weights)
        
        if (param_change < tol and rot_angle < np.radians(0.1)) or \
           (abs(prev_error - current_error) < tol * prev_error):
            break

        s, R, t = s_new, R_new, t_new
        prev_error = current_error
    
    return (s, R, t)