import numpy as np
import numpy.typing as npt
from scipy.special import huber
from scipy.optimize import minimize
from scipy.optimize import least_squares

from homography.estimate.vggt_long_sim3_utils import (
    robust_weighted_estimate_sim3
)


def estimate_sim3(
    src_points: npt.ArrayLike,
    tgt_points: npt.ArrayLike,
    weights: npt.ArrayLike=None
) -> npt.ArrayLike:
    N = src_points.shape[0]
    if weights is None:
        weights = np.ones((N))
    s, R, t = robust_weighted_estimate_sim3(
        src_points, tgt_points, weights
    )
    mat = np.eye(4)
    mat[:3, :3] = s*R
    mat[:3, 3] = t
    return mat


def estimate_se3(
    src_points: npt.ArrayLike,
    tgt_points: npt.ArrayLike,
    weights: npt.ArrayLike=None
) -> npt.ArrayLike:
    N = src_points.shape[0]
    if weights is None:
        weights = np.ones((N))
    _, R, t = robust_weighted_estimate_sim3(
        src_points, tgt_points, weights, using_sim3=False
    )
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = t
    return mat


def est_scale(
    scr_depth: npt.ArrayLike,
    tgt_depth: npt.ArrayLike,
    weights: npt.ArrayLike=None
) -> float:
    N = scr_depth.shape[0]
    if weights is None:
        weights = np.ones((N))
    else:
        weights = np.sqrt(np.asarray(weights).reshape(N, 1))

    loss = lambda s: huber(1e-3, weights*(tgt_depth - s*scr_depth)).sum()
    scale = minimize(loss, 1.0).x[0]

    return scale


def __to_homogeneous(points_nx3: npt.ArrayLike) -> npt.ArrayLike:
    """
    Transforms Nx3 points to Nx4 homogeneous coordinates.
    """
    n = points_nx3.shape[0]
    ones = np.ones((n, 1))
    return np.hstack((points_nx3, ones))


def __compute_initial_homography(
    X: npt.ArrayLike,
    Xp: npt.ArrayLike,
    weights: npt.ArrayLike=None
) -> npt.ArrayLike:
    """
    Compute initial 3D homography using Direct Linear Transform (DLT) with weights.

    Args:
        X: Nx3 numpy array, source points in Euclidean coordinates
        Xp: Nx3 numpy array, target points in Euclidean coordinates
        weights: N numpy array (or list), weights for each point pair. 
                 Defaults to 1.0 for all points.

    Returns:
        H_initial: 4x4 numpy array, initial homography estimate
    """
    X = __to_homogeneous(X)
    N = X.shape[0]
    
    if N < 5:
        raise ValueError(
            "At least 5 points are required for homographyt estimation"
        )
        
    if weights is None:
        weights = np.ones((N, 1))
    else:
        weights = np.sqrt(np.asarray(weights).reshape(N, 1))

    u = Xp[:, 0:1]
    v = Xp[:, 1:2]
    w = Xp[:, 2:3]
    
    A = np.zeros((3 * N, 16))
    
    A[0::3, 0:4] = -X * weights
    A[0::3, 12:16] = X * u * weights
    
    A[1::3, 4:8] = -X * weights
    A[1::3, 12:16] = X * v * weights
    
    A[2::3, 8:12] = -X * weights
    A[2::3, 12:16] = X * w * weights

    U, S, Vt = np.linalg.svd(A)
    
    h_initial = Vt[-1, :]
    
    H_initial = h_initial.reshape(4, 4)
    
    if H_initial[3, 3] == 0:
        raise RuntimeError(
            "Estimated transformation projects to infinity. H[3,3] = 0"
        )
    H_initial /= H_initial[3, 3]
        
    return H_initial


def __refine_3D_homography(
    H_initial: npt.ArrayLike,
    X: npt.ArrayLike,
    Xp: npt.ArrayLike,
    weights: npt.ArrayLike=None
) -> npt.ArrayLike:
    """
    Refine a 3D homography given initial estimate, 3D points, and weights.

    Args:
        H_initial: 4x4 numpy array, initial homography
        X: Nx3 numpy array, source points in Euclidean coordinates
        Xp: Nx3 numpy array, target points in Euclidean coordinates
        weights: N numpy array (or list), weights for each point pair. 
                 Defaults to 1.0 for all points.

    Returns:
        H_refined: 4x4 numpy array, refined homography
    """
    
    H_initial = H_initial / H_initial[3, 3]
    X = __to_homogeneous(X)
    N = X.shape[0]
    
    if weights is None:
        weights = np.ones((N, 1))
    else:
        weights = np.sqrt(np.asarray(weights).reshape(N, 1))
    
    def reprojection_error(h15):
        H = np.append(h15, 1.0).reshape(4, 4)
        X_proj = X @ H.T
        X_h = X_proj[:, :3] / X_proj[:, 3:4] 
        
        weighted_error = weights * (X_h - Xp)
        return weighted_error.ravel()
    
    def jacobian(h15):
        H = np.append(h15, 1.0).reshape(4, 4)
        X_proj = X @ H.T
        
        U = X_proj[:, 0:1]
        V = X_proj[:, 1:2]
        W = X_proj[:, 2:3]
        S = X_proj[:, 3:4]
        
        J = np.zeros((3 * N, 16))
        
        J[0::3, 0:4] = X / S
        J[0::3, 12:16] = -U * X / (S**2)
        
        J[1::3, 4:8] = X / S
        J[1::3, 12:16] = -V * X / (S**2)
        
        J[2::3, 8:12] = X / S
        J[2::3, 12:16] = -W * X / (S**2)
        
        J = J[:, :15]
        
        weights_repeated = np.repeat(weights, 3, axis=0)
        
        return J * weights_repeated

    h0 = H_initial.flatten()[:15]
    
    res = least_squares(
        reprojection_error,
        h0,
        jac=jacobian,
        method='lm',
        verbose=1
    )
    
    H_refined = np.append(res.x, 1.0).reshape(4, 4)
    
    return H_refined


def __compute_initial_affine(
    X: npt.ArrayLike,
    Xp: npt.ArrayLike,
    weights: npt.ArrayLike=None
) -> npt.ArrayLike:
    """
    Compute initial 3D affine transformation using linear least squares.

    Args:
        X: Nx3 numpy array, source points in Euclidean coordinates
        Xp: Nx3 numpy array, target points in Euclidean coordinates
        weights: N numpy array (or list), weights for each point pair.

    Returns:
        A_initial: 4x4 numpy array, exact affine transformation
    """
    X = __to_homogeneous(X)
    N = X.shape[0]
    
    if N < 4:
        raise ValueError(
            "Se requieren al menos 4 pares de puntos para una transformación afín 3D."
        )
        
    if weights is None:
        weights = np.ones((N, 1))
    else:
        weights = np.asarray(weights).reshape(N, 1)

    sqrt_weights = np.sqrt(weights)
    X_weighted = X * sqrt_weights
    Xp_weighted = Xp * sqrt_weights

    M_T, _, _, _ = np.linalg.lstsq(X_weighted, Xp_weighted, rcond=None)
    
    A_initial = np.eye(4)
    A_initial[:3, :4] = M_T.T
    
    return A_initial


def __refine_coplanar_affine(
    A_initial: npt.ArrayLike,
    X: npt.ArrayLike,
    Xp: npt.ArrayLike,
    weights: npt.ArrayLike=None,
    alpha: float=1.0
) -> npt.ArrayLike:
    """
    Refine a 12 DoF affine matrix for coplanar points using Tikhonov regularization.
    
    Args:
        A_initial: 4x4 numpy array, initial affine matrix from extra info
        X: Nx3 numpy array, source points in Euclidean coordinates
        Xp: Nx3 numpy array, target points in Euclidean coordinates
        weights: N numpy array, point confidences
        alpha: Float, regularization weight. Higher means stricter adherence to A_initial.
    """
    X = __to_homogeneous(X)
    N = X.shape[0]
    
    if weights is None:
        weights = np.ones((N, 1))
    else:
        weights = np.asarray(weights).reshape(N, 1)

    a0 = A_initial[:3, :4].flatten()
    
    sqrt_alpha = np.sqrt(alpha)

    def reprojection_error(a12):
        M = a12.reshape(3, 4)
        X_proj = X @ M.T
        
        point_error = (weights * (X_proj - Xp)).ravel()
        
        reg_error = sqrt_alpha * (a12 - a0)
        
        return np.concatenate([point_error, reg_error])
    
    def jacobian(a12):
        J_pts = np.zeros((3 * N, 12))
        J_pts[0::3, 0:4] = X
        J_pts[1::3, 4:8] = X
        J_pts[2::3, 8:12] = X
        
        weights_repeated = np.repeat(weights, 3, axis=0)
        J_pts *= weights_repeated

        J_reg = sqrt_alpha * np.eye(12)
        
        return np.vstack([J_pts, J_reg])

    res = least_squares(
        reprojection_error,
        a0,
        jac=jacobian,
        method='lm',
        verbose=1
    )
    
    A_refined = np.eye(4)
    A_refined[:3, :4] = res.x.reshape(3, 4)
    
    return A_refined


def estimate_homography(
    src_points: npt.ArrayLike,
    tgt_points: npt.ArrayLike,
    weights: npt.ArrayLike=None,
    H_initial: npt.ArrayLike=None,
) -> npt.ArrayLike:
    """
    Compute Homography between two point clouds.

    Args:
        src_points: Nx3 numpy array, source points in Euclidean coordinates
        tgt_points: Nx3 numpy array, target points in Euclidean coordinates
        weights: N numpy array (or list), weights for each point pair. 
                 Defaults to 1.0 for all points.
        H_initial: 4x4 numpy array, initial estimation.
                 Defaults to Direct Linear Transform (DLT) with weights solution
    Returns:
        H: 4x4 numpy array, homography estimate
    """
    if H_initial is None:
        H_initial = __compute_initial_homography(
            src_points, tgt_points, weights
        )
    
    return __refine_3D_homography(
        H_initial, src_points, tgt_points, weights
    )


def estimate_affine(
    src_points: npt.ArrayLike,
    tgt_points: npt.ArrayLike,
    weights: npt.ArrayLike=None,
    A_initial: npt.ArrayLike=None,
    alpha: float=1.0
) -> npt.ArrayLike:
    """
    Compute Affine transformation between two point clouds.

    Args:
        src_points: Nx3 numpy array, source points in Euclidean coordinates
        tgt_points: Nx3 numpy array, target points in Euclidean coordinates
        weights: N numpy array (or list), weights for each point pair. 
                 Defaults to 1.0 for all points.
        A_initial: 4x4 numpy array, initial estimation.
                 Defaults to Direct Linear Transform (DLT) with weights solution
        alpha: Float, regularization weight. Higher means stricter adherence to A_initial.
        
    Returns:
        H: 4x4 numpy array, homography estimate
    """
    if A_initial is None:
        A_initial = __compute_initial_affine(
            src_points, tgt_points, weights
        )
    
    return __refine_coplanar_affine(
        A_initial, src_points, tgt_points, weights, alpha
    )
