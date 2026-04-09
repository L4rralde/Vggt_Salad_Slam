from typing import Callable
import numpy as np
import numpy.typing as npt

from homography import estimate


def generate_orthogonal_matrix(n):
    mat = np.random.rand(n, n)
    q, _ = np.linalg.qr(mat)
    return q


def test_estimate(
    transfom_mat: npt.ArrayLike,
    estimate_fn: Callable
) -> bool:
    src_points = np.random.rand(1000, 4)
    src_points[:, 3] = 1.0
    tgt_points = src_points @ transfom_mat.T
    src_points = src_points[:, :3]
    tgt_points = tgt_points[:, :3] / tgt_points[:, 3:4]
    estimated = estimate_fn(src_points, tgt_points)
    if not np.allclose(transfom_mat, estimated):
        print("Expected transformation:")
        print(transfom_mat)
        print("Estimated one:")
        print(estimated)
        raise RuntimeError("Estimation mismatch")
    
    return True


def test_se3_estimate():
    mat = np.eye(4)
    mat[:3, :3] = generate_orthogonal_matrix(3)
    mat[:3, 3] = np.random.rand(3)
    return test_estimate(mat, estimate.estimate_se3)


def test_sim3_estimate():
    mat = np.eye(4)
    s = 10 * np.exp(np.random.rand(1)).item()
    mat[:3, :3] = s*generate_orthogonal_matrix(3)
    mat[:3, 3] = np.random.rand(3)
    return test_estimate(mat, estimate.estimate_sim3)


def test_affine_estimate():
    mat = np.eye(4)
    mat[:3] = np.random.rand(3, 4)
    return test_estimate(mat, estimate.estimate_affine)


def test_homography_estimate():
    mat = np.random.rand(4, 4)
    mat[3, 3] = 1.0
    return test_estimate(mat, estimate.estimate_homography)


def main():
    n_seeds = 100
    for i in range(n_seeds):
        test_se3_estimate()
        test_sim3_estimate()
        test_affine_estimate()
        test_homography_estimate()
        print(f"seed {i+1}. PASS")


if __name__ == '__main__':
    main()
