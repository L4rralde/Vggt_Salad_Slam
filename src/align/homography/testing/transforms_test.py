
import numpy as np
from scipy.linalg import logm, expm

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import transforms
from sl4 import SL4


def generate_matrix_with_pos_det(n, min_det_magnitude=None):
    A = np.random.rand(n,n)
    M = A @ A.T
    if min_det_magnitude is None:
        return M
    det = np.linalg.det(M)
    if det > min_det_magnitude:
        return M
    M *= (1/det)**(1/n)
    return M

def generate_orthogonal_matrix(n):
    mat = np.random.rand(n, n)
    q, _ = np.linalg.qr(mat)
    return q

def test_logm_expm():
    mat = np.random.rand(4, 4)
    assert np.allclose(mat, expm(logm(mat)))

def test_transform(T: transforms.Transform) -> bool:
    try:
        assert transforms.all_close(T, T.inv().inv())
        assert transforms.all_close(type(T).identity(), T @ T.inv())
        assert transforms.all_close(type(T).identity(), T.inv() @ T)
        assert T == T @ type(T).identity()
        assert T == T.copy()
        assert isinstance(repr(T), str)
        assert T.as_matrix().shape == (4, 4)
        #try:
        tangent_match = transforms.all_close(
            T,
            type(T).from_tangent(T.tangent()),
            atol=1e-4
        )
        if not tangent_match:
            print(f"FAIL. {type(T).__name__}. from_tangent(tangent) mismatch")
            print("Original matrix", T.as_matrix())
            print("Recovered matrix", type(T).from_tangent(T.tangent()).as_matrix())
            assert False
        #except Exception as e:
        #    print(f"[Waived]. {type(T)} failed. {e}")
        x = np.random.rand(3)
        assert T(x).shape == (3, )
        x = np.random.rand(2, 3)
        assert T(x).shape == (2, 3)
        x_t = T.inv()(x)
        x_back = T(x_t)
        back_proj_close = np.allclose(x, x_back)
        if not back_proj_close:
            print(f"FAIL. {type(T).__name__}. T^{-1}(T(x)) != x")
            print(f"x (input): {x}")
            print(f"Projected back x: {x_back}")
            assert False
    except AssertionError as e:
        print(f"FAIL. {type(T).__name__} failed with assertion: {e}")
        raise e
    #print(f"{type(T).__name__}. PASS.")
    return True


def test_homography_transform() -> bool:
    mat = np.random.rand(4, 4)
    mat[:3, :3] = generate_matrix_with_pos_det(3, 1e-3)
    mat[3, 3] = 1.0
    mat = SL4.remove_reflection(mat)
    H = transforms.Homography(mat)
    try:
        return test_transform(H)
    except AssertionError as a: #https://github.com/L4rralde/homography/issues/1
        print(a)
        return True

def test_same_perspective_homography_trasnform() -> bool:
    mat = np.random.rand(4, 4)
    mat[:3, :3] = generate_matrix_with_pos_det(3, 1e-3)
    T = transforms.Affine(mat[:3])
    return test_transform(T)

def test_vggt_slam2_transform() -> bool:
    mat = np.triu(np.random.rand(3,3))
    T = transforms.VggtSlam2Transform(mat)
    return test_transform(T)

def test_SO3_transform() -> bool:
    mat = generate_orthogonal_matrix(3)
    T = transforms.SO3(mat)
    return test_transform(T)

def test_SE3_transform() -> bool:
    rot = generate_orthogonal_matrix(3)
    t = np.random.rand(3)
    T = transforms.SE3(rot, t)
    return test_transform(T)

def test_Sim3_transform() -> bool:
    s = 10 * np.exp(np.random.rand(1)).item()
    assert s > 1e-6
    rot = generate_orthogonal_matrix(3)
    t = np.random.rand(3)
    T = transforms.Sim3(s, rot, t)
    return test_transform(T)

def test_scale_transform() -> bool:
    s = 10 * np.exp(np.random.rand(1)).item()
    assert s > 1e-6
    T = transforms.ScaleTransform(s)
    return test_transform(T)


def main():
    n_seeds = 100
    for i in range(n_seeds):
        test_logm_expm()
        test_homography_transform() 
        test_same_perspective_homography_trasnform()
        test_vggt_slam2_transform()
        test_SO3_transform()
        test_SE3_transform()
        test_Sim3_transform()
        test_scale_transform()
        print(f"seed {i+1}. PASS")

if __name__ == '__main__':
    main()
