from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.models.dtypes import Prediction
from dataclasses import replace


def canonical_homogeneous(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (3, 4):
        homo_matrix = np.eye(4, dtype=matrix.dtype)
        homo_matrix[:3] = matrix
    elif matrix.shape == (4, 4):
        homo_matrix = matrix.copy()
    else:
        raise ValueError(f"Expected 3x4 or 4x4 matrix, got {matrix.shape}")
    
    if abs(homo_matrix[3,3]) < 1e-12:
        raise ValueError("Homogeneous matrix has zero in (3,3) entry, cannot normalize")
    homo_matrix /= homo_matrix[3,3]
    return homo_matrix


class MatrixTransform(ABC):
    def __init__(self, matrix: np.ndarray|None=None) -> None:
        self._matrix = None

    def __call__(self, pred: Prediction) -> Prediction:
        return self.transform(pred)
    
    @abstractmethod
    def transform(self, pred: Prediction|np.ndarray) -> Prediction:
        raise NotImplementedError

    @abstractmethod
    def inv(self) -> "MatrixTransform":
        raise NotImplementedError()

    def copy(self) -> "MatrixTransform":
        return self.__class__(self._matrix.copy())

    def __matmul__(self, other: "MatrixTransform") -> "MatrixTransform":
        return matrix_transforms_matmul(self, other)


def matrix_transforms_matmul(left: MatrixTransform, right: MatrixTransform) -> MatrixTransform:
    if not isinstance(left, MatrixTransform):
        raise ValueError(f"input not supported: {left}")
    if not isinstance(right, MatrixTransform):
        raise ValueError(f"input not supported: {right}")
    
    if isinstance(left, Homography) or isinstance(right, Homography):
        cast_type = Homography
    elif isinstance(left, Affine) or isinstance(right, Affine):
        cast_type = Affine
    elif isinstance(left, Sim3) or isinstance(right, Sim3):
        cast_type = Affine
    elif isinstance(left, SE3) or isinstance(right, SE3):
        cast_type = SE3
    else:
        raise ValueError("Types not supported yet. Future")

    return cast_type(left._matrix @ right._matrix)

class SE3(MatrixTransform):
    def __init__(self, matrix: np.ndarray|None=None) -> None:
        if matrix is None:
            self._matrix = np.eye(4)
            return
        matrix = canonical_homogeneous(matrix)
        if not np.allclose(matrix[3, :3], np.asarray([0,0,0])):
            raise ValueError(f"Invalid SE(3) matrix: {matrix}")
        if not abs(np.linalg.det(matrix[:3, :3]) - 1.0) < 1e-6:
            raise ValueError(f"Invalid SE(3) matrix. Rotation matrix is orthogonal): {matrix}")
        self._matrix = np.eye(4, dtype=matrix.dtype)
        self._matrix[:3] = matrix[:3]
    
    def inv(self) -> "SE3":
        rot_inv = np.transpose(self._matrix[:3, :3])
        t_inv = - rot_inv @ self._matrix[:3, 3]
        new_mat = np.eye(4, dtype=self._matrix.dtype)
        new_mat[:3, :3] = rot_inv
        new_mat[:3, 3] = t_inv
        return self.__class__(new_mat)



class Sim3(MatrixTransform):
    def __init__(self, matrix: np.ndarray|None=None) -> None:
        if matrix is None:
            self._matrix = np.eye(4)
            return
        matrix = canonical_homogeneous(matrix)
        if not np.allclose(matrix[3, :3], np.asarray([0,0,0])):
            raise ValueError(f"Invalid Sim(3) matrix: {matrix}")
        if not np.linalg.det(matrix[:3, :3]) > 1e-6:
            raise ValueError(f"Invalid Sim(3) matrix. scale-rotation matrix with negative determinant: {matrix}")
        
        self._matrix = np.eye(4, dtype=matrix.dtype)
        self._matrix[:3] = matrix[:3]
    
    def inv(self) -> "Sim3":
        rot_scale = self._matrix[:3, :3]
        scale = np.linalg.det(rot_scale)**(1/3)
        rot = rot_scale/scale
        t = self._matrix[:3, 3]

        inv_rot_scale = np.transpose(rot)/scale
        inv_t = - inv_rot_scale @ t
        inv_mat = np.eye(4, dtype=self._matrix.dtype)
        inv_mat[:3, :3] = inv_rot_scale
        inv_mat[:3, 3] = inv_t

        return self.__class__(inv_mat)
    
    def transform(self, pred: Prediction) -> Prediction:
        sR = self._matrix[:3, :3]
        t = self._matrix[:3, 3]

        s = np.linalg.det(sR)**(1/3)
        new_depth = s * pred.depth
        new_extrinsics = s * pred.extrinsic @ np.linalg.inv(self._matrix)
        if pred.pointmap is None:
            new_pointmap = None
        else:
            new_pointmap = pred.pointmap @ sR.T + t
        new_pred = replace(
            pred,
            depth=new_depth,
            extrinsic=new_extrinsics,
            pointmap=new_pointmap
        )
        return new_pred


def rq_decomposition(A):
    Q, R = np.linalg.qr(
        np.swapaxes(np.flip(A, axis=-2), -1, -2)
    )
    
    # Correct the shapes and orientations
    R = np.flip(
        np.swapaxes(R, -1, -2),
        axis=-2
    )
    R = np.flip(R, axis=-1)
    Q = np.swapaxes(Q, -1, -2)
    Q = np.flip(Q, axis=-2)
    
    return R, Q


class Affine(MatrixTransform):
    def __init__(self, matrix: np.ndarray | None=None) -> None:
        if matrix is None:
            self._matrix = np.eye(4)
            return

        matrix = canonical_homogeneous(matrix)
        if not np.allclose(matrix[3, :3], np.asarray([0,0,0])):
            raise ValueError(f"Invalid Affine matrix: {matrix}")
        if not np.linalg.det(matrix[:3, :3]) > 1e-6:
            raise ValueError(f"Invalid Affine matrix. Matrix includes reflection: {matrix}")
        
        self._matrix = np.eye(4, dtype=matrix.dtype)
        self._matrix[:3] = matrix[:3]

    def inv(self) -> "Affine":
        A = self._matrix[:3, :3]
        t = self._matrix[:3, 3]

        inv_mat = np.eye(4, dtype=self._matrix.dtype)
        inv_mat[:3, :3] = np.linalg.inv(A)
        inv_mat[:3, 3] = - inv_mat[:3, :3] @ t

        return self.__class__(inv_mat)
    
    def transform(self, pred: Prediction) -> Prediction:
        A = self._matrix[:3, :3]
        t = self._matrix[:3, 3]

        K = pred.intrinsic
        R = pred.extrinsic[..., :3, :3]
        T = pred.extrinsic[..., :3, 3]

        M = K @ R @ np.linalg.inv(A)

        K_new, R_new = rq_decomposition(M)

        diag_sign = np.sign(np.diagonal(K_new, axis1=-2, axis2=-1))
        S = np.zeros_like(K_new)
        idx = np.arange(3)
        S[..., idx, idx] = diag_sign

        K_new = K_new @ S
        R_new = S @ R_new

        scale = 1.0 / K_new[..., 2, 2]
        scale = scale[..., None, None]

        K_new = scale * K_new

        K_corr = np.linalg.inv(K_new) @ K
        T_corr = np.squeeze(K_corr @ T[..., None], axis=-1)

        new_trans = T_corr - R_new @ t

        new_extr = pred.extrinsic.copy()
        new_extr[..., :3, :3] = R_new
        new_extr[..., :3, 3] = new_trans

        if pred.pointmap is None:
            new_pointmap = None
        else:
            new_pointmap = pred.pointmap @ A.T + t
        
        new_depth = scale * pred.depth

        new_pred = replace(
            pred,
            depth=new_depth,
            extrinsic=new_extr,
            intrinsic=K_new,
            pointmap=new_pointmap
        )
        return new_pred


class Homography(MatrixTransform):
    def __init__(self, matrix: Any | None=None) -> None:
        if matrix is None:
            self._matrix = np.eye(4)
            return
    
        matrix = canonical_homogeneous(matrix)
        if not np.linalg.det(matrix) > 1e-6:
            raise ValueError(f"Invalid homography matrix. Include reflection: {matrix}")
    
        self._matrix = matrix
    
    def inv(self) -> "Homography":
        return self.__class__(np.linalg.inv(self._matrix))

