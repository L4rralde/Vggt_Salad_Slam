from abc import ABC, abstractmethod
from typing import Type, Any

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
    def __init__(self, matrix: Any=None) -> None:
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
    

def get_predominant_instance(*transforms: MatrixTransform) -> Type:
    for mat_type in [Homography, Affine, Sim3, SE3]:
        if any([isinstance(t, mat_type) for t in transforms]):
            return mat_type
    raise ValueError("Types not supported yet. Future")


def matrix_transforms_matmul(left: MatrixTransform, right: MatrixTransform) -> MatrixTransform:
    cast_type = get_predominant_instance(left, right)
    return cast_type(left._matrix @ right._matrix)


class SE3(MatrixTransform):
    def __init__(self, matrix: Any=None) -> None:
        if matrix is None:
            self._matrix = np.eye(4)
            return
        
        if isinstance(matrix, MatrixTransform):
            matrix = matrix._matrix
        
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
    def __init__(self, matrix: Any=None) -> None:
        if matrix is None:
            self._matrix = np.eye(4)
            return
        
        if isinstance(matrix, MatrixTransform):
            matrix = matrix._matrix
        
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

class Affine(MatrixTransform):
    def __init__(self, matrix: Any=None) -> None:
        if matrix is None:
            self._matrix = np.eye(4)
            return

        if isinstance(matrix, MatrixTransform):
            matrix = matrix._matrix
        
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
    

class Homography(MatrixTransform):
    def __init__(self, matrix: Any=None) -> None:
        if matrix is None:
            self._matrix = np.eye(4)
            return
        
        if isinstance(matrix, MatrixTransform):
            matrix = matrix._matrix
    
        matrix = canonical_homogeneous(matrix)
        if not np.linalg.det(matrix) > 1e-6:
            raise ValueError(f"Invalid homography matrix. Include reflection: {matrix}")
    
        self._matrix = matrix
    
    def inv(self) -> "Homography":
        return self.__class__(np.linalg.inv(self._matrix))
