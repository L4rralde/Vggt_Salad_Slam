from abc import ABC, abstractmethod
from typing import Type, Any
from dataclasses import replace

import numpy as np

from src.models.dtypes import Prediction
from .utils import get_pointmap, extr_to_homogeneous


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

    def transform_extrinsics(self, extr: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    

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

    def transform(self, pred: Prediction|np.ndarray) -> Prediction|np.ndarray:
        R = self._matrix[:3, :3]
        t = self._matrix[:3, 3]
        
        if isinstance(pred, np.ndarray):
            if pred.shape[-1] != 3:
                raise ValueError("Invalid input")
            
            return pred @ R.T + t
        
        pointmap = get_pointmap(pred)
        return replace(
            pred,
            pointmap=pointmap
        )


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
    
    def __repr__(self):
        M = self._matrix

        # Translation
        t = M[:3, 3]

        # sR block
        sR = M[:3, :3]

        # Uniform scale (assuming R is orthonormal)
        s = np.cbrt(np.linalg.det(sR))

        # Recover rotation
        if np.isclose(s, 0):
            R = np.full((3, 3), np.nan)
        else:
            R = sR / s

        return (
            f"{self.__class__.__name__}(\n"
            f"  s = {s:.6g},\n"
            f"  R =\n{np.array2string(R, precision=4, suppress_small=True)},\n"
            f"  t = {np.array2string(t, precision=4, suppress_small=True)}\n"
            f")"
        )
    
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

    @property
    def s(self) -> float:
        rot_scale = self._matrix[:3, :3]
        scale = np.linalg.det(rot_scale)**(1/3)
        return scale
    
    @property
    def R(self) -> np.ndarray:
        rot_scale = self._matrix[:3, :3]
        scale = np.linalg.det(rot_scale)**(1/3)
        rot = rot_scale/scale
        return rot

    @property
    def t(self) -> np.ndarray:
        return self._matrix[:3, 3]

    def transform_extrinsics(self, extr: np.ndarray) -> np.ndarray:
        sR = self._matrix[:3, :3]
        s = np.linalg.norm(sR[:, 0])

        new_extrinsics = s * extr_to_homogeneous(extr) @ np.linalg.inv(self._matrix)
        n, r, c = extr.shape
        if r == 3:
            new_extrinsics = new_extrinsics[:, :3]

        return new_extrinsics
    
    def transform(self, pred: Prediction|np.ndarray) -> Prediction|np.ndarray:
        sR = self._matrix[:3, :3]
        t = self._matrix[:3, 3]

        if isinstance(pred, np.ndarray):
            if pred.shape[-1] != 3:
                raise ValueError("Invalid input")
            return pred @ sR.T + t

        s = np.linalg.det(sR)**(1/3)
        new_depth = s * pred.depth
        new_extrinsics = s * pred.extrinsic @ np.linalg.inv(self._matrix)

        pointmap = get_pointmap(pred)
        new_pointmap = pointmap @ sR.T + t

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
    
    def transform(self, pred: Prediction|np.ndarray) -> Prediction|np.ndarray:
        #print("Warning! By the moment the Affine class only transforms pointmaps")

        A = self._matrix[:3, :3]
        t = self._matrix[:3, 3]


        if isinstance(pred, np.ndarray):
            if pred.shape[-1] != 3:
                raise ValueError("Invalid input")

            return pred @ A.T + t
        
        pointmap = get_pointmap(pred)
        new_pointmap = pointmap @ A.T + t
        

        return replace(
            pred,
            pointmap=new_pointmap
        )
    
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

        pointmap = get_pointmap(pred)
        new_pointmap = pointmap @ A.T + t
        
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

    def transform(self, pred: Prediction|np.ndarray) -> Prediction|np.ndarray:
        #print("Warning! By the moment the Homography class only transforms pointmaps")
        A = self._matrix[:3, :3]
        t = self._matrix[:3, 3]
        v = self._matrix[3, :3]

        if isinstance(pred, np.ndarray):
            if pred.shape[-1] != 3:
                raise ValueError("Invalid input")
            return (pred @ A.T + t) / 1 + pred @ v[..., None]

        pointmap = get_pointmap(pred)

        new_pointmap = pointmap @ A.T + t
        pers_scale = 1 + pointmap @ v[..., None] #(n, h, w, 3) @ (3, 1) = (n, h, w, 1)
        new_pointmap = new_pointmap / pers_scale

        new_pred = replace(
            pred,
            pointmap=new_pointmap
        )

        return new_pred
