#Seeking for a linear transformation defined by an arbitrary homogeneus matrix 
# with 15 DoF such as $X_j = H_{j, i} X_i$
# For a python class we need to define its degreese of freedoms, an '@' operation,
# Inverse, identity and a tangent space of it. 
#
# This transformation is designed to work with 3D points. 
# Hence, Sim(3), SO(3) and SE(3) are special-case scenarios of the general 15-DoF one.

# Let's start with an abc class

from abc import ABC, abstractclassmethod
from typing import Any, List

import numpy as np
import torch
from scipy.spatial.transform import Rotation as scipy_R
import pypose as pp

from homography.sl4 import SL4


NP_DTYPE = np.float32
TORCH_DTYPE = torch.float32


def rotmat_to_quat(rotmat: np.ndarray) -> np.ndarray:
    return scipy_R.from_matrix(rotmat).as_quat()


class Transform(ABC):
    @abstractclassmethod
    def identity(cls) -> "Transform":
        raise NotImplementedError()

    @abstractclassmethod
    def inv(self) -> "Transform":
        raise NotImplementedError()
    
    @abstractclassmethod
    def __copy__(self) -> "Transform":
        raise NotImplementedError()

    def copy(self):
        return self.__copy__()

    @abstractclassmethod
    def __repr__(self) -> str:
        raise NotADirectoryError()
    
    @abstractclassmethod
    def as_matrix(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractclassmethod
    def __matmul__(self, other: "Transform") -> "Transform":
        raise NotImplementedError()

    def __eq__(self, other: "Transform") -> "Transform":
        return np.array_equal(self.as_matrix(), other.as_matrix())

    @staticmethod
    def all_close(a: "Transform", other: "Transform", rtol: float=1e-05, atol: float=1e-08) -> bool:
        return np.allclose(
            a.as_matrix(),
            other.as_matrix(),
            rtol=rtol,
            atol=atol
        )
    
    @abstractclassmethod
    def tangent(self) -> object:
        raise NotImplementedError()

    @abstractclassmethod
    def from_tangent(tangent: object) -> "Transform":
        raise NotImplementedError()

    @abstractclassmethod
    def __call__(self, *args: Any) -> Any:
        #Transforms points
        raise NotImplementedError()


def all_close(a: Transform, b: Transform, rtol: float=1e-05, atol: float=1e-08) -> bool:
    assert type(a) == type(b)
    return type(a).all_close(a, b, rtol=rtol, atol=atol)


class Homography(Transform):
    """
    In the context of 3D projections, an homography is a linear
    projection described by 15 deegres of freedom.
    Here I use the following convention:

    H = [
        sKR t
        v^T 1
    ]
    s: scale factor. s in R+
    K: intrinsics-like matrix:
        K = [
            fx gamma cx
            0  fy    cy
            0  0     1
        ]
        K is a 3x3 matrix with 6 DoF. 
    R: Rotation matrix. R in SO(3).
        SO(3) matrices have 3 DoF. 3 angles are enough to describe 3D rotations.
    t: Translation vector. t in R^3. 3 DoF.
    v: Perspective vector. v in R^3, hence 3 DoF.
        This modifies relative angles between lines.
        For instance, if you are in a toll road, it ends
        like a triangle, but from  the upper-view perspective,
        it is different.
    summing up, there are 15 DoF.

    However, To estimate a trnasformation of this class, 
    we need at least 5 non linear independent points.
    When using only points from planes, the solution is not stable.
    Recontruction of planes is really common in real world environments.
    Take walls as an example.
    By the moment, this class probably won't be implemented.

    Parameters:
        mat: Any 4x4 matrix with positive determinant
    """
    def __init__(
        self,
        mat: np.ndarray
    ) -> None:
        assert mat.shape == (4, 4)
        mat = mat/mat[3, 3]
        assert np.linalg.det(mat[:3, :3]) > 1e-6
        self.mat: np.ndarray = mat
        
        self.perspective: np.ndarray = self.mat[3, :3]
        self.translation: np.ndarray = self.mat[:3, 3]
        #A = sKR. A^{-1} = 1/s R^T K^{-1} = R^T (K^{-1}/s): QR Decomposition
        R_t, sK_inv = np.linalg.qr(self.mat[:3, :3])
        self.rotation = np.transpose(R_t)
        sK = np.linalg.solve(sK_inv, np.eye(3)) #Instead of np.linalg.inv
        self.scale = sK[2, 2]
        self.K = sK/self.scale

    @classmethod
    def identity(cls) -> "Homography":
        return cls(np.eye(4, dtype=NP_DTYPE))
    
    @classmethod
    def from_mat(cls, mat: np.ndarray) -> "Homography":
        return cls(mat)

    def inv(self) -> "Homography":
        return Homography(np.linalg.inv(self.mat))

    def __copy__(self) -> "Homography":
        return Homography(self.mat.copy())
    
    def __repr__(self) -> str:
        return f"Homography(mat: {self.mat.flatten()})"

    def as_matrix(self) -> np.ndarray:
        return self.mat
    
    def __matmul__(self, other: "Homography") -> "Homography":
        return Homography(self.mat @ other.mat)

    def __eq__(self, other: "Homography") -> bool:
        return np.array_equal(self.mat, other.mat)

    def tangent(self) -> torch.Tensor:
        return torch.Tensor(SL4(self.mat).Log()).to(TORCH_DTYPE)

    @classmethod
    def from_tangent(cls, tangent: torch.Tensor) -> "Affine":
        sl4_mat = SL4.Exp(tangent.cpu().numpy()).mat
        homography_mat = sl4_mat/sl4_mat[3, 3]
        return cls(homography_mat)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        original_shape = x.shape
        assert len(original_shape) <= 2
        if len(x.shape) == 1:
            x = x[None, ...]
        n, d = x.shape
        assert d == 3

        A = self.mat[:3, :3]
        t = self.mat[:3, 3]
        v = self.mat[3, :3]

        p = x @ A.T + t #Shape (n, 3)
        w = x @ v + 1.0 # shape (n,)
        w = np.repeat(np.expand_dims(w, axis=1), 3, axis=1) #shape (n, 3)
        return (p * w).reshape(original_shape)


class Affine(Transform):
    """
    Same as Homography, but the perspective vector is null, i.e. v = [0, 0, 0]
    Let X_i be a point cloud built from view v of reconstruction i.
    Let X_j be a pcd built from the same view but of another reconstruction j.
    We suppose, since they come from the same image, parallel lines are parralels
    and relative angles are the same in both.
    Hence we have:
    H = [
        sKR t
        0   1
    ]
    Those are 12 DoF.
    Still, there's a problem when all we have are planes. So sad.
    This class probably WILL be implemented.
    Since [skR | t] is a 3x4 matrix with 12 DoF. Let's use it as input
    Parameters:
        mat: 3x4 mat. first 3 rows of the homogeneous matrix.
    """
    def __init__(self, mat: np.ndarray) -> None:
        assert mat.shape == (3, 4)
        assert np.linalg.det(mat[:3, :3]) > 1e-6
        self.mat = np.zeros((4, 4), dtype=mat.dtype)
        self.mat[:3] = mat
        self.mat[3,3] = 1.0

        self.translation = self.mat[:3, 3]

        #A = sKR. A^{-1} = 1/s R^T K^{-1} = R^T (K^{-1}/s): QR Decomposition
        R_t, sK_inv = np.linalg.qr(self.mat[:3, :3])
        self.rotation = np.transpose(R_t)
        sK = np.linalg.solve(sK_inv, np.eye(3)) #Instead of np.linalg.inv
        self.scale = sK[2, 2]
        self.K = sK/self.scale
    
    @classmethod
    def identity(cls) -> "Affine":
        return cls(np.eye(3, M=4, dtype=NP_DTYPE))

    def inv(self) -> "Affine":
        pass
        #[A t| 0 1][A' t'| 0 1] = [AA' At' + t | 0 1]
        # AA' = A'A => A' = A^{-1}. A is 3x3 9DoF.
        # At' + t = 0 => t' = -A^{-1}t
        A = self.mat[:3, :3]
        t = self.mat[:3, 3]
        A_prime = np.linalg.inv(A)
        t_prime = - A_prime @ t
        mat = np.zeros((3, 4), dtype=self.mat.dtype)
        mat[:3, :3] = A_prime
        mat[:3, 3] = t_prime
        return Affine(mat)

    def __copy__(self) -> "Affine":
        return Affine(self.mat.copy()[:3])

    def __repr__(self) -> str:
        return f"Affine(mat: {self.mat.flatten()})"

    def as_matrix(self) -> np.ndarray:
        return self.mat

    def __matmul__(self, other: "Affine") -> "Affine":
        result_mat = (self.mat @ other.mat)[:3]
        return Affine(result_mat)

    def tangent(self) -> torch.Tensor:
        return torch.Tensor(SL4(self.mat).Log()).to(TORCH_DTYPE)

    @classmethod
    def from_tangent(cls, tangent: torch.Tensor) -> "Affine":
        sl4_mat = SL4.Exp(tangent.cpu().numpy()).mat
        homography_mat = sl4_mat/sl4_mat[3, 3]
        return cls(homography_mat[:3])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        original_shape = x.shape
        assert len(original_shape) <= 2
        A = self.mat[:3, :3]
        t = self.mat[:3, 3]
        if len(x.shape) == 1:
            x = x[None, ...]
        n, d = x.shape
        assert d == 3
        #A is 3x3. x is nx3. A x^T is 3 xn. Hence (A x^T)^T = x A^t
        return (x @ A.T + t).reshape(original_shape)


class VggtSlam2Transform(Transform):
    """
    Consider we are not aligning pcds using world's frame, but
    the projection to those into the cameras, e.g.,
    Let E (extrinsics) be the world to cam transformation. 
        E_iX_i = H_{i,j}E_jX_j
    Here we have they share the same
    orientation, origin and perspective.
    H = [
        sK 0
        0 1
    ]
    Only 6 DoF.
    However, this computes the transformation of projected PCDs,
    if we need PCDs in world coordinates this won't work, we still need
    to compute R and t.
    When does this work? When R and t can be trivially predicted.
    Properly said, K is not the intrinsic matrix, but is upper triangular.
    Let's allow any upper triangular matrix. sK is 3x3 6 DoF.
    Parameters:
        sK_mat: np.ndarray of shape 3x3 and upper triangular
    """
    def __init__(self, sK_mat: np.ndarray) -> None:
        assert sK_mat.shape == (3, 3)
        assert np.allclose(sK_mat, np.triu(sK_mat))
        self.sK_mat: np.ndarray = np.triu(sK_mat)
        self.scale: float = sK_mat[2, 2]
        self.K: np.ndarray = sK_mat/self.scale

    @classmethod
    def identity(cls) -> "VggtSlam2Transform":
        return cls(np.eye(3, dtype=NP_DTYPE))

    def inv(self) -> "VggtSlam2Transform":
        #[Triu | 0 \\ 0 1][Triu' | 0 \\ 0 1] = [TriuTriu' | 0 \\ 0 | 1]
        sK_inv = np.linalg.solve(self.sK_mat, np.eye(3))
        return VggtSlam2Transform(sK_inv)

    def __copy__(self) -> "VggtSlam2Transform":
        return VggtSlam2Transform(self.sK_mat.copy())

    def __repr__(self) -> str:
        return f"VggtSlam2Transform(sk: {self.sK_mat.flatten()})"

    def as_matrix(self) -> np.ndarray:
        mat = np.eye(4, dtype=self.sK_mat.dtype)
        mat[:3, :3] = self.sK_mat
        return mat

    def __matmul__(self, other: "VggtSlam2Transform") -> "VggtSlam2Transform":
        return VggtSlam2Transform(self.sK_mat @ other.sK_mat)

    def tangent(self) -> torch.Tensor:
        return torch.Tensor(SL4(self.as_matrix()).Log()).to(TORCH_DTYPE)

    @classmethod
    def from_tangent(cls, tangent: torch.Tensor) -> "VggtSlam2Transform":
        sl4_mat = SL4.Exp(tangent.cpu().numpy()).mat
        homography_mat = sl4_mat/sl4_mat[3, 3]
        assert np.allclose(homography_mat[:3, 3], np.zeros(3))
        assert np.allclose(homography_mat[3, :3], np.zeros(3))
        return cls(homography_mat[:3, :3])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        original_shape = x.shape
        assert len(original_shape) <= 2
        if len(original_shape) == 1:
            x = x[None, ...]
        n, d = x.shape
        assert d == 3
        return (x @ self.sK_mat.T).reshape(original_shape)


class SO3(Transform):
    """
    Special Orhogonal (n=3) lie group or 3D rotation group
    H = [
        R 0
        0 1
    ]
    3 DoF
    """
    def __init__(self, rotation: np.ndarray|List) -> None:
        self.rotation: np.ndarray = np.empty((3, 3), dtype=NP_DTYPE)
        if isinstance(rotation, np.ndarray):
            self.rotation = rotation
        else:
            self.rotation = scipy_R(rotation).as_matrix()

    @property
    def quaternion(self) -> np.ndarray:
        """
        Gets the quaternion q = (x, y, z, w)
        """
        return rotmat_to_quat(self.rotation)

    @classmethod
    def identity(cls) -> "SO3":
        rotation = np.eye(3, dtype=NP_DTYPE)
        return cls(rotation)

    def inv(self) -> "SO3":
        return SO3(np.transpose(self.rotation))

    def __copy__(self) -> "SO3":
        return SO3(self.rotation.copy())

    def __repr__(self) -> str:
        return f"SO3(q: {self.quaternion})"

    def as_matrix(self) -> np.ndarray:
        mat = np.eye(4, dtype=NP_DTYPE)
        mat[:3, :3] = self.rotation
        return mat

    def __matmul__(self, other: "SE3") -> "SE3":
        return SO3(self.rotation @ other.rotation)

    def tangent(self) -> torch.Tensor:
        return pp.SO3(self.quaternion).Log()

    @classmethod
    def from_tangent(cls, tangent: torch.Tensor) -> "SO3":
        rotation = scipy_R(tangent.Exp().rotation()).as_matrix()
        return cls(rotation)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        x' = Rx
        x = np.ndarray of shape (3,) or (n, 3)
        If x.shape == (n, 3). Then, we must transpose x to have a valid matmul.
        However, we want the shape of x' to be the same as x. So we transpose the result.
        This is x' = (R x^T)^T = x R^T
        If x is of shape 3, x^T = x, x'^T = T.   x' = x'^T = R x^T = R x
        """
        original_shape = x.shape
        assert len(original_shape) <= 2
        if len(x.shape) == 1:
            x = x[None, ...]
        n, d = x.shape
        assert d == 3
        result = x @ self.rotation.T
        if n == 1:
            result = np.squeeze(x, axis=0)
        return result


class SE3(SO3):
    """
    Special Euclidean (n=3) Lie Group
    H = [
        R t
        0 1
    ]
    6 DoF. Used in metric (both pcds' scale is known) reconstruction.
    When using distance sensors (LiDARs, Time of flight sensors, etc)
    """
    def __init__(self, rotation: Any | List, translation: np.ndarray) -> None:
        super().__init__(rotation)
        self.translation: np.ndarray = translation

    @classmethod
    def identity(cls) -> "SO3":
        R = np.eye(3, dtype=NP_DTYPE)
        t = np.zeros(3, dtype=NP_DTYPE)
        return SE3(R, t)

    def inv(self) -> "SO3":
        inv_R = np.transpose(self.rotation)
        inv_t = -inv_R @ self.translation
        return SE3(inv_R, inv_t)

    def __copy__(self) -> "SO3":
        return SE3(
            self.rotation.copy(),
            self.translation.copy()
        )

    def __repr__(self) -> str:
        return f"SE3(quat: {self.quaternion}, t: {self.translation})"

    def as_matrix(self) -> np.ndarray:
        mat = super().as_matrix() #SO(3) matrix
        mat[:3, :3] = self.rotation
        return mat

    def __matmul__(self, other: "SE3") -> "SE3":
        new_R = self.rotation @ other.rotation
        new_t = self.rotation @ other.translation + self.translation
        return SE3(new_R, new_t)

    def tangent(self) -> torch.Tensor:
        data = np.concatenate([self.translation, self.quaternion])
        return pp.SE3(data).Log()

    @classmethod
    def from_pypose(cls, pp: torch.Tensor) -> "SE3":
        rotation = scipy_R(pp.rotation()).as_matrix()
        translation = pp.translation()
        return cls(rotation, translation)

    @classmethod
    def from_tangent(cls, tangent: torch.Tensor) -> "SE3":
        pp_transform = tangent.Exp()
        return cls.from_pypose(pp_transform)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return super().__call__(x) + self.translation



class Sim3(SE3):
    """
    The Similiraty(n=3) Lie group.
    H = [
        sR t
        0  1
    ]
    7 DoF.
    Works even when having only planes.
    3 points are enough to get an estimation
    """
    def __init__(
        self,
        scale: float,
        rotation: Any | List, 
        translation: np.ndarray
    ) -> None:
        super().__init__(rotation, translation)
        self.scale = scale

    @classmethod
    def identity(cls) -> "Sim3":
        R = np.eye(3, dtype=NP_DTYPE)
        t = np.zeros(3, dtype=NP_DTYPE)
        s = 1.0
        return Sim3(s, R, t)

    @staticmethod
    def from_pypose(sim3: pp.Sim3) -> "Sim3":
        rot = scipy_R(sim3.rotation()).as_matrix()
        return Sim3(
            sim3.scale().item(),
            rot,
            sim3.translation().numpy()
        )

    def aspypose(self) -> pp.Sim3:
        data = np.concatenate([
            self.translation,
            scipy_R.from_matrix(self.R).as_quat(),
            np.array(self.scale).reshape((1,))
        ])
        return pp.Sim3(data)

    def inv(self) -> "Sim3":
        inv_s = 1.0/self.scale
        inv_R = self.rotation.T
        inv_t = -inv_s * (inv_R @ self.translation)
        return Sim3(inv_s, inv_R, inv_t)

    def __copy__(self) -> "Sim3":
        return Sim3(
            self.scale,
            self.rotation.copy(),
            self.translation.copy()
        )

    def __repr__(self) -> str:
        return f"Sim3(s: {self.scale:.4f}, quat: {self.quaternion}, t: {self.translation})"

    def as_matrix(self) -> np.ndarray:
        mat = super().as_matrix()
        mat[:3, :3] *= self.scale
        return mat

    def __matmul__(self, other: "Sim3") -> "Sim3":
        return Sim3(
            self.scale * other.scale,
            self.rotation @ other.rotation,
            (self.scale * self.rotation @ other.translation) + self.translation
        )

    def tangent(self) -> torch.Tensor:
        data = np.concatenate([
            self.translation,
            self.quaternion,
            np.array(self.scale).reshape((1,))
        ])
        return pp.Sim3(data).Log()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (self.scale * SO3.__call__(self, x)) + self.translation

# Assumption #1. Depth maps are robust, so we can estimate s directly using depthmaps.
#Hence, we treat s as constant in estimation methods.
    
class ScaleTransform(Transform):
    """
    H = [
        sI 0
        0  1
    ]
    s in R+
    """
    def __init__(self, s: float) -> None:
        assert s > 0
        self.scale = s

    @classmethod
    def identity(cls) -> "ScaleTransform":
        return cls(1.0)

    def inv(self) -> "ScaleTransform":
        return ScaleTransform(1/self.scale)

    def __copy__(self) -> "ScaleTransform":
        return ScaleTransform(self.scale)

    def __repr__(self) -> str:
        return f"ScaleTransform(s = {self.scale:.4f})"

    def as_matrix(self) -> np.ndarray:
        mat = np.eye(4, dtype=NP_DTYPE)
        mat[:3, :3] *= self.scale
        return mat

    def __matmul__(self, other: "ScaleTransform") -> "ScaleTransform":
        return ScaleTransform(self.scale * other.scale)

    def tangent(self) -> torch.Tensor:
        return torch.Tensor([np.log(self.scale)]).to(TORCH_DTYPE)

    @classmethod
    def from_tangent(cls, tangent: torch.Tensor) -> "ScaleTransform":
        s = np.exp(tangent.cpu()[0].item())
        return cls(s)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale



# Special case # 1. K matrix (intrinsics) are the same for all views.
# Assumption #2.
# Say we capture all images using the same camera. Assume no degradation or other
# effects happened to modify the real intrinsics. 
# Hence, we can estimate K using more robusts methods.
# Let $K_r$ the true/real/choosen reference intrinscis.
# For such $K_r$ we have a set of predicted pcds.
# Consider the case where we want to align two projected (wrt their cameras frames)
# points built from the same image.
# This is the
#
#   H = [
#    sK 0
#    0  1
#   ]
# Case found in VggtSlam 2.0
# Using intrinsics, extrinsics and intrinsics, this means:
#
#    D_i(u,v) K_i^{-1} p(u,v) = s K_{i,j} D_j(u,v) K_j^{-1} p(u,v)
# p are pixels. Same image, same pixels. So we drop it of the equality. 
# Let's solve for K_{i,j}
# D_i(u,v)/(s * D_j(u, v)) K_i^{-1} K_j = K_{i,j}
# But we assume we can estimate s using only the depthmaps. Hence D_i(u,v)/(s * D_j(u, v)) = 1
# Finally: K_{i, j} = K_i^{-1} K_j
# This is, we find another way to estimate K_{i, j} 
# Now, dropping s and K, we removed 6 variables of the equation. 6 of up to 15.
# Up to 9 DoFs that can be found with 3 linearly independent points, e.g, from a plane.


# This opens the door for using the full 3D homography matrix, If we initialize s and K,
# then we probably can rely on iterative methods to find a feasible solution.
# Nonetheless, using more variables cause greater drift when using less.
# Also, will make the optimization graph slower.
# In VGGT-SLAM 2.0 they opted to add SE(3) transformations between submaps to reduce the 
# drift, but this increase the size of the graph by a factor of the order of 10.
# Goal, to implement all and see the pros and cons of each.
# Clearly, for 3D point registering, SO(3) is useless.

# Actually, I know we can initialize 12 DoF and use an iterative method
# s is estimated from depth maps. Let's use hubber or L1.
# K is estimated from intrinsics predictions. K_{i,k} = K_i^{-1} K_j
# Consider Sim(3) transformations. Set s as known/constant.
# Using Cam to world  E^{-1} = [R' | t] (poses) predictions, we can estimate a SE(3)
# transformation. Hence we find an initial guess of R, t
# a total of 12 DoF.
# I don't know of a method to initialize the perspective. We can say the perspective remains constan

