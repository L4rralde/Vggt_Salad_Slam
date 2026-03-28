#Seeking for a linear transformation defined by an arbitrary homogeneus matrix 
# with 15 DoF such as $X_j = H_{j, i} X_i$
# For a python class we need to define its degreese of freedoms, an '@' operation,
# Inverse, identity and a tangent space of it. 
#
# This transformation is designed to work with 3D points. 
# Hence, Sim(3), SO(3) and SE(3) are special-case scenarios of the general 15-DoF one.

# Let's start with an abc class


from abc import ABC, abstractclassmethod
from typing import Any

import numpy as np


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

    @abstractclassmethod
    def __repr__(self) -> str:
        raise NotADirectoryError()
    
    @abstractclassmethod
    def asmatrix(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractclassmethod
    def __matmul__(self, other: "Transform") -> "Transform":
        raise NotImplementedError()
    
    @abstractclassmethod
    def tangent(self) -> object:
        raise NotImplementedError()

    def __call__(self, *args: Any) -> Any:
        #Transforms points
        raise NotImplementedError()


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
    """



class SamePerspectiveHomography(Transform):
    """
    Same as Homography, but the perspective vector is null, i.e. v = [0, 0, 0]
    Let X_i be a point cloud built from view v of reconstruction i.
    Let X_j be a pcd built from the same view but of another reconstruction j.
    We suppose, since they come from the same image, parallel lines are parralels
    and relative angles are the same on both.
    Hence we have:
    H = [
        sKR t
        0   1
    ]
    Those are 12 DoF.
    Still, there's a problem when all we have are planes. So sad.
    This class probably WILL be implemented.
    """


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
    """


class Sim3(Transform):
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


class SE3(Transform):
    """
    Special Euclidean (n=3) Lie Group
    H = [
        R t
        0 1
    ]
    6 DoF. Used in metric (both pcds' scale is known) reconstruction.
    When using distance sensors (LiDARs, Time of flight sensors, etc)
    """


class SO3(Transform):
    """
    Special Orhogonal (n=3) lie group or 3D rotation group
    H = [
        R 0
        0 1
    ]
    3 DoF
    """


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
        self.s = s

    @classmethod
    def identity(cls) -> "ScaleTransform":
        return cls(1.0)

    def inv(self) -> "ScaleTransform":
        return ScaleTransform(1/self.s)

    def __copy__(self) -> "ScaleTransform":
        return ScaleTransform(self.s)

    def __repr__(self) -> str:
        return f"ScaleTransform(s = {self.s:.4f})"

    def asmatrix(self) -> np.ndarray:
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] *= self.s
        return mat

    def __matmul__(self, other: "ScaleTransform") -> "ScaleTransform":
        raise ScaleTransform(self.s * other.s)

    def tangent(self) -> np.ndarray:
        return np.ndarray([max(1e-6, self.s)], dtype=np.float32)

    def __call__(self, x: np.ndarray) -> Any:
        return x[:3, ...] * self.s

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

