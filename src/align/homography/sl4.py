from abc import ABC, abstractmethod

import torch


#Base on (MiT Spark Lab) VGGT-SLAM:
#VGGT-SLAM: Dense RGB SLAM Optimized on the SL(4) Manifold
#Src code: https://github.com/MIT-SPARK/gtsam_with_sl4/blob/8f0e9d3e9a697821ab9c3dece2189ce77ea0e776/gtsam/geometry/SL4.cpp

class LieGroup(ABC):
    @abstractmethod
    def inv(self) -> "LieGroup":
        raise NotImplementedError()

    @abstractmethod
    def Log(self) -> torch.Tensor:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def Exp(cls, tangent: torch.Tensor) -> "LieGroup":
        raise NotImplementedError()


def logm(mat: torch.Tensor) -> torch.Tensor:
    mat_complex = mat.to(torch.complex128)
    eigenvalues, eigenvectors = torch.linalg.eig(mat_complex)
    log_eigenvalues = torch.log(eigenvalues)
    log_mat_complex = (
        eigenvectors @
        torch.diag(log_eigenvalues) @
        torch.linalg.inv(eigenvectors)
    )
    return log_mat_complex


class SL4(LieGroup):
    ndof = 15
    def __init__(self, mat: torch.Tensor) -> None:
        assert mat.shape == (4, 4)
        det = torch.linalg.det(mat)
        if det <= 1e-6:
            raise ValueError(f"Matrix determinant must be positive for SL(4) normalization. Got det: {det:.4f}")
        self.mat: torch.Tensor = (mat / det**0.25).to(torch.float32)

    def inv(self) -> "SL4":
        return self.__class__(torch.linalg.inv(self.mat))

    def __matmul__(self, other: "SL4") -> "SL4":
        return self.__class__(self.mat @ other.mat)

    def Log(self) -> torch.Tensor:
        log_mat = (logm(self.mat).real)
        
        x12 = log_mat[0,0]
        x13 = log_mat[1,1] + x12
        x14 = -log_mat[3,3]
        
        # Use torch.stack, NOT torch.Tensor
        return torch.stack([
            log_mat[0,1], log_mat[0,2], log_mat[0,3],
            log_mat[1,0], log_mat[1,2], log_mat[1,3],
            log_mat[2,0], log_mat[2,1], log_mat[2,3],
            log_mat[3,0], log_mat[3,1], log_mat[3,2],
            x12, x13, x14
        ]).to(torch.float32)

    @classmethod
    def Exp(cls, x: torch.Tensor) -> "SL4":
        assert x.shape == (15,)
        
        d11 = x[12]
        d22 = -x[12] + x[13]
        d33 = -x[13] + x[14]
        d44 = -x[14]

        # Use torch.stack to preserve the autograd graph!
        mat = torch.stack([
            torch.stack([d11, x[0], x[1], x[2]]),
            torch.stack([x[3], d22, x[4], x[5]]),
            torch.stack([x[6], x[7], d33, x[8]]),
            torch.stack([x[9], x[10], x[11], d44])
        ])

        return cls(torch.linalg.matrix_exp(mat))

    @staticmethod
    def remove_reflection(mat: torch.Tensor) -> torch.Tensor:
        """
        Following official SL4 implementation in GTSAM:
        We can ensure the transformation does not reflect points
        by modifying it via SVD decomposition.
        Parameters:
            mat: Any 4x4 matrix with not null determinant.
        """
        assert mat.shape == (4,4)
        U, S, VH = torch.linalg.svd(mat)
        det_UV = torch.linalg.det(U @ VH)
        if det_UV < 0.0:
            print("[WARNING]. Found transformation with reflection")
            U[:, -1] *= -1
        corrected_mat = (U * S) @ VH
        return corrected_mat


class SL4Affine(SL4):
    ndof = 12
    def __init__(self, mat: torch.Tensor) -> None:
        assert torch.allclose(
            mat[3, :3],
            torch.zeros(3, dtype=mat.dtype, device=mat.device)
        )
        mask = torch.ones_like(mat)
        mask[3, :3] = 0
        mat = mat * mask
        super().__init__(mat)

    def Log(self) -> torch.Tensor:
        mask = torch.ones(15, dtype=torch.bool)
        mask[9:12] = False
        eps = super().Log()[mask]
        assert eps.shape == (12,)
        return eps

    @classmethod
    def Exp(cls, x: torch.Tensor) -> "SL4Affine":
        assert x.shape == (12,)
        sl4_eps = torch.cat((
            x[:-3],
            torch.zeros(3, dtype=x.dtype, device=x.device),
            x[-3:]
        ))
        return super().Exp(sl4_eps)
