import numpy as np
from scipy.linalg import logm, expm


#Base on (MiT Spark Lab) VGGT-SLAM:
#VGGT-SLAM: Dense RGB SLAM Optimized on the SL(4) Manifold
#Src code: https://github.com/MIT-SPARK/gtsam_with_sl4/blob/8f0e9d3e9a697821ab9c3dece2189ce77ea0e776/gtsam/geometry/SL4.cpp
class SL4:
    def __init__(self, mat: np.ndarray) -> None:
        assert mat.shape == (4, 4)
        det = np.linalg.det(mat)
        if det <= 1e-6:
            raise ValueError(f"Matrix determinant must be positive for SL(4) normalization. Got det: {det:.4f}")
        self.mat: np.ndarray = mat / det**0.25

    def inv(self) -> "SL4":
        return SL4(np.linalg.inv(self.mat))

    def __matmul__(self, other: "SL4") -> "SL4":
        return SL4(self.mat @ other.mat)

    def Log(self) -> np.ndarray:
        log_mat = logm(self.mat)
        
        x12 = log_mat[0,0]
        x13 = log_mat[1,1] + x12
        x14 = -log_mat[3,3]
        return np.asarray([
            log_mat[0,1], log_mat[0,2], log_mat[0,3],
            log_mat[1,0], log_mat[1,2], log_mat[1,3],
            log_mat[2,0], log_mat[2,1], log_mat[2,3],
            log_mat[3,0], log_mat[3,1], log_mat[3,2],
            x12, x13, x14
        ])

    @classmethod
    def Exp(cls, x: np.ndarray) -> "SL4":
        assert x.shape == (15,)
        
        d11 = x[12]
        d22 = -x[12] + x[13]
        d33 = -x[13] + x[14]
        d44 = -x[14]

        mat = np.asarray([
            [d11, x[0], x[1], x[2]],
            [x[3], d22, x[4], x[5]],
            [x[6], x[7], d33, x[8]],
            [x[9], x[10], x[11], d44]
        ])

        return cls(expm(mat))

    @staticmethod
    def remove_reflection(mat: np.ndarray) -> np.ndarray:
        """
        Following official SL4 implementation in GTSAM:
        We can ensure the transformation does not reflect points
        by modifying it via SVD decomposition.
        Parameters:
            mat: Any 4x4 matrix with not null determinant.
        """
        assert mat.shape == (4,4)
        U, S, VH = np.linalg.svd(mat)
        det_UV = np.linalg.det(U @ VH)
        if det_UV < 0.0:
            print("[WARNING]. Found transformation with reflection")
            U[:, -1] *= -1
        corrected_mat = (U * S) @ VH
        return corrected_mat