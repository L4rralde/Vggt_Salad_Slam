
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


class Sim3:
    def __init__(self, s: float, R: np.ndarray, t: np.ndarray):
        self.s = s
        self.R = R
        self.t = t

    @staticmethod
    def identity() -> "Sim3":
        return Sim3(
            1.0,
            np.eye(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32)
        )

    def __repr__(self) -> str:
        R = Rotation.from_matrix(self.R)
        s_str = f"{self.s:.3f}"
        rot_vec_str = ", ".join([f"{x:.4f}" for x in R.as_rotvec()])
        t_str = ", ".join([f"{x:.4f}" for x in self.t])
        return f"Sim3(s={s_str}, rot_vec=[{rot_vec_str}], t=[{t_str}])"

    def astuple(self) -> Tuple[float, np.ndarray, np.ndarray]:
        return self.s, self.R, self.t

    def asmatrix(self) -> np.ndarray:
        mat = np.eye(4).astype(np.float32)
        mat[:3, :3] = self.s * self.R
        mat[:3, 3] = self.t
        return mat

    def inv(self) -> "Sim3":
        return Sim3(
            1.0/self.s,
            self.R.T,
            -(self.R.T @ self.t)/self.s
        )

    def __matmul__(self, other: "Sim3") -> "Sim3":
        return Sim3(
            self.s * other.s,
            self.R @ other.R,
            self.s * self.R @ other.t + self.t
        )
