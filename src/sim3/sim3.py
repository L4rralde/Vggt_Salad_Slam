
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

    def copy(self) -> "Sim3":
        return Sim3(self.s, self.R.copy(), self.t.copy())

    def __copy__(self) -> "Sim3":
        return self.copy()

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

    def nthroot(self, n: int) -> "Sim3":
        assert isinstance(n, int), "n must be integer"
        if n == 0:
            return Sim3.identity()
        if n < 0:
            n *= -1
            sim3 = self.inv()
        else:
            sim3 = self
        if n == 1:
            return sim3

        sR_pow = np.eye(3, dtype=sim3.R.dtype)
        root_s = sim3.s ** (1/n)
        root_R = (Rotation.from_matrix(sim3.R) ** (1/n)).as_matrix()

        sR_acc = np.zeros((3, 3), dtype=sim3.R.dtype)
        for _ in range(n):
            sR_acc = sR_acc + sR_pow
            sR_pow = sR_pow @ (root_s * root_R)

        #sR_acc = I + sR + (sR)^2 + ... + (sR)^n-1
        #s_pow = s^|n|
        #R_pow = R^|n|
        
        root_t = np.linalg.solve(sR_acc, sim3.t)
        root = Sim3(root_s, root_R, root_t)
    
        root_to_n = Sim3.identity()
        for _ in range(n):
            root_to_n = root_to_n @ root
        print(f"nthroot computatin. Input: {self}, rooth: {root}, root_to_n={root_to_n}")

        return root
