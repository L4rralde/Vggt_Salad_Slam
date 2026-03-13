import torch

from .core import Algorithm, Solver


class GaussNewton(Algorithm):
    def __init__(self, solver: Solver):
        super().__init__(solver)

    def optimize(self, n_iter: int) -> None:
        for _ in range(n_iter):
            H, b = self.compute_h_and_b()
            d = torch.linalg.solve(H, -b).view(self.n, self.m) #Descent direction
            #Now we must compute 
            if torch.linalg.norm(d) < self.eps:
                break
            alpha = self.step_size(d)
            self.update(alpha * d)
