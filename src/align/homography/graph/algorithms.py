import torch

from .core import Algorithm


class GaussNewton(Algorithm):
    def optimize(self, n_iter: int) -> None:
        for i in range(n_iter):
            H, b = self.compute_h_and_b()
            d = torch.linalg.solve(H, -b).view(self.n, self.m) #Descent direction
            print(f"{i+1}. Loss: {self.loss()}. d.norm: {torch.linalg.norm(d)}")
            #Now we must compute 
            if torch.linalg.norm(d) < self.eps:
                break
            alpha = self.step_size(d)
            self.update_vertices(alpha * d)
        self.update_edges()
