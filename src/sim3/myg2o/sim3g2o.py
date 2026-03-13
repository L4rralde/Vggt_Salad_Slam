import pypose as pp
import torch


from .core import Vertex, Edge, Solver, Algorithm, Optimizer


class Sim3Vertex(Vertex):
    def __init__(self, idx: int, estimate: pp.Sim3):
        self.idx = idx
        self.estimate = pp.Sim3(estimate) #Create a copy

    def update(self, delta: torch.Tensor) -> None:
        delta = pp.sim3(delta)
        self.estimate = delta.Exp() @ self.estimate

    def copy(self) -> "Sim3Vertex":
        return Sim3Vertex(self.idx, pp.Sim3(self.estimate))


class Sim3Edge(Edge):
    def __init__(
        self,
        parent: Sim3Vertex,
        child: Sim3Vertex,
        transform: pp.Sim3,
        information: torch.Tensor|None = None
    ):
        self.parent: Sim3Vertex = parent
        self.child: Sim3Vertex = child
        self.transform: pp.Sim3 = pp.Sim3(transform) #Create a copy
        self.information: torch.Tensor = (
            information if information is not None
            else torch.eye(7)
        )

    def copy(self) -> "Sim3Edge":
        return Sim3Edge(
            self.parent.copy(),
            self.child.copy(),
            pp.Sim3(self.transform),
            self.information.clone()
        )

    def update(self) -> None:
        self.transform = self.parent.estimate.Inv() @ self.child.estimate


class Sim3Solver(Solver):
    def residual(self, parent_est: pp.Sim3, child_est: pp.Sim3, meas: pp.Sim3) -> torch.Tensor:
        prediction = parent_est.Inv() @ child_est
        return (meas.Inv() @ prediction).Log()

    def edge_jacobian(self, edge: Sim3Edge) -> torch.Tensor:
        parent_est = edge.parent.estimate.clone().requires_grad_(True)
        child_est = edge.child.estimate.clone().requires_grad_(True)
        meas = edge.transform

        zero_perturbation = pp.sim3(torch.zeros(7)) #Lie algebra

        parent_j = torch.autograd.functional.jacobian(
            lambda d: self.residual(d.Exp() @ parent_est, child_est, meas),
            zero_perturbation
        )
        
        child_j = torch.autograd.functional.jacobian(
            lambda d: self.residual(parent_est, d.Exp() @ child_est, meas),
            zero_perturbation
        )

        return parent_j, child_j


class Sim3Optimizer(Optimizer):
    def __init__(self, algorithm: Algorithm):
        super().__init__(Sim3Solver, algorithm)
