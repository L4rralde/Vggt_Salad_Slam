from typing import List, Any, Tuple, Type
from abc import ABC, abstractmethod

import torch


class Vertex(ABC):
    def __init__(self, idx: int, estimate: Any) -> None:
        self.idx: int = idx
        self.estimate: Any = estimate

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError


class Edge(ABC):
    def __init__(
        self,
        parent: Vertex,
        child: Vertex,
        transform: Any,
        information: torch.Tensor|None = None
    ):
        self.parent: Vertex = parent
        self.child: Vertex = child
        self.transform: Any = transform
        self.information: torch.Tensor|None = information


class Solver(ABC):
    @abstractmethod
    def residual(self, parent_est: Any, child_est: Any, meas: Any) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def edge_jacobian(self, edge: Edge) -> torch.Tensor:
        raise NotImplementedError

    def edge_residual(self, edge: Edge) -> torch.Tensor:
        return self.residual(
            edge.parent.estimate,
            edge.child.estimate,
            edge.transform
        )

    def loss(self, edges: List[Edge]) -> None:
        acc = 0.0
        for edge in edges:
            residual = self.edge_residual(edge)
            edge_loss = residual.T @ edge.information @ residual
            acc = acc + edge_loss
        return acc


class Algorithm(ABC):
    def __init__(self, solver: Solver, eps: float = 1e-6):
        self.solver: Solver = solver
        self.edges: List[Edge] = []
        self.vertices: List[Vertex] = []
        self.eps: float = eps
    
    @property
    def n(self) -> int:
        return len(self.vertices)

    @property
    def m(self) -> int:
        return len(self.edges[0].information)

    def compute_h_and_b(self) -> Tuple[torch.Tensor, torch.Tensor]:
        #1. Compute the errors
        n, m = self.n, self.m

        H = torch.zeros(n, n, m, m)
        b = torch.zeros(n, m)

        for edge in self.edges:
            residual = self.solver.edge_residual(edge)
            jacob_i, jacob_j = self.solver.edge_jacobian(edge)
            parent_idx = edge.parent.idx
            child_idx = edge.child.idx
            omega = edge.information
            H[parent_idx, parent_idx] += jacob_i.T @ omega @ jacob_i
            H[parent_idx, child_idx] += jacob_i.T @ omega @ jacob_j
            H[child_idx, parent_idx] += jacob_j.T @ omega @ jacob_i
            H[child_idx, child_idx] += jacob_j.T @ omega @ jacob_j

            b[parent_idx] += jacob_i.T @ omega @ residual
            b[child_idx] += jacob_j.T @ omega @ residual

        #Fix the first node
        H[0, :] = 0
        H[:, 0] = 0
        H[0, 0] = torch.eye(m)
        b[0] = 0

        H = H.permute(0, 2, 1, 3).reshape(n*m, n*m)
        b = b.view(n * m)
        return H, b

    def step_size(self, delta: torch.Tensor) -> float:
        #FUTURE. Use backtracking or something like that
        return 1.0

    def update(self, delta: torch.Tensor) -> None:
        for i, vertex in enumerate(self.vertices):
            vertex.update(delta[i])
        for edge in self.edges:
            edge.update()

    @abstractmethod
    def optimize(self, n_iter: int) -> None:
        raise NotImplementedError


class Optimizer:
    def __init__(
        self,
        solver: Type[Solver],
        algorithm: Type[Algorithm],
        vertices: List[Vertex] = [],
        edges: List[Edge] = [],
    ) -> None:
        self.solver: Solver = solver()
        self.vertices = vertices
        self.edges = edges
        self.algorithm: Algorithm = algorithm(self.solver)
        self.algorithm.vertices = self.vertices
        self.algorithm.edges = self.edges

    def append_vertex(self, vertex: Vertex) -> None:
        assert vertex.idx == len(self.vertices) #TODO. Allow having different idcs
        self.vertices.append(vertex)

    def append_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def optimize(self, n_iter) -> None:
        self.algorithm.optimize(n_iter)
