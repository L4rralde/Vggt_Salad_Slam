from typing import(
    List, Set, Sequence, Optional, Tuple
)

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr


class ScaleGraph:
    def __init__(self) -> None:
        # Edge structure: (parent_i, child_j, measurement, weight)
        self.edges: List[Tuple[int, int, float, float]] = []
        self._known_nodes: Set[int] = set()

    @property
    def num_nodes(self) -> int:
        return len(self._known_nodes)
    
    def add_measurement(
        self,
        i: int,
        j: int,
        measurement: float,
        weight: float=1.0
    ) -> None:
        """
        i: parent id
        j: child id
        measurement: s_{i,j}
        weight: edge's weight
        """
        if measurement <= 1e-6:
            raise ValueError(f"All measurements must be possitive. Got {measurement}")
        self.edges.append((i, j, measurement, weight))
        self._known_nodes.add(i)
        self._known_nodes.add(j)
    
    def _validate_contiguous_nodes(self) -> None:
        """Helper to ensure node IDs are 0, 1, ..., num_nodes-1."""
        if self.num_nodes > 0 and max(self._known_nodes) >= self.num_nodes:
            raise RuntimeError(
                "Node IDs must be contiguous integers starting from 0. "
                f"Max node ID is {max(self._known_nodes)} but graph has {self.num_nodes} nodes."
            )
    
    def _compute_residuals(self, free_values: List[float]) -> np.ndarray:
        """
        values: Ordered list (node 0, node 1, ...) of log(s_i) values
        """
        if len(free_values) + 1 != len(self._known_nodes):
            raise RuntimeError("len mismatch")
        
        full_x = np.empty(len(self._known_nodes))
        full_x[0] = 0
        full_x[1:] = free_values

        residuals = [
            np.sqrt(w) * (-np.log(meas) + full_x[j] - full_x[i])
            for i, j, meas, w in self.edges
        ]
        residuals = np.array(residuals)
        return residuals
    
    def optimize(self, values: List[float]) -> Sequence[float]:
        """
        values: Ordered list (node 0, node 1, ...) of s_i values
        """
        self._validate_contiguous_nodes()

        if len(values) != len(self._known_nodes):
            raise RuntimeError("len mismatch")
        if values[0] != 1.0:
            raise RuntimeError("Gauge conflict")
    
        initial_x = np.log(values[1:])
        result = least_squares(
            fun=self._compute_residuals,
            x0=initial_x,
            method='lm'          # Notice: 'args' is completely gone
        )
        optimized_x = np.zeros(self.num_nodes)
        optimized_x[1:] = result.x

        return np.exp(optimized_x)


class SparseScaleGraph(ScaleGraph):
    def optimize(self, initial_values: Optional[List[float]] = None) -> Sequence[float]:
        """
        Solves the log-scale linear least squares system Ax = b.
        
        Note: `initial_values` is optional because a linear least-squares problem 
        does not require an initial guess to converge to the global optimum.
        """
        if self.num_nodes == 0:
            return []
        
        self._validate_contiguous_nodes()

        # Handle trivial case with only anchor node
        if self.num_nodes == 1:
            return np.array(1.0)

        # Construct sparse linear system Ax = b
        # Columns correspond to free variables: x_1, x_2, ..., x_{N-1}
        # Column index for node k (k > 0) is (k - 1)
        num_free_vars = self.num_nodes - 1
        num_edges = len(self.edges)

        rows = []
        cols = []
        data = []
        b = np.zeros(num_edges)

        for edge_idx, (i, j, meas, weight) in enumerate(self.edges):
            sqrt_w = np.sqrt(weight)
            log_meas = np.log(meas)
            
            # Equation: sqrt(w) * (x_j - x_i) = sqrt(w) * log(meas)
            b[edge_idx] = sqrt_w * log_meas

            # Add x_j term (+1) if j is a free variable (j > 0)
            if j > 0:
                rows.append(edge_idx)
                cols.append(j - 1)
                data.append(sqrt_w)

            # Add x_i term (-1) if i is a free variable (i > 0)
            if i > 0:
                rows.append(edge_idx)
                cols.append(i - 1)
                data.append(-sqrt_w)

        # Build CSR matrix
        A = csr_matrix((data, (rows, cols)), shape=(num_edges, num_free_vars))

        # Solve Ax = b using sparse least squares
        # lsqr returns a tuple where index 0 is the solution array
        solution = lsqr(A, b)[0]

        # Reconstruct full x array (anchoring node 0 to log(1.0) = 0.0)
        optimized_x = np.zeros(self.num_nodes)
        optimized_x[1:] = solution

        return np.exp(optimized_x)
    
# ==========================================
# Example Usage & Verification
# ==========================================
if __name__ == "__main__":
    # 1. Define Ground Truth Scales (Node 0 is fixed to 1.0)
    true_scales = np.array([1.0, 2.5, 0.4, 10.0])
    N = len(true_scales)

    optimizer = ScaleGraph()

    # 2. Simulate noisy measurements: s_hat_{i,j} ≈ s_i / s_j
    np.random.seed(42)
    edges_to_create = [(0, 1), (1, 2), (2, 3), (0, 3), (1, 3)]

    noise = np.random.normal(1.0, 0.05)
    for i, j in edges_to_create:
        ideal_s_hat = true_scales[i] / true_scales[j]
        noisy_s_hat = ideal_s_hat * noise # 5% multiplicative noise
        
        optimizer.add_measurement(i, j, noisy_s_hat)

    # 3. Provide a rough initial estimation
    bad_initial_guess = np.array([1.0, 1.5, 1.0, 6.0])

    # 4. Optimize
    recovered_scales = optimizer.optimize(bad_initial_guess)

    print("Ground Truth Scales: ", true_scales)
    print("Initial Guess:       ", bad_initial_guess)
    print("Optimized Scales:    ", np.round(recovered_scales, 4))