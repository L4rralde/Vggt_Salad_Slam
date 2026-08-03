from typing import List, Set

import numpy as np
from scipy.optimize import least_squares


class ScaleGraph:
    def __init__(self) -> None:
        self.edges: List[tuple] = []
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