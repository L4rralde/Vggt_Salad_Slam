import numpy as np
import pytest

from scale_graph import ScaleGraph


def test_empty_graph():
    g = ScaleGraph()
    assert g.num_nodes == 0


def test_add_measurement():
    g = ScaleGraph()
    g.add_measurement(0, 1, 2.0)

    assert g.num_nodes == 2
    assert len(g.edges) == 1


def test_optimize_requires_correct_length():
    g = ScaleGraph()
    g.add_measurement(0, 1, 0.5)

    with pytest.raises(RuntimeError):
        g.optimize([1.0, 2.0, 3.0])


def test_gauge_must_be_one():
    g = ScaleGraph()
    g.add_measurement(0, 1, 0.5)

    with pytest.raises(RuntimeError):
        g.optimize([2.0, 1.0])


def test_two_node_exact():
    """
    0 ----> 1

    measurement = s0 / s1
    """
    g = ScaleGraph()

    true_scales = np.array([1.0, 4.0])

    measurement = true_scales[0] / true_scales[1]

    g.add_measurement(0, 1, measurement)

    result = g.optimize([1.0, 2.0])

    np.testing.assert_allclose(result, true_scales, atol=1e-8)


def test_chain_exact():
    """
    0 -> 1 -> 2 -> 3
    """
    true = np.array([1.0, 2.5, 0.4, 10.0])

    g = ScaleGraph()

    edges = [(0, 1), (1, 2), (2, 3)]

    for i, j in edges:
        g.add_measurement(i, j, true[i] / true[j])

    initial = np.ones_like(true)

    recovered = g.optimize(initial)

    np.testing.assert_allclose(recovered, true, atol=1e-8)


def test_redundant_graph_exact():
    """
    Overconstrained but perfectly consistent.
    """
    true = np.array([1.0, 2.5, 0.4, 10.0])

    g = ScaleGraph()

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
        (0, 2),
        (1, 3),
    ]

    for i, j in edges:
        g.add_measurement(i, j, true[i] / true[j])

    recovered = g.optimize(np.ones_like(true))

    np.testing.assert_allclose(recovered, true, atol=1e-8)


def test_noisy_measurements():
    """
    Optimizer should recover scales approximately.
    """
    rng = np.random.default_rng(42)

    true = np.array([1.0, 2.5, 0.4, 10.0])

    g = ScaleGraph()

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
        (1, 3),
        (0, 2),
    ]

    for i, j in edges:
        noise = rng.normal(1.0, 0.02)
        g.add_measurement(i, j, (true[i] / true[j]) * noise)

    recovered = g.optimize(np.ones_like(true))

    np.testing.assert_allclose(recovered, true, rtol=0.05)


def test_weighted_solution_prefers_high_weight_edge():
    """
    Two inconsistent measurements.
    High-weight edge should dominate.
    """
    g = ScaleGraph()

    # True scale should be near 2.0
    g.add_measurement(0, 1, 0.5, weight=100.0)

    # Bad measurement
    g.add_measurement(0, 1, 0.25, weight=1.0)

    result = g.optimize([1.0, 1.0])

    assert abs(result[1] - 2.0) < abs(result[1] - 4.0)


def test_solution_independent_of_initial_guess():
    true = np.array([1.0, 2.5, 0.4, 10.0])

    g = ScaleGraph()

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 3),
        (1, 3),
    ]

    for i, j in edges:
        g.add_measurement(i, j, true[i] / true[j])

    guess1 = np.array([1.0, 1.0, 1.0, 1.0])
    guess2 = np.array([1.0, 100.0, 0.01, 50.0])

    sol1 = g.optimize(guess1)
    sol2 = g.optimize(guess2)

    np.testing.assert_allclose(sol1, sol2, atol=1e-8)