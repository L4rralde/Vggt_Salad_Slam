#conformal orthogonal group in R^3
#
# S = \begin{bmatrix}
#        sR & 0\\
#         0 & 1
#     \end{bmatrix} 
#

import gtsam
import numpy as np
from gtsam import CustomFactor, noiseModel, NonlinearFactorGraph, Values
from gtsam import LevenbergMarquardtOptimizer, LevenbergMarquardtParams
from gtsam import Similarity3, Rot3


# ══════════════════════════════════════════════════════════════════════════════
# CO(3) via Similarity3 — just fix t = 0 everywhere
# ══════════════════════════════════════════════════════════════════════════════

ZERO = np.zeros(3)

def make(R: Rot3, s: float) -> Similarity3:
    return Similarity3(R, ZERO, s)


# ── Numerical Jacobian helper ─────────────────────────────────────────────────
# Sim(3) tangent is 7-DOF: [omega(3), upsilon(3), lambda(1)]
# For CO(3) we only perturb omega(3) and lambda(1); upsilon stays 0.

CO3_DIMS = [0, 1, 2, 6]   # indices in the 7-DOF Sim(3) tangent we actually use

def _numerical_jacobian_sim3(err_fn, x: Similarity3,
                              out_dim: int, eps=1e-6) -> np.ndarray:
    """
    4-column Jacobian: perturb only the CO(3) directions (omega + lambda).
    Uses Similarity3.retract which is already implemented in GTSAM.
    """
    J = np.zeros((out_dim, 4))
    for col, tang_idx in enumerate(CO3_DIMS):
        delta_p = np.zeros(7); delta_p[tang_idx] =  eps
        delta_m = np.zeros(7); delta_m[tang_idx] = -eps
        J[:, col] = (err_fn(x.retract(delta_p)) -
                     err_fn(x.retract(delta_m))) / (2 * eps)
    return J


# ── Error between two Similarity3 elements (translation ignored) ──────────────

def co3_local(x_meas: Similarity3, x_cur: Similarity3) -> np.ndarray:
    """
    log(x_meas^{-1} · x_cur) projected onto CO(3) directions.
    Returns 4-vector: [omega(3), log_scale(1)]
    """
    rel  = x_meas.inverse().compose(x_cur)          # Similarity3
    xi   = rel.localCoordinates(make(Rot3(), 1.0))  # 7-vector in sim(3)
    return xi[CO3_DIMS]                              # pick omega + lambda


# ══════════════════════════════════════════════════════════════════════════════
# Factors
# ══════════════════════════════════════════════════════════════════════════════

def prior_factor(key: int, R_meas: Rot3, s_meas: float, sigma: float):
    x_meas = make(R_meas, s_meas)
    noise  = noiseModel.Isotropic.Sigma(4, sigma)

    def error_func(this, values, jacobians):
        x_cur = values.atSimilarity3(this.keys()[0])
        err   = co3_local(x_meas, x_cur)

        if jacobians is not None:
            jacobians[0] = _numerical_jacobian_sim3(
                lambda x: co3_local(x_meas, x), x_cur, out_dim=4)
        return err

    return CustomFactor(noise, [key], error_func)


def between_factor(key1: int, key2: int,
                   R_rel: Rot3, s_rel: float, sigma: float):
    x_rel_meas = make(R_rel, s_rel)
    noise      = noiseModel.Isotropic.Sigma(4, sigma)

    def error_func(this, values, jacobians):
        x1 = values.atSimilarity3(this.keys()[0])
        x2 = values.atSimilarity3(this.keys()[1])

        x_rel_pred = x1.inverse().compose(x2)
        err        = co3_local(x_rel_meas, x_rel_pred)

        if jacobians is not None:
            jacobians[0] = _numerical_jacobian_sim3(
                lambda x1_: co3_local(x_rel_meas, x1_.inverse().compose(x2)),
                x1, out_dim=4)
            jacobians[1] = _numerical_jacobian_sim3(
                lambda x2_: co3_local(x_rel_meas, x1.inverse().compose(x2_)),
                x2, out_dim=4)
        return err

    return CustomFactor(noise, [key1, key2], error_func)


# ══════════════════════════════════════════════════════════════════════════════
# Demo
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    R1 = Rot3.RzRyRx(0.1, 0.2, 0.3);  s1 = 1.5
    R2 = Rot3.RzRyRx(0.4, 0.1, 0.5);  s2 = 2.0
    R3 = Rot3.RzRyRx(0.2, 0.3, 0.1);  s3 = 0.8

    R_12 = R1.inverse().compose(R2);  s_12 = s2 / s1
    R_23 = R2.inverse().compose(R3);  s_23 = s3 / s2

    def noisy(R: Rot3, s: float) -> Similarity3:
        dR = Rot3.RzRyRx(*np.random.randn(3) * 0.05)
        return make(R.compose(dR), s * np.exp(np.random.randn() * 0.1))

    graph   = NonlinearFactorGraph()
    initial = Values()

    graph.add(prior_factor(0, R1, s1, sigma=0.01))
    graph.add(between_factor(0, 1, R_12, s_12, sigma=0.05))
    graph.add(between_factor(1, 2, R_23, s_23, sigma=0.05))

    for i, (R, s) in enumerate([(R1,s1), (R2,s2), (R3,s3)]):
        initial.insert(i, noisy(R, s))

    params = LevenbergMarquardtParams()
    result = LevenbergMarquardtOptimizer(graph, initial, params).optimize()

    for i, (R_gt, s_gt) in enumerate([(R1,s1),(R2,s2),(R3,s3)]):
        x_opt  = result.atSimilarity3(i)
        R_err  = np.linalg.norm(R_gt.inverse().compose(x_opt.rotation()).axisAngle()[1])
        s_err  = abs(np.log(x_opt.scale() / s_gt))
        t_norm = np.linalg.norm(x_opt.translation())        # should stay ≈ 0
        print(f"Node {i}: rot_err={R_err:.2e} rad  "
              f"scale_err={s_err:.2e}  |t|={t_norm:.2e}")