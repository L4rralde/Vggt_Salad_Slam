from typing import Hashable, Dict, Type, Tuple
from collections import OrderedDict

import gtsam
import numpy as np

import src.transforms.matransforms as matt


def matt_sim3_to_gtsam(sim3: matt.Sim3) -> gtsam.Similarity3:
    assert isinstance(sim3, matt.Sim3)
    sR = sim3._matrix[:3, :3]
    s = np.linalg.norm(sR[:, 0]) #Instead of \det(sR)^{1/3}
    U, _, Vt = np.linalg.svd(sR / s)
    R = U @ Vt  # guaranteed orthonormal
    t = sim3._matrix[:3, 3]/s
    return gtsam.Similarity3(R, t, s)


def gtsam_to_matt_sim3(gtsam_sim3: gtsam.Similarity3) -> matt.Sim3:
    R_mat = gtsam_sim3.rotation().matrix()
    t_vec = gtsam_sim3.translation()
    s_val = gtsam_sim3.scale()

    matrix = np.eye(4)
    matrix[:3, :3] = s_val*R_mat
    matrix[:3, 3] = s_val*t_vec
    return matt.Sim3(matrix)


def matt_homography_to_gtsam_sl4(mat_transform: matt.MatrixTransform) -> gtsam.SL4:
    assert isinstance(mat_transform, matt.MatrixTransform)
    return gtsam.SL4(mat_transform._matrix)


def gtsam_sl4_to_matt_homography(gtsam_sl4: gtsam.SL4) -> matt.Homography:
    sl4_matrix = gtsam_sl4.matrix()
    sl4_matrix /= sl4_matrix[3, 3]
    return matt.Homography(sl4_matrix)


class Sim3Graph:
    NDOF = 7
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.full(self.NDOF, 1e-6, dtype=float)
        )
        self._meas_noise = gtsam.noiseModel.Diagonal.Sigmas(
            0.05*np.ones(self.NDOF, dtype=float)
        )
        self._ids_map: Dict[Hashable, int] = {}
        self._prior_estimations: Dict[int, matt.MatrixTransform] = OrderedDict()
        self._current_id = 0

    def add_anchor_prior(
        self,
        id: Hashable,
        est: matt.Sim3|None = None
    ) -> None:
        if self._current_id != 0:
            raise RuntimeError("This must be the first node to add")
        if id in self._ids_map:
            raise RuntimeError("Id already in graph. This must be the first node to add")
        
        self._ids_map[id] = self._current_id
        self._current_id += 1
        
        if est is None:
            est = matt.Sim3()
        
        self._prior_estimations[0] = est
        gtsam_sim3 = matt_sim3_to_gtsam(est)
 
        self.graph.add(
            gtsam.PriorFactorSimilarity3(
                0, gtsam_sim3, self.anchor_noise
            )
        )

    def add_measurement(
        self,
        parent: Hashable,
        child: Hashable,
        meas: matt.Sim3
    ) -> None:
        if not parent in self._ids_map:
            raise ValueError("Unknown parent")
        
        type_check = isinstance(meas, matt.Sim3)
        if not type_check:
            raise ValueError(f"Incorrect measurment type: {type(meas)}") 

        child_exists = child in self._ids_map
        if not child_exists:
            self._ids_map[child] = self._current_id
            self._current_id += 1
        
        parent_id = self._ids_map[parent]
        child_id = self._ids_map[child]

        if not child_exists:
            parent_est = self._prior_estimations[parent_id]
            child_est = parent_est @ meas
            assert isinstance(child_est, matt.Sim3)
            self._prior_estimations[child_id] = child_est

        self.graph.add(
            gtsam.BetweenFactorSimilarity3(
                parent_id,
                child_id,
                matt_sim3_to_gtsam(meas),
                noiseModel=self._meas_noise
            )
        )
        
    def optimize(self, verbose:bool=False) -> Tuple[Dict[Hashable, matt.Sim3]]:
        values = gtsam.Values()
        for i, est in self._prior_estimations.items():
            assert isinstance(est, matt.Sim3)
            values.insert(i, matt_sim3_to_gtsam(est))
        
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, values, params)

        initial_error = self.graph.error(values)
        result = optimizer.optimize()
        final_error = self.graph.error(result)

        if verbose:
            print(f"Previous error: {initial_error}")
            print(f"New error: {final_error}")

        reversed_ids_map = {int_id: key for key, int_id in self._ids_map.items()}

        new_estimates = OrderedDict()
        for i in self._prior_estimations.keys():
            sim3_result = result.atSimilarity3(i)
            new_estimates[reversed_ids_map[i]] = gtsam_to_matt_sim3(sim3_result)
        
        prev_estimates = OrderedDict({
            reversed_ids_map[i]: est
            for i, est in self._prior_estimations.items()
        })

        return (
            OrderedDict(sorted(prev_estimates.items())), 
            OrderedDict(sorted(new_estimates.items()))
        )


class SL4Graph:
    NDOF=15
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.full(self.NDOF, 1e-6, dtype=float)
        )
        self._meas_noise = gtsam.noiseModel.Diagonal.Sigmas(
            0.05*np.ones(self.NDOF, dtype=float)
        )
        self._ids_map: Dict[Hashable, int] = {}
        self._prior_estimations: Dict[int, matt.MatrixTransform] = OrderedDict()
        self._current_id = 0

    def add_anchor_prior(
        self,
        id: Hashable,
        est: matt.MatrixTransform|None=None
    ) -> None:
        if self._current_id != 0:
            raise RuntimeError("This must be the first node to add")
        if id in self._ids_map:
            raise RuntimeError("Id already in graph. This must be the first node to add")
        
        self._ids_map[id] = self._current_id
        self._current_id += 1

        if est is None:
            est = matt.Homography()
        
        self._prior_estimations[0] = matt.Homography(est)
        sl4 = matt_homography_to_gtsam_sl4(est)

        self.graph.add(
            gtsam.PriorFactorSL4(0, sl4, self.anchor_noise)
        )
    
    def add_measurement(
        self,
        parent: Hashable,
        child: Hashable,
        meas: matt.MatrixTransform
    ) -> None:
        if not parent in self._ids_map:
            raise ValueError("Unknown parent")
        
        child_exists = child in self._ids_map
        if not child_exists:
            self._ids_map[child] = self._current_id
            self._current_id += 1
        
        parent_id = self._ids_map[parent]
        child_id = self._ids_map[child]

        if not child_exists:
            parent_est = self._prior_estimations[parent_id]
            child_est = parent_est @ meas
            assert isinstance(child_est, matt.Homography)
            self._prior_estimations[child_id] = child_est

        self.graph.add(
            gtsam.BetweenFactorSL4(
                parent_id,
                child_id,
                matt_homography_to_gtsam_sl4(meas),
                noiseModel=self._meas_noise
            )
        )

    def optimize(self, verbose:bool=False) -> Tuple[Dict[Hashable, matt.Homography]]:
        values = gtsam.Values()
        for i, est in self._prior_estimations.items():
            assert isinstance(est, matt.Homography)
            values.insert(i, matt_homography_to_gtsam_sl4(est))
        
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, values, params)

        initial_error = self.graph.error(values)
        result = optimizer.optimize()
        final_error = self.graph.error(result) 

        if verbose:
            print(f"Previous error: {initial_error}")
            print(f"New error: {final_error}")

        reversed_ids_map = {int_id: key for key, int_id in self._ids_map.items()}

        new_estimates = OrderedDict()
        for i in self._prior_estimations.keys():
            sl4 = result.atSL4(i)
            new_estimates[reversed_ids_map[i]] = gtsam_sl4_to_matt_homography(sl4)
        
        prev_estimates = OrderedDict({
            reversed_ids_map[i]: est
            for i, est in self._prior_estimations.items()
        })

        
        return (
            OrderedDict(sorted(prev_estimates.items())), 
            OrderedDict(sorted(new_estimates.items()))
        )