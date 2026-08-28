from typing import Hashable, Dict, Type, Tuple, List, Any
from collections import OrderedDict
from abc import ABC, abstractmethod
from time import perf_counter

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


def matt_aff_to_gtsam(mat_transform: matt.Affine):
    assert isinstance(mat_transform, matt.Affine)
    return gtsam.Aff3(mat_transform._matrix)


def gtsam_to_matt_aff(gtsam_aff3) -> matt.Affine:
    aff3_matrix = gtsam_aff3.matrix()
    aff3_matrix /= aff3_matrix[3, 3]
    return matt.Affine(aff3_matrix)


class LieGraph(ABC):
    NDOF = None
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

        self._matt_type: Type[matt.MatrixTransform]|None = None
        self._gtsam_prior_factor_type = None
        self._gtsam_between_factor_type = None
    
    def add_anchor_prior(
        self,
        node_id: Hashable,
        est: matt.MatrixTransform|None = None
    ) -> None:
        if self._current_id != 0:
            raise RuntimeError("This must be the first node to add")
        if node_id in self._ids_map:
            raise RuntimeError("Id already in graph. This must be the first node to add")
        
        if est is None:
            est = self._matt_type()
        
        type_check = isinstance(est, self._matt_type)
        if not type_check:
            try:
                est = self._matt_type(est._matrix)
            except:
                raise ValueError(f"Estimation matrix is not compatible: {est}")
        
        self._ids_map[node_id] = self._current_id
        self._prior_estimations[self._current_id] = est
        self._current_id += 1

        gtsam_est = self.matt_to_gtsam(est)
 
        self.graph.add(
            self._gtsam_prior_factor_type(
                0, gtsam_est, self.anchor_noise
            )
        )

    @abstractmethod
    def matt_to_gtsam(self, matt_matrix: matt.MatrixTransform) -> object:
        raise NotImplementedError()

    @abstractmethod
    def gtsam_to_matt(self, gtsam_lie_el: object) -> matt.MatrixTransform:
        raise NotImplementedError()

    @abstractmethod
    def result_at(self, result: object, index: int) -> object:
        raise RuntimeError()

    def add_measurement(
        self,
        parent: Hashable,
        child: Hashable,
        meas: matt.MatrixTransform
    ) -> None:
        if not parent in self._ids_map:
            raise ValueError("Unknown parent")
        
        type_check = isinstance(meas, self._matt_type)
        if not type_check:
            try:
                meas = self._matt_type(meas._matrix)
            except:
                raise ValueError(f"Incorrect measurment type: {type(meas)}, expected: {self._matt_type}") 

        child_exists = child in self._ids_map
        if not child_exists:
            self._ids_map[child] = self._current_id
            self._current_id += 1
        
        parent_id = self._ids_map[parent]
        child_id = self._ids_map[child]

        if not child_exists:
            parent_est = self._prior_estimations[parent_id]
            child_est = parent_est @ meas
            assert isinstance(child_est, self._matt_type)
            self._prior_estimations[child_id] = child_est

        self.graph.add(
            self._gtsam_between_factor_type(
                parent_id,
                child_id,
                self.matt_to_gtsam(meas),
                noiseModel=self._meas_noise
            )
        )

    def optimize(self, verbose:bool=False) -> Tuple[Dict[Hashable, matt.MatrixTransform]]:
        values = gtsam.Values()
        for i, est in self._prior_estimations.items():
            assert isinstance(est, self._matt_type)
            values.insert(i, self.matt_to_gtsam(est))
        
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, values, params)

        initial_error = self.graph.error(values)
        result = optimizer.optimize()
        final_error = self.graph.error(result) 

        reversed_ids_map = {int_id: key for key, int_id in self._ids_map.items()}

        new_estimates = {}
        for i in self._prior_estimations.keys():
            gtsam_result = self.result_at(result, i)
            new_estimates[reversed_ids_map[i]] = self.gtsam_to_matt(gtsam_result)

        if verbose:
            print(f"Previous error: {initial_error}")
            print(f"New error: {final_error}")

        reversed_ids_map = {int_id: key for key, int_id in self._ids_map.items()}

        new_estimates = OrderedDict()
        for i in self._prior_estimations.keys():
            gtsam_result = self.result_at(result, i)
            new_estimates[reversed_ids_map[i]] = self.gtsam_to_matt(gtsam_result)
        
        prev_estimates = OrderedDict({
            reversed_ids_map[i]: est
            for i, est in self._prior_estimations.items()
        })

        return (
            OrderedDict(sorted(prev_estimates.items())), 
            OrderedDict(sorted(new_estimates.items()))
        )

    def eval(self, estimations: Dict[Hashable, matt.MatrixTransform]) -> Dict[str, Any]:
        values = gtsam.Values()
        for i, est in estimations.items():
            assert isinstance(est, self._matt_type)
            values.insert(self._ids_map[i], self.matt_to_gtsam(est))
        
        error = self.graph.error(values)
        marginals = gtsam.Marginals(self.graph, values)
        
        reversed_ids_map = {int_id: key for key, int_id in self._ids_map.items()}
        
        est_variances = OrderedDict()
        for int_key, ext_id in reversed_ids_map.items():
            pose_covariance = marginals.marginalCovariance(int_key)
            est_var = np.diag(pose_covariance)
            est_variances[ext_id] = est_var
        
        return {
            'error': error,
            'variances': OrderedDict(sorted(est_variances.items()))
        }

    def update_estimation(self, new_estimations: Dict[Hashable, matt.MatrixTransform]) -> None:
        new_inner_estimations = {
            self._ids_map[k_id]: est
            for k_id, est in new_estimations.items()
        }
        assert self._prior_estimations.keys() == new_inner_estimations.keys()
        self._prior_estimations = new_inner_estimations
    
    def get_prior_estimations(self):
        reversed_ids_map = {int_id: key for key, int_id in self._ids_map.items()}

        prev_estimates = OrderedDict({
            reversed_ids_map[i]: est
            for i, est in self._prior_estimations.items()
        })

        return OrderedDict(sorted(prev_estimates.items()))


class Sim3Graph(LieGraph):
    NDOF = 7
    def __init__(self):
        super().__init__()
        self._matt_type: Type[matt.Sim3] = matt.Sim3
        self._gtsam_prior_factor_type = gtsam.PriorFactorSimilarity3
        self._gtsam_between_factor_type = gtsam.BetweenFactorSimilarity3

    def matt_to_gtsam(self, sim3: matt.Sim3) -> gtsam.Similarity3:
        return matt_sim3_to_gtsam(sim3)

    def gtsam_to_matt(self, gtsam_sim3: gtsam.Similarity3) -> matt.Sim3:
        return gtsam_to_matt_sim3(gtsam_sim3)

    def result_at(self, result: object, index: int) -> object:
        return result.atSimilarity3(index)


class SL4Graph(LieGraph):
    NDOF = 15
    def __init__(self):
        super().__init__()
        self._matt_type: Type[matt.Homography] = matt.Homography
        self._gtsam_prior_factor_type = gtsam.PriorFactorSL4
        self._gtsam_between_factor_type = gtsam.BetweenFactorSL4

    def matt_to_gtsam(self, homo: matt.Homography) -> gtsam.SL4:
        return matt_homography_to_gtsam_sl4(homo)

    def gtsam_to_matt(self, gtsam_sl4: gtsam.SL4) -> matt.Homography:
        return gtsam_sl4_to_matt_homography(gtsam_sl4)

    def result_at(self, result: object, index: int) -> object:
        return result.atSL4(index)


class SL4Aff3Graph(LieGraph):
    NDOF = 12
    def __init__(self):
        super().__init__()
        self._matt_type:  Type[matt.MatrixTransform] = matt.Affine
        self._gtsam_prior_factor_type = gtsam.PriorFactorAff3
        self._gtsam_between_factor_type = gtsam.BetweenFactorAff3
    
    def matt_to_gtsam(self, matt_matrix: matt.MatrixTransform) -> object:
        return matt_aff_to_gtsam(matt_matrix)

    def gtsam_to_matt(self, gtsam_lie_el: object) -> matt.Affine:
        return gtsam_to_matt_aff(gtsam_lie_el)
    
    def result_at(self, result: object, index: int) -> object:
        return result.atAff3(index)


def average_transforms(transform_list: List[matt.MatrixTransform]):
    if len(transform_list) == 1:
        return transform_list[0]
    
    pose_type = matt.get_predominant_instance(*transform_list)

    if isinstance(pose_type(), matt.Sim3):
        graph = Sim3Graph()
    elif isinstance(pose_type(), matt.Affine):
        graph = SL4Aff3Graph()
    elif isinstance(pose_type, matt.Homography):
        graph = SL4Graph()
    else:
        raise ValueError(f"Transformation type {pose_type} not supported")

    graph.add_anchor_prior(0)

    for transform in transform_list:
        graph.add_measurement(0, 1, transform)

    _, estimates = graph.optimize(verbose=False)

    return estimates[1]


def pick_best_transform(transform_list: List[matt.Sim3]):
    if len(transform_list) == 1:
        return transform_list[0]
    
    pose_type = matt.get_predominant_instance(*transform_list)

    if isinstance(pose_type(), matt.Sim3):
        graph_type = Sim3Graph
    elif isinstance(pose_type(), matt.Affine):
        graph_type = SL4Aff3Graph
    elif isinstance(pose_type, matt.Homography):
        graph_type = SL4Graph
    else:
        raise ValueError(f"Transformation type {pose_type} not supported")

    min_error = 1e9
    best = None
    for i, transform in enumerate(transform_list):
        graph = graph_type()
        graph.add_anchor_prior(0)
        graph.add_measurement(0, 1, transform)
        pose_estimations = {
            0: transform.__class__(),
            1: transform
        }
        error = graph.eval(pose_estimations)['error']
        if error < min_error:
            min_error = error
            best = transform
    return best
