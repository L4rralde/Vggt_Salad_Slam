from time import perf_counter
from typing import Dict, Hashable, Tuple
from collections import OrderedDict

import numpy as np

from src.transforms.sgraph.scale_graph import ScaleGraph, SparseScaleGraph
from src.transforms.graphs import Sim3Graph
import src.transforms.matransforms as matt
from src.transforms.estimate import EstimateScaleAnchorIntrinsic
from src.models import Prediction


class Sim3SGraph(Sim3Graph):
    def __init__(self):
        super().__init__()
        self.sgraph = SparseScaleGraph()
        self.s_estimater_by_k = None
        self.sim3_s_w = 1.0
        self.refk_s_w = 0.25 #Comment: I don't know how to set this value.
                                #I expect be lower than self.sim3_s_w
                                #because sim_3.s is a better estimator than 
                                # focal lengths ratios
        self.parents = {}
        self.scale_values = {0: 1.0}
    
    def set_ref_intrinsic(self, ref_intrinsic: np.ndarray) -> None:
        self.s_estimater_by_k = EstimateScaleAnchorIntrinsic(ref_intrinsic)

    def add_measurement(self, parent: Hashable, child: Hashable, meas: matt.Sim3) -> None:
        child_exists = child in self._ids_map
        super().add_measurement(parent, child, meas)
        parent_id = self._ids_map[parent]
        child_id = self._ids_map[child]
        s = meas.s
        self.sgraph.add_measurement(parent_id, child_id, s, self.sim3_s_w)
        if not child_exists:
            self.parents[child_id] = parent_id
            self.scale_values[child_id] = self.scale_values[parent_id] * s

    def add_global_s_measurement(self, child: Hashable, preds: Prediction) -> None:
        parent_id = 0
        child_id = self._ids_map[child]
        s = self.s_estimater_by_k(preds)
        self.sgraph.add_measurement(parent_id, child_id, s, self.refk_s_w)
    
    def scale_adjust(self, *, verbose: bool=False, update: bool=False) ->  Tuple[Dict[Hashable, matt.Sim3], Dict[Hashable, matt.Sim3]]:
        s_values = [
            self.scale_values[i]
            for i in range(self._current_id)
        ]

        start = perf_counter()
        new_values = self.sgraph.optimize(s_values)
        end = perf_counter()
        if verbose:
            print(f"Scale optimization took: {end - start:.4f} seconds.")


        for i, new_s in enumerate(new_values):
            self.scale_values[i] = new_s

        current_est = self._prior_estimations[0]
        scaled_estimations = {
            0: current_est
        }
        for i in range(1, self._current_id):
            parent = self.parents[i]
            child = i
            sim3_meas = (
                self._prior_estimations[parent].inv() @ 
                self._prior_estimations[child]
            )
            s_meas = self.scale_values[child]/self.scale_values[parent]

            new_sim3_mat = np.eye(4)
            new_sim3_mat[:3, :3] = s_meas * sim3_meas.R
            #Comment: We must correct the translation (or origin)
            # because we adopted the notation: sR | t, where t is 
            # already iscaled, hence, scale-invariant translation is:
            # sR | st'. When correcting the scaling factor, we seek for ds
            # such as ds s R | ds s t', but ds = new_s/s
            new_sim3_mat[:3, 3] = sim3_meas.t * s_meas/sim3_meas.s
            #new_sim3_mat[:3, 3] = sim3_meas.t
            new_sim3 = matt.Sim3(new_sim3_mat)

            current_est = current_est @ new_sim3
            scaled_estimations[child] = current_est

        reversed_ids_map = {int_id: key for key, int_id in self._ids_map.items()}

        prev_estimates = OrderedDict({
            reversed_ids_map[i]: est
            for i, est in self._prior_estimations.items()
        })

        new_estimates = OrderedDict({
            reversed_ids_map[i]: est
            for i, est in scaled_estimations.items()
        })

        if update:
            if verbose:
                print(f"Prev scales: {s_values}")
                print(f"New scales: {self.scale_values}")
            self._prior_estimations = scaled_estimations
        return (
            OrderedDict(sorted(prev_estimates.items())), 
            OrderedDict(sorted(new_estimates.items()))
        )


    def optimize(self, verbose: bool = False) -> Tuple[Dict[Hashable, matt.Sim3], Dict[Hashable, matt.Sim3]]:
        #self.scale_adjust(verbose)
        return super().optimize(verbose)

    def update_estimation(self, new_estimations: Dict[Hashable, matt.MatrixTransform]) -> None:
        new_scale_values = {
            self._ids_map[k_id]: est.s
            for k_id, est in new_estimations.items()
        }
        self.scale_values = new_scale_values
        super().update_estimation(new_estimations)
