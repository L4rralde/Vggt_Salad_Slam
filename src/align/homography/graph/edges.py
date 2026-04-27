from typing import Type

import torch
import pypose as pp

from .. import transforms
from .core import Vertex, Edge
from ..sl4 import SL4, SL4Affine


class EdgeSim3(Edge):
    transform_type: Type[transforms.Transform] = transforms.Sim3

    def __init__(
        self,
        parent: Vertex,
        child: Vertex,
        transform: transforms.Transform,
        information: torch.Tensor = None
    ):
        self.check_transform_type(transform)
        super().__init__(parent, child, transform, information)

    @property
    def ndof(self) -> int:
        return self.transform.ndof

    def check_transform_type(self, transform) -> bool:
        return isinstance(transform, self.transform_type)

    def residual_fn(
        self,
        parent_est: pp.Sim3,
        child_est: pp.Sim3,
        meas: pp.Sim3
    ) -> torch.Tensor:
        prediction = parent_est.Inv() @ child_est
        return (meas.Inv() @ prediction).Log().tensor()

    def edge_residual(self) -> torch.Tensor:
        parent_est_transform: transforms.Sim3 = self.parent.estimate
        child_est_transform: transforms.Sim3 = self.child.estimate
        meas_transform: transforms.Sim3 = self.transform
        return self.residual_fn(
            parent_est_transform.aspypose(),
            child_est_transform.aspypose(),
            meas_transform.aspypose()
        )

    def edge_jacobian(self) -> torch.Tensor:
        parent_est_transform: transforms.Sim3 = self.parent.estimate
        child_est_transform: transforms.Sim3 = self.child.estimate
        meas_transfom: transforms.Sim3 = self.transform

        parent_est = parent_est_transform.aspypose().clone().requires_grad_(True)
        child_est = child_est_transform.aspypose().clone().requires_grad_(True)
        meas = meas_transfom.aspypose()

        zero_perturbation = pp.sim3(torch.zeros(self.ndof)) #Lie algebra $\mathfrank{sl(4)}$

        parent_j = torch.autograd.functional.jacobian(
            lambda d: self.residual_fn(
                d.Exp() @ parent_est,
                child_est,
                meas
            ),
            zero_perturbation
        )
        
        child_j = torch.autograd.functional.jacobian(
            lambda d: self.residual_fn(
                parent_est,
                d.Exp() @ child_est,
                meas
            ),
            zero_perturbation
        )

        return parent_j, child_j
    

class EdgeSL4(Edge):
    lie_group: Type[SL4] = SL4
    transform_type: Type[transforms.Transform] = transforms.Homography
    def __init__(
        self,
        parent: Vertex,
        child: Vertex,
        transform: transforms.Homography,
        information: torch.Tensor = None
    ):
        assert self.check_transform_type(transform)
        super().__init__(parent, child, transform, information)

    def check_transform_type(self, transform) -> bool:
        return isinstance(transform, self.transform_type)

    @property
    def ndof(self) -> int:
        return self.lie_group.ndof

    def residual_fn(self, parent_est: SL4, child_est: SL4, meas: SL4) -> torch.Tensor:
        """
        Computes residual in Lie algebra.
        r = Log(H_{i,j}^{-1} @ (H_i^{-1} H_j))
        parameters:
            - parent_est: SL4 matrix
            - child_est: SL4 matrix
            - meas: SL4 matrix
        """
        prediction = parent_est.inv() @ child_est
        delta_transform = meas.inv() @ prediction
        return (delta_transform).Log()

    def edge_residual(self) -> torch.Tensor:
        parent_est = self.parent.estimate.as_matrix()
        child_est = self.child.estimate.as_matrix()
        meas = self.transform.as_matrix()
        return self.residual_fn(
            self.lie_group(torch.from_numpy(parent_est)),
            self.lie_group(torch.from_numpy(child_est)),
            self.lie_group(torch.from_numpy(meas))
        )

    def edge_jacobian(self) -> torch.Tensor:
        parent_pose = torch.from_numpy(
            self.parent.estimate.as_matrix()
        )
        parent_est_in_lie = self.lie_group(parent_pose)

        child_pose = torch.from_numpy(
            self.child.estimate.as_matrix()
        )
        child_est_in_lie = self.lie_group(child_pose)

        meas = torch.from_numpy(self.transform.as_matrix())
        meas_in_lie = self.lie_group(meas)

        zero_perturbation = torch.zeros(self.ndof).to(torch.float32)
        parent_j = torch.autograd.functional.jacobian(
            lambda d: self.residual_fn(
                self.lie_group.Exp(d) @ parent_est_in_lie,
                child_est_in_lie,
                meas_in_lie
            ),
            zero_perturbation
        )
        child_j = torch.autograd.functional.jacobian(
            lambda d: self.residual_fn(
                parent_est_in_lie,
                self.lie_group.Exp(d) @ child_est_in_lie,
                meas_in_lie
            ),
            zero_perturbation
        )

        return parent_j, child_j


class EdgeSL4Affine(EdgeSL4):
    lie_group: Type[SL4Affine] = SL4Affine
    transform_type: Type[transforms.Transform] = transforms.Affine


