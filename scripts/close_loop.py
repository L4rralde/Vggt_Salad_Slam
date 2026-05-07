import sys
import os

import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.align import preds_align
from src.align.align_utils import get_conf_mask
from src.align.homography import transforms
from src.align.homography.graph.core import Vertex, Optimizer
from src.align.homography.graph.edges import EdgeSL4Affine
from src.align.homography.graph.algorithms import GaussNewton


def preds_to_pcd(preds, pointmap):
    mask = get_conf_mask(preds, lower_p=60, min_conf=1.005, upper_p=90)
    colors = preds['images'][mask].reshape(-1, 3)
    points = pointmap[mask].reshape(-1, 3)
    #points[:, 0] *= -1
    #points[:, 1] *= -1
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def main():
    root = "calderon_output"

    preds = [
        dict(np.load(f"{root}/{i}/all.npz"))
        for i in range(1, 9)
    ]

    for p in preds:
        p.pop('model', None)
    
    measures = []
    for tgt, src in zip(preds[:-1], preds[1:]):
        measures.append(preds_align.FitAffine().fit(src, tgt))
    measures.append(preds_align.FitAffine().fit(preds[0], preds[-1]))

    current_est = preds_align.FitAffine()
    current_est._transform = transforms.Affine.identity()
    estimations = [current_est]
    for meas in measures[:-1]:
        current_est = current_est@meas
        estimations.append(current_est)
    
    vertices = [
        Vertex.Affine(i, est._transform.copy())
        for i, est in enumerate(estimations)
    ]

    Edges = [
        EdgeSL4Affine(v_parent, v_child, meas._transform.copy())
        for (v_parent, v_child), meas in \
            zip(zip(vertices[:-1], vertices[1:]), measures[:-1])
    ]
    Edges.append(
        EdgeSL4Affine(
            vertices[-1],
            vertices[0],
            measures[-1]._transform.copy()
        )
    )

    optim = Optimizer(
        GaussNewton,
        [v for v in vertices],
        [e for e in Edges]
    )

    optim.optimize(10)

    new_aligners = []
    for v in optim.vertices:
        aligner = preds_align.FitAffine()
        aligner._transform = v.estimate.copy()
        new_aligners.append(aligner)
    
    new_pcd = [
        preds_to_pcd(p, est.transform(p))
        for p, est in zip(preds, new_aligners)
    ]

    o3d.visualization.draw_geometries(new_pcd)


if __name__ == '__main__':
    main()