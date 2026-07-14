import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from src.models import Prediction
from scripts.eff_optimize_groups_of_preds import find_npz_files


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('results_path')
    return parser.parse_args()


def extrinsics_to_poses(extrinsics: np.ndarray) -> np.ndarray:
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
        raise ValueError("Expected input shape (N, 3, 4).")

    R = extrinsics[:, :, :3]          # (N, 3, 3)
    t = extrinsics[:, :, 3:]          # (N, 3, 1)

    R_inv = np.transpose(R, (0, 2, 1))
    t_inv = -R_inv @ t

    poses = np.concatenate([R_inv, t_inv], axis=2)
    return poses


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 homogeneous transformation to an array of 3D points.

    Parameters
    ----------
    T : (4, 4) ndarray
        Homogeneous transformation matrix.
    points : (n, 3) ndarray
        3D points.

    Returns
    -------
    transformed_points : (n, 3) ndarray
        Transformed 3D points.
    """
    points = np.asarray(points)
    T = np.asarray(T)

    if T.shape != (4, 4):
        raise ValueError("T must be a 4x4 matrix.")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3).")

    # Convert to homogeneous coordinates
    homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    # Apply transformation
    transformed = (T @ homogeneous.T).T

    # Convert back to Cartesian coordinates
    transformed /= transformed[:, 3][:, np.newaxis]

    return transformed[:, :3]


def main():
    args = parse_args()

    preds_dir = os.path.join(
        args.results_path, 'unaligned_preds'
    )
    
    all_preds = [
        Prediction.from_npz_file(f)
        for f in find_npz_files(preds_dir)
    ]

    estimations = np.load(
        os.path.join(args.results_path, 'estimations.npy')
    )

    cov = np.load(
        os.path.join(args.results_path, 'vars.npy')
    )
    cov_traces = [v.sum() for v in cov]

    view_origin = {}
    for pred, est, trace in zip(all_preds, estimations, cov_traces):
        extrs = pred.extrinsic
        poses = extrinsics_to_poses(extrs)
        origins = poses[:, :3, 3]
        
        for opt_center, cam_id in zip(origins, pred.ids):
            if not cam_id in view_origin: #First estimation
                view_origin[cam_id] = {
                    'center': opt_center,
                    'est': est,
                    'trace': trace
                }
                continue
            #New estimation
            if trace < view_origin[cam_id]['trace']: #Actually, a better estimation
                view_origin[cam_id] = {
                    'center': opt_center,
                    'est': est,
                    'trace': trace
                }
        
    data = []
    for cam_id in view_origin.keys():
        est = view_origin[cam_id]['est']
        center = view_origin[cam_id]['center']
        coord = transform_points(est, center[None])[0]

        data.append({
            'frame_id': cam_id,
            'x': coord[0],
            'y': coord[1],
            'z': coord[2],
        })
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(args.results_path,'trajectory.csv')
    df.to_csv(csv_path)


if __name__ == '__main__':
    main()