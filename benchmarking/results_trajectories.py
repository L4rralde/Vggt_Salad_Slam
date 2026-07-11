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


def get_preds_id_from_pred_path(pred_path: str) -> int:
    bname = os.path.basename(pred_path)
    pred_id, _ = os.path.splitext(bname)
    return int(pred_id)


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
        R_T = extrs[:, :3, :3].transpose(0, 2, 1)
        t = extrs[:, :3, 2:3]
        origins = np.squeeze(-R_T @ t, axis=-1)
        
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
        coord = est[:3, :3] @ center + est[:3, 3]

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