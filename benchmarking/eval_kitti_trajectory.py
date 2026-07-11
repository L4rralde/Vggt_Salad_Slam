from argparse import ArgumentParser
from pathlib import Path
import os
import copy

import pandas as pd
from evo.tools import file_interface
from evo.core.trajectory import PosePath3D
import matplotlib.pyplot as plt
import evo.tools.plot as plot
from evo.core import metrics


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('estimated')
    parser.add_argument('gt')
    return parser.parse_args()


def align_and_compute_ate_sim3(gt_path: str, est_path: str):
    """
    Loads KITTI format trajectories, aligns the estimation to the ground truth
    using a Sim(3) transformation (correcting scale), and computes the ATE.
    
    Args:
        gt_path (str): Path to the ground truth KITTI poses file.
        est_path (str): Path to the estimated KITTI poses file.
                              
    Returns:
        traj_gt: The ground truth trajectory object.
        traj_est_aligned: The Sim(3) aligned estimated trajectory object.
        stats (dict): Dictionary containing the ATE metrics (RMSE, mean, etc.).
    """
    # 1. Load trajectories (KITTI format)
    traj_gt = file_interface.read_kitti_poses_file(gt_path)
    traj_est = file_interface.read_kitti_poses_file(est_path)
    
    # 2. Align estimation to ground truth with Sim(3)
    traj_est_aligned = copy.deepcopy(traj_est)
    # correct_scale=True enforces Sim(3) alignment
    traj_est_aligned.align(traj_gt, correct_scale=True) 
    
    # 3. Compute ATE (Absolute Pose Error focusing only on translation)
    pose_relation = metrics.PoseRelation.translation_part
    ape_metric = metrics.APE(pose_relation)
    
    ape_metric.process_data((traj_gt, traj_est_aligned))
    stats = ape_metric.get_all_statistics()
    
    return traj_gt, traj_est_aligned, stats


def save_trajectory_plot(traj_gt, traj_est_aligned, save_path: str, plot_mode_str: str = "xyz"):
    """
    Plots the ground truth and aligned estimated trajectories and saves to disk.
    
    Args:
        traj_gt: Ground truth trajectory object.
        traj_est_aligned: Aligned estimated trajectory object.
        save_path (str): The file path where the plot image will be saved.
        plot_mode_str (str): "xyz" for 3D plot, or "xy", "xz", "yz" for 2D projections.
    """
    fig = plt.figure(figsize=(8, 8))
    
    # Parse plot mode
    if plot_mode_str.lower() == "xy":
        plot_mode = plot.PlotMode.xy
    elif plot_mode_str.lower() == "xz":
        plot_mode = plot.PlotMode.xz
    elif plot_mode_str.lower() == "yz":
        plot_mode = plot.PlotMode.yz
    else:
        plot_mode = plot.PlotMode.xyz
        
    ax = plot.prepare_axis(fig, plot_mode)
    
    # Plot trajectories
    plot.traj(ax, plot_mode, traj_gt, style='--', color='gray', label='Ground Truth', alpha=0.8)
    plot.traj(ax, plot_mode, traj_est_aligned, style='-', color='blue', label='Estimated (Aligned)', alpha=0.8)
    
    # Formatting
    fig.axes.append(ax)
    plt.title(f"Trajectory Alignment ({plot_mode_str.upper()})")
    plt.legend()
    plt.tight_layout()
    
    # Save the plot and close the figure to free memory
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot successfully saved to: {save_path}")


def df_to_kitti_poses(df: pd.DataFrame, output_file: str | Path) -> None:
    """
    Convert a DataFrame with columns ['frame_id', 'x', 'y', 'z']
    into a KITTI poses file with identity rotations.
    """
    required = {"frame_id", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort a copy by frame_id
    df_sorted = df.sort_values("frame_id").copy()

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w") as f:
        for row in df_sorted.itertuples(index=False):
            pose = (
                1.0, 0.0, 0.0, row.x,
                0.0, 1.0, 0.0, row.y,
                0.0, 0.0, 1.0, row.z,
            )
            f.write(" ".join(f"{v:.6f}" for v in pose) + "\n")


def main():
    args = parse_args()

    tmp_dir = os.path.join(
        os.path.dirname(args.estimated),
        'tmp_traj_files_benchmarking'
    )
    #1. Create temporal directory to dump temporal files
    os.makedirs(tmp_dir, exist_ok=True)

    #2. Read estimated trajectory csv
    est_traj_df = pd.read_csv(args.estimated)

    #3. Dump estimated trajectory to temporal file in kitti format
    df_kitti_poses_path = os.path.join(tmp_dir, 'estimated.txt')
    df_to_kitti_poses(est_traj_df, df_kitti_poses_path)
    
    #4. Read gt trajectory file
    gt_traj = file_interface.read_kitti_poses_file(args.gt)

    #5. Select only the estimated camera poses
    views = [int(view) for view in est_traj_df['frame_id']]
    filtered_traj = PosePath3D(
        poses_se3=[
            gt_traj.poses_se3[i]
            for i in views
            if 0 <= i < gt_traj.num_poses
        ]
    )
    # 5.2 And dump those into a temporal file
    ft_gt_tr_path = os.path.join(tmp_dir, "filtered_gt_trajectory.txt")
    file_interface.write_kitti_poses_file(
        ft_gt_tr_path,
        filtered_traj
    )

    gt_traj, est_aligned_traj, ate_stats = align_and_compute_ate_sim3(ft_gt_tr_path, df_kitti_poses_path)

    print("=== Sim(3) Absolute Trajectory Error ===")
    print(f"RMSE:   {ate_stats['rmse']:.4f} m")
    print(f"Mean:   {ate_stats['mean']:.4f} m")
    print("========================================")

    output_image = f'{args.estimated}.png'
    save_trajectory_plot(gt_traj, est_aligned_traj, save_path=output_image, plot_mode_str="xz")


if __name__ == '__main__':
    main()
