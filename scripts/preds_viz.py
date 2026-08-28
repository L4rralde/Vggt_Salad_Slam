import argparse

import numpy as np
import open3d as o3d
from tqdm import tqdm

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.sim3.utils import get_conf_mask, unproject_depth_map_to_point_map


def to_pointcloud(conf, images, world_points):
    images = np.transpose(images, (0, 2, 3, 1))
    mask = get_conf_mask(conf, lower_p=40.0, min_conf=1.02)

    points = world_points[mask].reshape(-1, 3)
    colors = images[mask].reshape(-1, 3)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


def scene_pointcloud(scene: dict):
    if not 'world_points' in scene:
        world_points = unproject_depth_map_to_point_map(
            scene['depth'],
            scene['intrinsic'],
            scene['extrinsic']
        )
    else:
        world_points = scene['world_points']

    return to_pointcloud(
        scene['depth_conf'],
        scene['images'],
        world_points
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', type=str, nargs='+')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    preds = (np.load(p, allow_pickle=True) for p in args.paths)

    print("Generating point clouds for Open3D")
    pcds = [
        scene_pointcloud(pred)
        for pred in tqdm(preds)
    ]

    o3d.visualization.draw_geometries(pcds)


if __name__ == '__main__':
    main()
