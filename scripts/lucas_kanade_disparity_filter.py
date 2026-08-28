from argparse import ArgumentParser
import sys, os

import cv2

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from slam_utils.video_publisher import get_publisher
from src.keyframes.frame_overlap import FrameTracker


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('video_path')
    parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    video_tracker = FrameTracker()
    video = get_publisher(args.video_path)

    num_total_imgs = 0
    num_selected_imgs = 0
    
    while True:
        frame = video.read()
        if frame is None:
            break

        num_total_imgs += 1

        enough_disparity = video_tracker.compute_disparity(
            frame,
            args.min_disparity
        )

        num_selected_imgs += int(enough_disparity)

        if enough_disparity:
            cv2.imshow('video', frame)
            cv2.waitKey(20)

    video.release()
    cv2.destroyWindow("video")
    print(f"{num_selected_imgs}/{num_total_imgs}")


if __name__ == '__main__':
    main()
