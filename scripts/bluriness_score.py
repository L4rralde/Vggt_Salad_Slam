from argparse import ArgumentParser
import sys, os
from time import perf_counter

import cv2
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from slam_utils.video_publisher import get_publisher
from src.keyframes.bluriness import (
    variance_of_laplacian,
    tenengrad,
    brenner
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('video_path')
    args = parser.parse_args()
    return args


def score_all_frames(video_path, score_f, wait=0):
    video = get_publisher(video_path)

    img_cnt = 0
    scores = []
    proc_time = 0.0

    while True:
        frame = video.read()
        if frame is None:
            break

        img_cnt += 1

        start = perf_counter()
        score = score_f(frame)
        end = perf_counter()
        proc_time += end - start

        scores.append(score)

        # Display the score on the frame
        if wait >= 0:
            display = frame.copy()
            cv2.putText(
                display,
                f"Score: {score:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Scored Frames", display)

            # Exit early if 'q' is pressed
            if cv2.waitKey(wait) & 0xFF == ord("q"):
                break

    video.release()
    if wait >= 0:
        cv2.destroyAllWindows()

    print(f"Processed {img_cnt} images with {score_f.__name__}")
    print(f"Processing took {proc_time:.3f} seconds")
    return scores


def main():
    args = parse_args()

    scores = {
        'laplacian': score_all_frames(
            args.video_path, 
            variance_of_laplacian,
            wait=30
        ),
        'tenengrad': score_all_frames(
            args.video_path, 
            tenengrad,
            wait=30
        ),
        'brenner': score_all_frames(
            args.video_path, 
            brenner,
            wait=30
        ),
    }

    df = pd.DataFrame(scores)
    df.to_csv('bluriness_score.csv')


if __name__ == '__main__':
    main()
