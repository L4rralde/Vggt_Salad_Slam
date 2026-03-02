import os
from typing import List, Tuple
from multiprocessing import Queue
from dataclasses import dataclass, asdict
from time import perf_counter, sleep
import gc

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from addict import Dict

from src.models import ViewPrediction, Prediction


@dataclass
class Frame:
    id: int
    stamp: float
    img: Image.Image


def cv2_to_pil(img: np.ndarray) -> Image.Image:
    cv_frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(cv_frame_rgb)
    return pil_frame

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    pil_data = img.convert('RGB')
    return np.array(pil_data)[:, :, ::-1]

def video_publisher(video_path: str, frame_q: Queue) -> None:
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    if not fps:
        print("WARNING. Could not get FPS of the video. Defaulting to 30 fps")
        fps = 30.0
    period = 1.0/fps
    frame_cnt = 0
    sleep(20)
    print(f"Playing video at {fps} FPS")
    while True:
        ret, frame = video.read()
        if not ret:
            frame_q.put(None)
            print("Finished playing video")
            break
        pil_img = cv2_to_pil(frame)
        stamp = frame_cnt/fps
        id = frame_cnt + 1
        if frame_q.full():
            print("[WARNING], frame_q is full.")
        frame_q.put(Frame(id, stamp, pil_img))
        frame_cnt += 1
        sleep(period)
        #cv2.imshow('video', frame)
        #cv2.waitKey(25)

    video.release()
    #cv2.destroyWindow("video")


class KeyFramesDetector:
    def __init__(self):
        self.last_descriptor: torch.Tensor|None = None
        self.key_frames: List[int] = []

    def __call__(
        self,
        frame_ids: List[int],
        view_preds: ViewPrediction,
        th: float=0.75
    ) -> Tuple[List[int], ViewPrediction]:
        descriptors = view_preds.descriptors

        idcs = []
        kf_ids = []
        if self.last_descriptor is None:
            self.last_descriptor = descriptors[0].clone()
            idcs.append(0)
            kf_ids.append(frame_ids[0])

        ref = self.last_descriptor
        for i, desc in enumerate(descriptors):
            sim = (desc @ ref).item()
            if sim < th:
                ref = desc
                idcs.append(i)
                kf_ids.append(frame_ids[i])
        
        if not idcs:
            return [], {}
        
        self.last_descriptor = descriptors[idcs[-1]].clone()
        self.key_frames += kf_ids

        kf_preds = ViewPrediction(
            view_preds.images[idcs],
            view_preds.patch_tokens[idcs],
            view_preds.descriptors[idcs]            
        )

        return kf_ids, kf_preds


def keyframe_publisher(keyframe_q: Queue) -> None:
    while True:
        kf: Frame|None = keyframe_q.get()
        if kf is None:
            break
        kf = pil_to_cv2(kf.img)
        cv2.imshow("keyframes", kf)
        cv2.waitKey(25)
    cv2.destroyWindow("keyframes")

@dataclass
class ViewToken:
    frame_id: int
    token: torch.Tensor
    processed_img: torch.Tensor


def video_processing(frame_q: Queue, kframes_q: Queue, preds_q: Queue) -> None:
    from src.models import get_model

    print("Loading model...")
    start = perf_counter()
    model = get_model(
        'da3-base',
        '/home/emmanuel/Desktop/tesis/Visual_Place_Recognition'
    )
    end = perf_counter()
    print(f'Finished loading model. Took {end - start:.2f} seconds')

    from src.storage import(
        FrameRepository,
        FIFOCache,
        TensorRepository,
        SoftLink
    )

    frames = FrameRepository('output/frames') #Frames and metadata in disk
    keyframes_cp = SoftLink('output/keyframes', frames.root)
    frames_cache = FIFOCache(64)
    descriptors = TensorRepository('output/viewpreds/descriptors')
    #patch_tokens = TensorRepository('output/viewpreds/patch_tokens')
    #processed_imgs = TensorRepository('output/viewpreds/processed_imgs')
    viewpreds_cache = FIFOCache(32)

    to_encode_ids = [] #Ids to encode.
    to_predict_ids = []
    predicted_ids = []
    keyframe_detector = KeyFramesDetector() #Detects if a frame is a new frame
    while True:
        #1. Append frames to a batch for encoding
        frame: Frame = frame_q.get()
        if frame is None: #Check if video has not finished
            preds_q.put(None)
            kframes_q.put(None)
            break

        #This block takes 0.01 seconds
        _id = frames.append(frame.img, frame.stamp) #Save frame in disk
        if _id != frame.id:
            raise RuntimeError(f"Packet loss: ({frame.id}, {_id})")
        frames_cache.append(frame.id, frame) #Add frame to cache
        to_encode_ids.append(frame.id)
        
        if len(to_encode_ids) < 32: #If the number of frames not encoded is not 32, receive another frame
            continue

        #2. Encode the batch
        image_batch = [
            frames_cache.get(id).img
            for id in to_encode_ids
        ]
        start = perf_counter()
        #Can't achieve real time @ 30FPS
        #With VGGT-SALAD Encoding almost takes (in average) 0.27 seconds.
        #Wait period between frames @30FPS is 0.33. So, only 0.06 seconds remain.
        # Preprocesing 32 frames takes: .22 seconds.
        # Moving data to cpu takes .12 seconds.
        # Calling torch.cuda.empty_cache() takes 0.12 seconds
        # Only cpu preprocessing PIL-to-PIL takes 5ms per Image. For VGGT-SALAD. Check for da3-salad
        #     Hence it takes 0.06 seconds to move 32 images to gpu.
        view_preds = model.views_encoding(image_batch)
        end = perf_counter()
        print(f"Encoding of {len(image_batch)} frames took {end - start:.2f} seconds.")
        encoded_ids = to_encode_ids
        to_encode_ids = []

        #3. Keyframe selection
        #This is fast. Not a bottleneck. Approx 50ns per frame.
        kf_ids, kf_preds = keyframe_detector(encoded_ids, view_preds) 
        if not kf_ids:
            continue

        print(f"Found new keyframes: {kf_ids}")
        #Publish and save keyframes:
        for id in kf_ids:
            #kf = frames_cache.get(id)
            #kframes_q.put(kf) #Publish
            path = frames.get_path(id)
            #save keyframe. #In average takes 500ns per keyframe. Not a bottleneck
            keyframes_cp.copy(path)

        #Saving filtered predictions in disk and cache. 
        #This is a bottleneck. takes 0.17 per individual key frame.
        #Does not scale linearly.
        #Due to disk operations, it may be better to use multiprocssing here.
        start = perf_counter()
        zip_iter = zip(kf_ids, kf_preds.patch_tokens, kf_preds.images, kf_preds.descriptors)
        for frame_id, token, img, desc in zip_iter:
            descriptors.append(frame_id, desc) #save in disk
            #patch_tokens.append(frame_id, token) #disk
            #processed_imgs.append(frame_id, img) #disk
            viewpreds_cache.append(frame_id, ViewToken(frame_id, token, img))#Ram
        
        #Both explicitly deleting a variable and calling the gc is slow.
        #del view_preds #Remove unused views 
        #gc.collect()
        end = perf_counter()
        print(f"Saving and caching {len(kf_ids)} keyframes pred took {end - start:.4f} seconds")

        to_predict_ids += kf_ids
        if len(to_predict_ids) < 4:
            continue
        #3 Chunk Sequence prediction
        #Preparing chunk
        chunk_ids = predicted_ids[-4:] + to_predict_ids #Ids of frames to chunk
        print(f"Processing sequence of size {len(chunk_ids)}...")
        start = perf_counter()
        chunk_tokens: List[ViewToken] = [viewpreds_cache.get(id) for id in chunk_ids]
        chunk_preds = Dict()
        chunk_preds.patch_tokens = torch.stack([token.token for token in chunk_tokens])
        chunk_preds.images = torch.stack([token.processed_img for token in chunk_tokens])
        preds = model.chunk_prediction(chunk_preds)
        end = perf_counter()
        print(f"Sequence prediction took {end - start:.3f} seconds")
        predicted_ids += to_predict_ids
        to_predict_ids = []

        preds.ids = np.asarray(chunk_ids, dtype=np.uint32)
        if preds_q.full():
            print("[WARNING] preds_q is full")
        preds_q.put(preds)


def prediction_aligning(predictions_q: Queue) -> None:
    from src.sim3 import VggtlongAlign
    registered_ids = []
    prev_preds: Dict[str, np.ndarray] = {}
    chunk_cnt = 0
    os.makedirs('preds', exist_ok=True)
    while True:
        curr_preds = predictions_q.get()
        if curr_preds is None:
            break

        if prev_preds:
            start = perf_counter()
            print(f"Aligning scenes...")
            curr_preds = VggtlongAlign().fit_transform(prev_preds, curr_preds)
            end = perf_counter()
            print(f"Aligning took {end - start:.4f} seconds")

        prev_preds = curr_preds
        registered_ids += list(curr_preds.ids)
        path = f'./preds/{chunk_cnt}.npz'
        np.savez(path, **asdict(curr_preds))

        chunk_cnt += 1


def main():
    mp.set_start_method("spawn")
    q_frames = mp.Queue(256)
    q_preds = mp.Queue(maxsize=8)
    q_kframes = mp.Queue(maxsize=8)
    p_publisher = mp.Process(
        target=video_publisher,
        args=("/home/emmanuel/Downloads/cimat_loop.mp4", q_frames)
    )
    p_processing = mp.Process(
        target=video_processing,
        args=(q_frames, q_kframes, q_preds)
    )
    p_aligning = mp.Process(
        target=prediction_aligning,
        args=(q_preds, )
    )
    #p_kfpublisher = mp.Process(
    #    target=keyframe_publisher,
    #    args=(q_kframes, )
    #)

    p_publisher.start()
    p_processing.start()
    p_aligning.start()
    #p_kfpublisher.start()

    p_publisher.join()
    p_processing.join()
    p_aligning.join()
    #p_kfpublisher.join()


if __name__ == '__main__':
    main()
