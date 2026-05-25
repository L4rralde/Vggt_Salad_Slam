from typing import List, Tuple
from multiprocessing import Queue
from dataclasses import dataclass, asdict
from time import perf_counter, sleep, time

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from addict import Dict

from slam_utils import cv2_to_pil, pil_to_cv2, get_publisher

@dataclass
class Frame:
    id: int
    stamp: float
    img: Image.Image


def video_publisher(video_path: str, frame_q: Queue, model_ready: mp.Event) -> None:
    video = get_publisher(video_path)
    fps = video.get_fps()
    if fps <= 0:
        print("WARNING. Could not get FPS of the video. Defaulting to 30 fps")
        fps = 30.0
    period = 1.0/fps
    frame_cnt = 0
    model_ready.wait()
    print(f"Playing video at {fps} FPS")
    prev_time = time()
    while True:
        frame = video.read()
        if frame is None:
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
        cv2.imshow('video', frame)
        cv2.waitKey(20)
        while time() - prev_time < period:
            sleep(period/20.0)
        prev_time = time()

    video.release()
    cv2.destroyWindow("video")


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


def video_processing(frame_q: Queue, kframes_q: Queue, preds_q: Queue, model_ready: mp.Event) -> None:
    from collections import defaultdict
    from src.models import get_model
    from src.keyframes import KeyFramesDetector, CloseLoopDetector

    print("Loading model...")
    start = perf_counter()
    model = get_model(
        'mapanything',
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

    frames = FrameRepository('output/frames', clear=True) #Frames and metadata in disk
    keyframes_cp = SoftLink('output/keyframes', frames.root, clear=True)
    frames_cache = FIFOCache(64)
    #descriptors = TensorRepository('output/viewpreds/descriptors', clear=True)
    patch_tokens = TensorRepository('output/viewpreds/patch_tokens', clear=True)
    processed_imgs = TensorRepository('output/viewpreds/processed_imgs', clear=True)
    viewpreds_cache = FIFOCache(32)
    loop_detector = CloseLoopDetector()

    to_encode_ids = [] #Ids to encode.
    to_predict_ids = []
    predicted_ids = []
    chunk_cnt = 0
    chunk_dict = defaultdict(list)

    keyframe_detector = KeyFramesDetector() #Detects if a frame is a new frame
    model_ready.set()
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
        if _id % 4 != 0:
            continue
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
        # For DA3, they use a concurrent procedure:  
        # Moving data to cpu takes .012 seconds. 10 Takes 0.05 seconds. 32 takes 0.15 seconds.
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
        kf_ids, kf_preds = keyframe_detector(encoded_ids, view_preds, th=0.85)
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
            #descriptors.append(frame_id, desc) #save in disk
            patch_tokens.append(frame_id, token) #disk
            processed_imgs.append(frame_id, img) #disk
            viewpreds_cache.append(frame_id, ViewToken(frame_id, token, img))#Ram
        
        #Both explicitly deleting a variable and calling the gc is slow.
        #del view_preds #Remove unused views 
        #gc.collect()
        end = perf_counter()
        print(f"Saving and caching {len(kf_ids)} keyframes pred took {end - start:.4f} seconds")

        #Closed loop detection
        for id, descriptor in zip(kf_ids, kf_preds.descriptors):
            loop_detector.append(id, descriptor)

        to_predict_ids += kf_ids
        if len(to_predict_ids) < 4:
            continue

        start = perf_counter()
        match = loop_detector(
            query_ids = predicted_ids[-2:] + to_predict_ids,
            ref_ids = predicted_ids[:-20],
            th = 0.7
        )
        end = perf_counter()

        if match is not None:
            query_match, ref_match, sim = match
            print(f"Found match ({query_match}, {ref_match}) with sim {sim}")
            print(f"Loop detection call took {end - start:.4f} seconds")
            found_closed_chunk = False
            for chunk, frame_ids in chunk_dict.items():
                if ref_match in frame_ids:
                    found_closed_chunk = True
                    break
            if not found_closed_chunk:
                raise RuntimeError(f"Found not corresponding chunk of closing frame with id: {ref_match}")
            
            loop_aligning_ids = frame_ids
            print(f"Appending ids to next sequence prediction: {loop_aligning_ids}")
        else:
            loop_aligning_ids = []

        #3 Chunk Sequence prediction
        #Preparing chunk
        chunk_ids = predicted_ids[-4:] + to_predict_ids + loop_aligning_ids #Ids of frames to chunk
        start = perf_counter()
        chunk_tokens: List[ViewToken] = []
        for id in chunk_ids:
            if id in viewpreds_cache:
                chunk_tokens.append(viewpreds_cache.get(id))
                continue
            view_token = ViewToken(
                id,
                patch_tokens.get(id),
                processed_imgs.get(id)
            )
            chunk_tokens.append(view_token)

        chunk_preds = Dict()
        chunk_preds.patch_tokens = torch.stack([token.token for token in chunk_tokens])
        chunk_preds.images = torch.stack([token.processed_img for token in chunk_tokens])
        preds = model.chunk_prediction(chunk_preds)
        end = perf_counter()
        print(f"Sequence prediction of size {len(chunk_ids)} took {end - start:.3f} seconds")

        chunk_dict[chunk_cnt] = to_predict_ids
        predicted_ids += to_predict_ids
        chunk_cnt += 1
        to_predict_ids = []

        preds.ids = np.asarray(chunk_ids, dtype=np.uint32)
        if preds_q.full():
            print("[WARNING] preds_q is full")
        preds_q.put(preds)


def prediction_aligning(predictions_q: Queue) -> None:
    from src.sim3 import VggtlongAlign, OptimizationGraph, sim3_transform
    from src.storage import NdarrayRepository, FIFOCache
    from src.sim3.ppgraph import OptimizationGraph
    from src.sim3.myg2o import GaussNewton

    chunks_cache = FIFOCache(2)
    chunks_repo = NdarrayRepository('output/chunks', clear=True)
    graph = OptimizationGraph(GaussNewton, chunks_cache, chunks_repo, VggtlongAlign)

    registered_ids = []
    chunk_cnt = 0

    while True:
        curr_preds = predictions_q.get()
        if curr_preds is None:
            break
        
        chunks_cache.append(chunk_cnt, curr_preds) #Stores chunk in cache
        chunks_repo.append(chunk_cnt, asdict(curr_preds)) #Stores chunk in disk
        graph.append(chunk_cnt, curr_preds.ids)
        if graph.requires_optimization:
            graph.optimize(10)

        registered_ids += list(curr_preds.ids)

        chunk_cnt += 1
    
    transformed_chunks_repo = NdarrayRepository('output/aligned_chunks', clear=True)
    for id in chunks_repo.list_all_ids():
        chunk = Dict(chunks_repo.get(id, copy=True))
        transform = graph.get_absolute_transform([id])[0]
        transformed_chunk = sim3_transform(transform, chunk)
        transformed_chunks_repo.append(id, transformed_chunk)


def main():
    mp.set_start_method("fork")
    q_frames = mp.Queue(256)
    q_preds = mp.Queue(maxsize=8)
    q_kframes = mp.Queue(maxsize=8)
    model_ready = mp.Event()
    p_publisher = mp.Process(
        target=video_publisher,
        args=("/home/emmanuel/Desktop/videos/dji_fly_20260323_195503_0_1774317303930_video_low_quality.mp4", q_frames, model_ready)
    )
    p_processing = mp.Process(
        target=video_processing,
        args=(q_frames, q_kframes, q_preds, model_ready)
    )
    p_aligning = mp.Process(
        target=prediction_aligning,
        args=(q_preds, )
    )
    #p_kfpublisher = mp.Process(
    #    target=keyframe_publisher,
    #    args=(q_kframes, )
    #)

    processes = [p_publisher, p_processing, p_aligning]
    for p in processes:
        p.start()

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
