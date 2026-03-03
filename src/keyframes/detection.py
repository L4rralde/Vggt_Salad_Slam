from typing import List, Tuple, Any
from copy import copy


class KeyFramesDetector:
    def __init__(self):
        self.last_descriptor: Any|None = None
        self.key_frames: List[int] = []

    @staticmethod
    def filter_obj(obj: Any, idcs: List[int]) -> Any:
        f_obj = copy(obj)
        for k, v in vars(obj).items():
            setattr(f_obj, k, v[idcs])
        return f_obj

    def __call__(
        self,
        frame_ids: List[int],
        view_preds: Any,
        th: float=0.75
    ) -> Tuple[List[int], Any]:
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

        kf_preds = KeyFramesDetector.filter_obj(view_preds, idcs)
        
        return kf_ids, kf_preds
