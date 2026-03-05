from typing import List, Tuple, Set

import torch
import torch.nn.functional as F



class CloseLoopDetector:
    QUANT_K: int = 1.0
    def __init__(self):
        self.known_ids: Set[int] = set()
        self.idx_to_id: List[int] = []
        self.descriptors: torch.Tensor|None = None

    def append(self, id: int, descriptor: torch.Tensor) -> None:
        if id in self.known_ids:
            raise ValueError(f"{id} already in set of descriptors")
        self.known_ids.add(id)
        self.idx_to_id.append(id)
        if self.descriptors is None:
            self.descriptors = descriptor.unsqueeze(0)
        else:
            self.descriptors = torch.cat((self.descriptors, descriptor.unsqueeze(0)), axis=0)

    def topk(self, descriptor: torch.Tensor, k: int, descriptors: torch.Tensor|None = None) -> Tuple[List[int], List[float]]:
        if descriptors is None:
            descriptors = self.descriptors
        sim = descriptors @ descriptor
        sim_v, idcs = torch.topk(sim, k)
        topk_ids = [self.idx_to_id[idx] for idx in idcs]
        return topk_ids, sim_v

    def __call__(self, id: int, descriptor: torch.Tensor, th: float, dt: int=0) -> Tuple[List[int], List[float]]:
        if self.descriptors is None:
            self.append(id, descriptor)
            return [], []

        descriptors = self.descriptors if dt==0 else self.descriptors[:-dt]
        if len(descriptors) < 3:
            self.append(id, descriptor)
            return [], []

        sim = descriptors @ descriptor
        sim[sim < th] = 0
        kernel = torch.tensor([0.5, 1.0, 0.5], dtype=torch.float32)
        kernel = kernel.view(1, 1, -1)
        filtered_sim = F.conv1d(sim.view(1, 1, -1), kernel, padding='same').squeeze()
        idx = filtered_sim.argmax().item()
        l_idx = max(0, idx - 1)
        r_idx = min(len(sim), idx+2)
        sim_v = sim[l_idx: r_idx]
        ids = [self.idx_to_id[idx] for idx in range(l_idx, r_idx)]

        self.append(id, descriptor)

        if sim_v.sum() < 1e-6:
            return [], []
        return ids, sim_v.tolist()
