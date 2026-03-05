from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F


KERNEL = torch.tensor(
    [[1/16, 1/8, 1/16],
    [1/8, 1/4, 1/8],
    [1/16, 1/8, 1/16]],
    dtype=torch.float32
)

class CloseLoopDetector:
    QUANT_K: int = 1.0
    def __init__(self):
        self.id_to_idx: Dict[int, int] = {}
        self.idx_to_id: List[int] = []
        self.descriptors: torch.Tensor|None = None

    def append(self, id: int, descriptor: torch.Tensor) -> None:
        if id in self.id_to_idx:
            raise ValueError(f"{id} already in set of descriptors")
        self.id_to_idx[id] = len(self.idx_to_id)
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
    
    def get_descriptors(self, ids: int|List[int]) -> torch.Tensor:
        if isinstance(ids, int):
            ids = [ids]
        #Sanity check
        for id in ids:
            if not id in self.id_to_idx:
                raise ValueError(f"Id {id} not in CloseLoopDetector Memory")
        idcs = [self.id_to_idx[id] for id in ids]
        return self.descriptors[idcs]

    def __call__(
        self,
        query_ids: List[int],
        ref_ids: List[int],
        th: float
    ) -> Tuple[int, int, float]|None:
        if len(query_ids) == 0 or len(ref_ids) == 0:
            return None
        ref_desciptors = self.get_descriptors(ref_ids)
        query_descriptors = self.get_descriptors(query_ids)
        sim = ref_desciptors @ query_descriptors.T
        m, n = len(ref_ids), len(query_ids)
        filtered_sim = F.conv2d(
            sim.view(1, 1, m, n),
            KERNEL.view(1, 1, 3, 3),
            stride=1,
            padding=1
        ).view(m, n)
        filtered_sim[filtered_sim < th] = 0
        if filtered_sim.sum() < 1e-4:
            return None
        max_value = filtered_sim.max()
        max_idx = torch.where(filtered_sim == max_value)
        match_ref_idx, match_query_idx = max_idx
        match_ref_id = ref_ids[match_ref_idx.item()]
        match_query_id = query_ids[match_query_idx.item()]
        sim_value = sim[max_idx]
        return match_query_id, match_ref_id, sim_value
