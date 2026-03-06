from typing import Dict, List, Set, Optional
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from addict import Dict

from .align import Sim3Align


@dataclass
class Chunk:
    frame_ids: List[int]
    locked: Optional[bool] = False


class ChunkStorageInterface(ABC):
    @abstractmethod
    def get(self, id: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError()
    @abstractmethod
    def __contains__(self, item) -> bool:
        raise NotImplementedError


class OptimizationGraph:
    def __init__(self,
        chunk_cache: ChunkStorageInterface|object,
        chunk_repo: ChunkStorageInterface|object,
        aligning: Sim3Align
    ):
        self.chunks_dict: Dict[int, Chunk] = {}
        self.unique_frame_ids: Set[int] = set()
        self.chunk_cache = chunk_cache
        self.chunk_repo = chunk_repo
        self.aligning_class: Sim3Align = aligning
        #Tells the chunk that an id belongs to. Only first appearence.
        #For example, chunk 0 = [0, 1, 2], chunk_1 = [2, 3, 4].
        # 0, 1, 2 belongs to chunk_0, while 3, 4 belongs to chunk_1. 
        # 2 actually belongs to both, but we only consider the first one.
        self.frame_ids_map: Dict[int, int] = {} 
        #Edges[x][y] = S (Sim3Aling). Sy -> x. S transform aligns y(src) to x(tgt)
        #We say x is a father of y
        self.edges: Dict[int, Dict[int, Sim3Align]] = defaultdict(dict)
        self.root: int = 0
        self.loops_to_optimize: List[List[int]] = []
    
    def get_chunk_prediction(self, id: int) -> Dict[str, np.ndarray]:
        if id in self.chunk_cache:
            return self.chunk_cache.get(id)
        if id in self.chunk_repo:
            return Dict(dict(self.chunk_repo.get(id)))
        raise KeyError(f"id {id} unknown")

    def add_edge(self, father: id, child: id) -> None:
        src_preds = self.get_chunk_prediction(father)
        tgt_preds = self.get_chunk_prediction(child)
        self.edges[father][child] = self.aligning_class().fit(tgt_preds, src_preds)

    def find_route(self, root: int, node: int) -> List[int]:
        if not root in self.edges or not self.edges[root]:
            return []
        if node in self.edges[root]:
            return [root]
        for child in self.edges[root].keys():
            subroute = self.find_route(child, node)
            if subroute:
                return [root] + subroute
        return []

    def find_loop(self, root_closing: int, closing: int) -> List[int]:
        route = self.find_route(root_closing, closing)
        if not route:
            return []
        return route + [closing]

    def append(self, id_of_chunk: int, chunk_frames_ids: List[int]) -> int:
        self.chunks_dict[id_of_chunk] = Chunk(chunk_frames_ids)
        ids_set = set(chunk_frames_ids)
        connecting_ids = set([id for id in chunk_frames_ids if id in self.unique_frame_ids])
        new_ids = ids_set - connecting_ids
        chunks_connected_to = {self.frame_ids_map[id] for id in connecting_ids}

        self.unique_frame_ids |= new_ids
        for id in new_ids:
            self.frame_ids_map[id] = id_of_chunk

        for registered_chunk in sorted(chunks_connected_to, reverse=True):
            if id_of_chunk == registered_chunk + 1:
                self.add_edge(registered_chunk, id_of_chunk)
            else:
                self.add_edge(id_of_chunk, registered_chunk) #Here we have a closed loop.
                loop = self.find_loop(registered_chunk, id_of_chunk)
                self.loops_to_optimize.append(loop)
                print(f"found new loop: {loop}")

    def optimize_loop(self, loop: List[int]) -> None:
        for id in loop:
            self.chunks_dict[id].locked = True
        raise NotImplementedError()

    def optimize(self) -> None:
        if not self.loops_to_optimize:
            return
        for loop in self.loops_to_optimize:
            self.optimize_loop(loop)
        raise NotImplementedError
