from typing import List, Set, Optional, Deque, Tuple, Dict, Any
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import addict

from .align import Sim3Align
from .sim3 import Sim3
from .loop_closing import refine_sim3_loop, refine_sim3_sequence, refine_sim3_loop_with_interpolation


@dataclass
class Chunk:
    chunk_id: int
    frame_ids: List[int]
    locked: Optional[bool] = False


class ChunkStorageInterface(ABC):
    @abstractmethod
    def get(self, id: int) -> Dict[str, np.ndarray]:
        raise NotImplementedError()
    @abstractmethod
    def __contains__(self, item) -> bool:
        raise NotImplementedError
    @abstractmethod
    def append(self, id: int, data: Any) -> None:
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
        #We say x is a parent of y
        self.edges: Dict[int, Dict[int, Sim3]] = defaultdict(dict)
        self.root: int = 0
        self.loops_to_optimize: List[List[int]] = []
    
    def get_chunk_prediction(self, id: int) -> Dict[str, np.ndarray]:
        if id in self.chunk_cache:
            return self.chunk_cache.get(id)
        if id in self.chunk_repo:
            return addict.Dict(dict(self.chunk_repo.get(id)))
        raise KeyError(f"id {id} unknown")

    def add_edge(self, parent: id, child: id) -> None:
        src_preds = self.get_chunk_prediction(parent)
        tgt_preds = self.get_chunk_prediction(child)
        self.edges[parent][child] = self.aligning_class().fit(tgt_preds, src_preds).sim3

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
        self.chunks_dict[id_of_chunk] = Chunk(id_of_chunk, chunk_frames_ids)
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
        if self.loops_to_optimize:
            self.optimize()

    def sort_locked_chunks_first_in_loop(self, ids_loop: List[int]) -> List[int]:
        chunks: Deque[Chunk] = deque([self.chunks_dict[id] for id in ids_loop])

        #1. Check if there's any chunk locked
        found_locked = any((chunk.locked for chunk in chunks))
        if not found_locked:
            return ids_loop
        
        while chunks[0].locked:
            chunks = chunks.appendleft(chunks.pop()) #Rotate right

        while not chunks[0].locked:
            chunks = chunks.append(chunks.popleft()) #Rotate left

        for prev_chunk, curr_chunk in zip(chunks[:-1], chunks[1:]):
            if curr_chunk.locked and not prev_chunk.locked:
                print([(chunk.id, chunk.locked) for chunk in chunks])
                raise RuntimeError(f"Found another sequence of locked chunks after first one.")
    
        return [chunk.chunk_id for chunk in chunks]

    def split_into_locked_and_not(self, ids_loop: List[int]) -> Tuple[List[int], List[int]]:
        chunks: List[Chunk] = [self.chunks_dict[id] for id in ids_loop]
        locked_chunk_ids = [chunk.chunk_id for chunk in chunks if chunk.locked]
        non_locked_chunk_ids = [chunk.chunk_id for chunk in chunks if not chunk.locked]
        return [locked_chunk_ids, non_locked_chunk_ids]

    def simple_optimize_loop(self, loop: List[int]) -> None:
        #Here we do not distinguish betweend locked nodes and not-locked nodes
        loop.append(loop[0])
        edges = list(zip(loop[:-1], loop[1:]))
        sim3_seq = [
            self.edges[parent][child]
            for parent, child in edges
        ]
        #We expect S_10 S_21 S_32 S_03 = I
        #Hence S_10 S_21 S_32 = S_03^-1
        sim3_seq = refine_sim3_loop_with_interpolation(sim3_seq)
        
        for (parent, child), sim3 in zip(edges, sim3_seq):
            self.edges[parent][child] = sim3

        for id in loop:
            self.chunks_dict[id].locked = True

    def optimize_loop_with_sorting(self, loop: List[int]) -> None:
        sorted_loop = self.sort_locked_chunks_first_in_loop(loop)
        locked, to_optimize = self.split_into_locked_and_not(sorted_loop)
        if not to_optimize:
            return
        
        if not locked:
            to_optimize.append(to_optimize[0])
            to_optimize_edges = list(zip(to_optimize[:-1], to_optimize[1:]))
            sim3_seq = [
                self.edges[parent][child]
                for parent, child in to_optimize_edges
            ]
            sim3_seq = refine_sim3_sequence(
                sim3_seq,
                Sim3.identity(),
            )
        else:
            locked.insert(0, to_optimize[-1])
            locked_edges = [
                (parent, child)
                for parent, child in zip(locked[:-1], locked[1:])
            ]
            locked_transform = Sim3.identity()
            for parent, child in locked_edges:
                locked_transform = locked_transform @ self.edges[parent][child]
            
            to_optimize.insert(0, locked[-1])
            to_optimize_edges = [
                (parent, child)
                for parent, child in zip(to_optimize[:-1], to_optimize[:1])
            ]
            sim3_seq = [
                self.edges[parent][child]
                for parent, child in to_optimize_edges
            ]
            sim3_seq = refine_sim3_sequence(
                sim3_seq,
                locked_transform.inv()
            )

        for (child, parent), sim3 in zip(to_optimize_edges, sim3_seq):
            self.edges[parent][child] = sim3

        for id in to_optimize:
            self.chunks_dict[id].locked = True

    def optimize(self) -> None:
        if not self.loops_to_optimize:
            return
        for loop in self.loops_to_optimize:
            self.simple_optimize_loop(loop)
        self.loops_to_optimize = []

    def update_chunks_tree(
        self,
        root: int=-1,
        world_transform: Sim3|None=None,
        updated: List[int] = []
    ) -> None:
        if root == -1:
            root = self.root
        if world_transform is None:
            world_transform = Sim3.identity()
        
        if root in updated:
            return

        chunk = addict.Dict(dict(self.chunk_repo.get(root)))
        transform = Sim3Align()
        transform.sim3 = world_transform
        print(f"applying new transform to chunk {root}: {world_transform}")
        chunk = transform.transform(chunk)
        self.chunk_repo.append(root, chunk)
        updated += [root]

        if not self.edges[root]:
            return
        for child in self.edges[root]:
            self.update_chunks_tree(
                root = child,
                world_transform = world_transform@self.edges[root][child],
                updated = updated
            )
            self.edges[root][child] = Sim3.identity()

    def finish(self) -> None:
        print("Updating all chunks")
        self.update_chunks_tree()
