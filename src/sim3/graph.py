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
    global_transform: Optional[Sim3]|None = None


@dataclass
class Edge:
    parent: int
    child: int
    transform: Sim3 #Aligns child's coordinate system with father's one
    locked: Optional[bool] = False

    def __repr__(self) -> str:
        return f"Edge(parent: {self.parent}, child: {self.child}, locked: {self.locked})"


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
    #Following the Separation of concerns principle, this class shouldn't write back
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
        self.edges: Dict[int, Dict[int, Edge]] = defaultdict(dict)
        self.root: int = 0
        self.loops_to_optimize: List[List[int]] = []
    
    def get_chunk_prediction(self, id: int) -> Dict[str, np.ndarray]:
        if id in self.chunk_cache:
            return self.chunk_cache.get(id)
        if id in self.chunk_repo:
            return addict.Dict(dict(self.chunk_repo.get(id)))
        raise KeyError(f"id {id} not sotred")

    def add_edge(self, parent: id, child: id, sim3: Sim3|None=None) -> Sim3:
        """
        Parent: id of the parent (src) chunk
        child: id of the child (dst) chunk
        sim3: sim3 transformation used to align child's coordinate system with parent's one.
                It may be computed before.
        """
        tgt_preds = self.get_chunk_prediction(parent)
        src_preds = self.get_chunk_prediction(child)
        if sim3 is None:
            sim3 = self.aligning_class().fit(tgt_preds, src_preds).sim3
        self.edges[parent][child] = Edge(parent, child, sim3)
        return sim3

    def __get_path_no_loops_recursive(self, root: int, node: int, visited: Set[int]) -> List[int]:
        if not self.edges[root]:
            return []
        if node in self.edges[root]:
            return [root]
        visited = set(visited) #Create a copy to not use a shared set in all recursive calls
        visited.add(root)
        for child in self.edges[root].keys():
            if not self.edges[child] or child in visited:
                continue
            subpath = self.__get_path_no_loops_recursive(child, node, visited)
            if subpath:
                return [root] + subpath
        return []

    def get_path(self, root: int, node: int) -> List[int]:
        #It's suboptimal to always traverse from self.root.
        if not root in self.chunks_dict:
            raise ValueError(f"Src chunk {root} not in graph")
        if not node in self.chunks_dict:
            raise ValueError(f"Dst chunk {node} not in graph")
        return self.__get_path_no_loops_recursive(root, node, visited=set())

    def get_loop_path(self, root_closing: int, closing: int) -> List[int]:
        path = self.get_path(root_closing, closing)
        if not path:
            raise RuntimeError(f"Found no path from Node {root_closing} and {closing}")
        return path + [closing]

    def append(self, id_of_chunk: int, chunk_frames_ids: List[int], aligned: bool=False) -> int:
        """
        Appends a new chunk of predictions to the Optimization Graph.
        id_of chunk: Unique int to identify chunk of predictions.
        chunk_frames_ids: Ids of the frames used to produce this chunk.
        Aligned: Tells you this chunked was already aligned (and transformed) to lie in its father coordinate system.
                    In such a case father -> child transform is the identity.
        """
        global_transform = (
            Sim3.identity()
            if id_of_chunk == self.root else None
        )
            
        self.chunks_dict[id_of_chunk] = Chunk(
            id_of_chunk,
            chunk_frames_ids,
            global_transform
        )

        ids_set = set(chunk_frames_ids)
        #These are the frame ids, of the current chunk, that are used to align the chunk with another.
        connecting_ids = set([
            id for id in chunk_frames_ids
            if id in self.unique_frame_ids
        ])
        new_ids = ids_set - connecting_ids #ids that yet are not included in any other chunk in the graph.
        chunks_connected_to = {self.frame_ids_map[id] for id in connecting_ids}

        self.unique_frame_ids |= new_ids
        for id in new_ids:
            self.frame_ids_map[id] = id_of_chunk

        for registered_chunk in sorted(chunks_connected_to, reverse=True):
            if id_of_chunk == registered_chunk + 1: #Consecutive chunks are treated as parent-child edges
                sim3 = None if not aligned else Sim3.identity()
                sim3 = self.add_edge(registered_chunk, id_of_chunk, sim3)
                parent_global = self.chunks_dict[registered_chunk].global_transform
                self.chunks_dict[id_of_chunk].global_transform = parent_global @ sim3
            else:
                self.add_edge(id_of_chunk, registered_chunk) #Here we have a closed loop.
                loop = self.get_loop_path(registered_chunk, id_of_chunk) #This function does not find a loop, just returns the ids contained in the already found loop.
                #self.loops_to_optimize.append(loop)
                print(f"found new loop: {loop}")
                self.simple_loop_optimization(loop)
                self.update_global_transforms(registered_chunk)
                #Let's do the optimization in-place
                #self.simple_optimize_loop(loop)
                #self.update_loop(loop, registered_chunk)
                #Then. update it. But we must update the whole subtree
                #We must optimize the loop, update the chunks, the chunks repo and update the appending chunk in the fifo cache
            #By the moment I'll update the predictions.
            #FIXME. Remove when loop optimization actually works.
            self.update_chunks_tree(registered_chunk)

    def update_global_transforms(self, root: int, visited: Set[int]=set()) -> None:
        for child in self.edges[root]:
            if child in visited:
                continue
            global_transform = self.chunks_dict[root].global_transform
            rel_transform = self.edges[root][child].transform
            self.chunks_dict[child].global_transform = global_transform @ rel_transform
            self.update_global_transforms(child, visited | set([root]))

    def __split_locked_unlocked(self, edges: List[Edge]) -> Tuple[List[Edge], List[Edge]]:
        locked = [edge.locked for edge in edges]
        if not any(locked):
            return [], list(edges)
        if all(locked):
            return list(edges), []

        edge_dq = Deque(edges) #Used as circular buffer
        while edge_dq[0].locked:
            edge_dq.appendleft(edge_dq.pop()) #Rotate right
        while not edge_dq[0].locked:
            edge_dq.append(edge_dq.popleft()) #Rotate left

        sorted_edges = list(edge_dq)

        partition_idx = [edge.locked for edge in sorted_edges].index(False)
        return sorted_edges[:partition_idx], sorted_edges[partition_idx:]

    def simple_loop_optimization(self, loop: List[int]) -> None:
        loop.append(loop[0])
        edges = [
            self.edges[parent][child]
            for parent, child in zip(loop[:-1], loop[1:])
        ]
        locked_edges, unlocked_edges = self.__split_locked_unlocked(edges)
        if not unlocked_edges:
            #Nothing to update
            return

        locked_transform = Sim3.identity()
        for edge in locked_edges:
            locked_transform = locked_transform @ edge.transform
        locked_transform = locked_transform @ unlocked_edges.pop().transform
        constraint = locked_transform.inv()
        unlocked_transforms = [edge.transform for edge in unlocked_edges]
        unlocked_transforms = refine_sim3_sequence(unlocked_transforms, constraint)
        for edge, transform in zip(unlocked_edges, unlocked_transforms):
            edge.transform = transform
            edge.locked = True

    def simple_optimize_loop(self, loop: List[int]) -> None:
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
            self.edges[parent][child].transform = sim3

    def optimize(self) -> None:
        if not self.loops_to_optimize:
            return
        for loop in self.loops_to_optimize:
            self.simple_optimize_loop(loop)
        self.loops_to_optimize = []

    def update_chunks_tree(
        self,
        root: int|None = None,
        world_transform: Sim3|None=None,
        updated: List[int] = []
    ) -> None:
        #
        if root is None:
            root = self.root
        if world_transform is None:
            world_transform = Sim3.identity()

        if world_transform != Sim3.identity(): #update only when there's an actual change
            chunk = addict.Dict(dict(self.chunk_repo.get(root)))
            transform = Sim3Align()
            transform.sim3 = world_transform
            chunk = transform.transform(chunk)
            self.update_storage(root, chunk)

        updated = updated + [root]
        print(f"Updating chunk {root} with global transform {world_transform}")

        if not self.edges[root]:
            return
        for child in self.edges[root]:
            if child in updated:
                continue
            rel_transform = self.edges[root][child].transform
            self.update_chunks_tree(
                root = child,
                world_transform = world_transform @ rel_transform,
                updated = updated
            )
            self.edges[root][child].transform = Sim3.identity()

    def finish(self) -> None:
        print("Updating all chunks")
        self.update_chunks_tree()

    def update_storage(self, chunk_id: int, data: Any) -> None:
        #FIXME. This violates the Separation of concerns principle
        if chunk_id in self.chunk_cache:
            self.chunk_cache.append(chunk_id, data)
        self.chunk_repo.append(chunk_id, data)

