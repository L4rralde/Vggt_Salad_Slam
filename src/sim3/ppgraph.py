from typing import List, Set, Dict, Any, Type, Iterable
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import addict
import pypose as pp

from .myg2o import Sim3Vertex, Sim3Edge, Sim3Optimizer, Algorithm
from .align import Sim3Align
from .sim3 import Sim3


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


class ChunkVertex(Sim3Vertex):
    def __init__(
        self,
        chunk_id: int,
        frame_ids: List[int],
        absolute_transform: pp.Sim3
    ):
        self.frame_ids: List[int] = frame_ids
        self.chunk_id = chunk_id
        super().__init__(self.chunk_id, absolute_transform)


class ChunkEdge(Sim3Edge):
    def __repr__(self) -> str:
        return f"Edge(parent: {self.parent.idx}, child: {self.child.idx})"


class OptimizationGraph(Sim3Optimizer):
    def __init__(
        self,
        algorithm: Type[Algorithm],
        chunk_cache: ChunkStorageInterface,
        chunk_repo: ChunkStorageInterface,
        aligning_method: Type[Sim3Align]
    ):
        super().__init__(algorithm)
        self.unique_frame_ids: Set[int] = set()
        self.children_dict: Dict[int, List[int]] = defaultdict(list)
        self.chunk_cache: ChunkStorageInterface = chunk_cache
        self.chunk_repo: ChunkStorageInterface = chunk_repo
        self.aligning_class: Type[Sim3Align] = aligning_method

        #Tells the chunk that an id belongs to. Only first appearence.
        #For example, chunk 0 = [0, 1, 2], chunk_1 = [2, 3, 4].
        # 0, 1, 2 belongs to chunk_0, while 3, 4 belongs to chunk_1. 
        # 2 actually belongs to both, but we only consider the first one.
        self.frame_ids_map: Dict[int, int] = {}
        
        self.root: int = 0
        self.__requires_optimization: bool = False

    @property
    def requires_optimization(self) -> bool:
        return self.__requires_optimization

    def get_chunk_prediction(self, id: int) -> Dict[str, np.ndarray]:
        if id in self.chunk_cache:
            return self.chunk_cache.get(id)
        if id in self.chunk_repo:
            return addict.Dict(dict(self.chunk_repo.get(id)))
        raise KeyError(f"id {id} not sotred")

    def __get_path_no_loops_recursive(
        self,
        root_id: int,
        node_id: int,
        visited: Set[int]
    ) -> List[int]:
        if not self.children_dict[root_id]:
            return []
        if node_id in self.children_dict[root_id]:
            return [root_id]
        
        visited = visited + [root_id] #Now this is a copy
        for child_id in self.children_dict[root_id]:
            if not self.children_dict[child_id] or child_id in set(visited):
                continue
            subpath = self.__get_path_no_loops_recursive(
                child_id, node_id, visited
            )
            if subpath:
                return [root_id] + subpath
        return []

    def get_path(self, root_id: int, node_id: int) -> List[int]:
        if root_id >= len(self.vertices):
            raise ValueError(f"Src chunk {root_id} not in graph")
        if node_id >= len(self.vertices):
            raise ValueError(f"Dst chunk {node_id} not in graph")
        return self.__get_path_no_loops_recursive(root_id, node_id, visited=[])

    def get_loop_path(self, root_closing: int, closing: int) -> List[int]:
        path = self.get_path(root_closing, closing)
        if not path:
            raise RuntimeError(f"Found no path from Node {root_closing} and {closing}")
        return path + [closing]

    def append(self, id_of_chunk: int, chunk_frames_ids: List[int]) -> int:
        ids_set = set(chunk_frames_ids)

        #These are the frame ids, of the current chunk, that are used to align the chunk with another.
        connecting_ids = set([
            id for id in chunk_frames_ids
            if id in self.unique_frame_ids
        ])
        new_ids = ids_set - connecting_ids #ids that yet are not included in any other chunk in the graph.
        self.unique_frame_ids |= new_ids
        for id in new_ids:
            self.frame_ids_map[id] = id_of_chunk

        if id_of_chunk == self.root: #First chunk
            self.append_vertex(ChunkVertex(
                id_of_chunk, chunk_frames_ids, pp.identity_Sim3()
            ))
            return

        chunks_connected_to = {self.frame_ids_map[id] for id in connecting_ids}
        assert len(chunks_connected_to) > 0, "Tried to append an isolatd nodde"
        chunks_connected_to = sorted(chunks_connected_to, reverse=True)

        #Getting parent->child transform
        parent_id = chunks_connected_to.pop(0)
        assert parent_id == id_of_chunk - 1, "Broke sequence"
        rel_transform = self.get_relative_transform(parent_id, id_of_chunk)

        #Appending new vertex to grpah
        absolute_transform = self.vertices[parent_id].estimate @ rel_transform
        vertex = ChunkVertex(id_of_chunk, chunk_frames_ids, absolute_transform)
        self.append_vertex(vertex)
        
        #Appending new edge
        parent_vertex = self.vertices[parent_id]
        edge = ChunkEdge(parent_vertex, vertex, rel_transform)
        self.append_edge(edge)
        self.children_dict[parent_id].append(id_of_chunk)

        if not chunks_connected_to:
            return
        
        self.__requires_optimization = True

        #Found new closed loops
        for registered_chunk in chunks_connected_to:
            loop = self.get_loop_path(registered_chunk, id_of_chunk)
            print(f"Found new loop: {loop}")
            edge = ChunkEdge(
                vertex,
                self.vertices[registered_chunk],
                self.get_relative_transform(id_of_chunk, registered_chunk)
            )
            self.edges.append(edge)
            self.children_dict[id_of_chunk].append(registered_chunk)
        
        
    def get_relative_transform(self, parent_id: int, child_id: int) -> pp.Sim3:
        parent_preds = self.get_chunk_prediction(parent_id)
        child_preds = self.get_chunk_prediction(child_id)

        sim3 = self.aligning_class().fit(parent_preds, child_preds).sim3
        return sim3.aspypose()

    def optimize(self, n_iter):
        if not self.__requires_optimization:
            raise RuntimeWarning("Tried to optimize graph when its not required.")
        output = super().optimize(n_iter)
        self.__requires_optimization = False
        return output

    def get_absolute_transform(self, ids: Iterable[int]) -> List[Sim3]:
        vertices = self.vertices
        return [Sim3.from_pypose(vertices[i].estimate) for i in ids]
