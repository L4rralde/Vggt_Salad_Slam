from typing import Any, Dict
from collections import deque


class FIFOCache:
    def __init__(self, max_size: int|None = None):
        self.max_size = max_size
        self.cache_dict: Dict[Any, Any] = {}
        self.fifo = deque()

    def clear(self) -> None:
        self.cache_dict = {}
        self.fifo = deque()

    def __contains__(self, key: int) -> bool:
        return key in self.cache_dict

    def get(self, key: Any) -> Any:
        return self.cache_dict[key]

    def append(self, key: Any, value: Any) -> None:
        if len(self.fifo) == self.max_size:
            key_to_remove = self.fifo.popleft()
            self.cache_dict.pop(key_to_remove)
        
        self.fifo.append(key)
        self.cache_dict[key] = value
