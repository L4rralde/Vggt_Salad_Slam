from abc import ABC, abstractmethod
from typing import List, Dict

from PIL import Image
from torch import Tensor
from numpy import ndarray


class VggtLikeSaladSplit(ABC):
    @abstractmethod
    def preprocess_images(self, pil_img_list: List[Image.Image]) -> Tensor:
        pass

    @abstractmethod
    def views_encoding(self, img_list: List[Image.Image], **kwargs) -> Dict[str, Tensor]:
        pass

    @abstractmethod
    def sequence_encoding(self, view_preds: Dict[str, Tensor], **kwargs) -> Dict[str, Tensor]:
        pass

    @abstractmethod
    def heads_prediction(self, seq_preds: Dict[str, Tensor], **kwargs) -> Dict[str, ndarray]:
        pass

    @abstractmethod
    def chunk_prediction(self, view_preds: Dict[str, Tensor], **kwargs) -> Dict[str, ndarray]:
        pass
