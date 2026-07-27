# Abstract Base Classes (ABC)

from abc import ABC, abstractmethod
from typing import List
import torch
from .entities import SceneGraphDTO, NodeEntity

class IDataParser(ABC):
    """Kişi 2'nin uygulaması gereken arayüz"""
    
    @abstractmethod
    def get_scene_graph(self, image_id: str) -> SceneGraphDTO:
        pass

    @abstractmethod
    def get_all_image_ids(self) -> List[str]:
        pass


class IVisionExtractor(ABC):
    """Kişi 3'ün uygulaması gereken arayüz"""
    
    @abstractmethod
    def extract_node_features(self, image_path: str, nodes: List[NodeEntity]) -> torch.Tensor:
        """Görüntüyü ve düğümleri alır, N x 512 boyutunda tensör döner."""
        pass