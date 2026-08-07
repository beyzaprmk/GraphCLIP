from __future__ import annotations

from abc import ABC, abstractmethod

from core.entities import SceneGraph


class IDataParser(ABC):
  

    @abstractmethod
    def parse_image(
        self,
        image_id: str
    ) -> SceneGraph | None:
      
        raise NotImplementedError


class IVisionExtractor(ABC):
   
    @abstractmethod
    def extract_features(
        self,
        graph: SceneGraph
    ) -> SceneGraph:
        
        raise NotImplementedError


class IGraphFusionModel(ABC):
   

    @abstractmethod
    def forward(
        self,
        graph: SceneGraph
    ):
        
        raise NotImplementedError