from abc import ABC, abstractmethod
from typing import List
from core.entities import SceneGraph

class IDataParser(ABC):
    """
    Veri ve Graf Mühendisi için zorunlu arayüz.
    Görevi: JSON dosyalarını okuyup ilişkileri ayrıştırmak.
    """
    @abstractmethod
    def parse_image(self, image_id: str) -> SceneGraph:
        """
        Girdi Bir görüntünün ID'si .
        Çıktı İçerisinde düğümlerin (labels/bboxes) ve kenarların 
               olduğu ama tensörlerin HENÜZ OLMADIĞI ham bir SceneGraph nesnesi.
        """
        pass

class IVisionExtractor(ABC):
    """
    Görüntü İşleme Sorumlusu için zorunlu arayüz.
    Görevi Ham SceneGraph'ı alıp nesneleri kırpmak ve ViT/SAM 2'den geçirmek.
    """
    @abstractmethod
    def extract_features(self, graph: SceneGraph) -> SceneGraph:
        """
        Girdi: Tensörleri eksik olan SceneGraph.
        Çıktı Kırpılan her nesnenin ViT  modelinden geçirilip 
               'feature_tensor' (512 boyutlu) alanının doldurulduğu ZENGİNLEŞTİRİLMİŞ SceneGraph.
        """
        pass

class IGraphFusionModel(ABC):
    """
   Sistem Mimarı tasarlayacağın üst seviye GraphCLIP modelinin arayüzü.
    """
    @abstractmethod
    def forward(self, graph: SceneGraph):
        """
        Girdi Tamamen doldurulmuş  SceneGraph.
        Çıktı GNN ve OpenCLIP katmanlarından geçmiş Fusion uzamsal vektör.
        """
        pass