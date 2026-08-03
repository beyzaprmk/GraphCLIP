from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch

@dataclass(frozen=True)
class BoundingBox:
    """
    Görüntü üzerindeki nesnenin koordinatlarını temsil eder.
    Vision  kırpma işlemlerinde bu nesneyi kullanacak.
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float

@dataclass(frozen=True)
class Node:
    """
    Sahne grafındaki her bir nesneyi (Düğümü) temsil eder.
    Başlangıçta sadece etiket ve kutu bilgisi varken, 
    Kişi 3'ün işleminden sonra 'feature_tensor' (512 boyutlu vektör) ile dolar.
    """
    node_id: int
    label: str  
    bbox: BoundingBox
    #  N x 512 boyutlu tensör
    feature_tensor: Optional[torch.Tensor] = None 

@dataclass(frozen=True)
class Edge:
    """
    İki düğüm arasındaki uzamsal ve anlamsal ilişkiyi temsil eder.
    Veri Mühendisi bu ilişkileri Visual Genome'dan çekecek.
    """
    source_id: int          
    target_id: int         
    relation_label: str     # Örn: "üstünde", "yanında"
    # Merkez noktalar arası mesafe veya IoU gibi uzamsal özellikler eklenebilir
    spatial_features: Optional[torch.Tensor] = None 

@dataclass(frozen=True)
class SceneGraph:
    """
    Bir görüntüye ait tüm grafiksel yapıyı kapsayan ana taşıyıcı nesne.
    Sistem Mimarı modele argüman olarak alacağın nihai paket budur.
    """
    image_id: str
    nodes: List[Node]
    edges: List[Edge]
    
    def get_edge_index(self) -> torch.Tensor:
        """
        PyTorch Geometric'in beklediği [2, M] boyutundaki edge_index matrisini
        dinamik olarak oluşturur. GNN katmanına doğrudan verilebilir.
        """
        if not self.edges:
            return torch.empty((2, 0), dtype=torch.long)
            
        source_indices = [edge.source_id for edge in self.edges]
        target_indices = [edge.target_id for edge in self.edges]
        return torch.tensor([source_indices, target_indices], dtype=torch.long)

    def get_node_features(self) -> torch.Tensor:
        """
        GNN'in 'x' parametresi için tüm düğümlerin tensörlerini
        tek bir [N, 512] matrisi haline getirir.
        """
        # Özellik vektörleri henüz çıkarılmamışsa
        if not self.nodes or self.nodes[0].feature_tensor is None:
            raise ValueError(f"Image {self.image_id} için node tensörleri henüz oluşturulmamış!")
            
        # Tüm tensörleri alt alta (0. boyutta) birleştirir
        return torch.stack([node.feature_tensor for node in self.nodes])