# Çoklu iş parçacığı (Multi-threading) yöneticisi

import torch
from core.interfaces import IDataParser, IVisionExtractor
from model.graph_clip import MyGraphCLIPModel


class GraphClipPipeline:
    """Sistemi yönetecek olan Orkestratör Sınıf (Kişi 1)"""
    
    def __init__(self, data_parser: IDataParser, vision_extractor: IVisionExtractor):
        # Polymorphism: Gelen sınıfların içi sahte mi gerçek mi pipeline umursamaz.
        self.data_parser = data_parser
        self.vision_extractor = vision_extractor
        
        # Cihaz ayarı (Apple Silicon için mps, yoksa cpu)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Kendi yazdığın model mimarisi
        self.model = MyGraphCLIPModel().to(self.device)

    def process_single_image(self, image_id: str):
        # 1. Veri mühendisinin kodunu tetikle (JSON'dan DTO'ya)
        scene_graph = self.data_parser.get_scene_graph(image_id)
        
        # 2. Görüntü işlemecisinin kodunu tetikle (DTO'dan Tensöre)
        node_features = self.vision_extractor.extract_node_features(
            scene_graph.image_path, 
            scene_graph.nodes
        )
        
        # 3. Kendi GNN / CLIP Modelini çalıştır
        # node_features'ı PyTorch Geometric Data formatına çevir ve modele ver...
        output = self.model(node_features, scene_graph.edges)
        
        return output