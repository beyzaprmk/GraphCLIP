import sys
from pathlib import Path

# Proje ana dizinini (GraphCLIP) Python'ın arama yoluna ekliyoruz.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from dataclasses import replace
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from core.interfaces import IVisionExtractor
from core.entities import SceneGraph
from vision.cropper import ImageCropper # Kırpıcıyı içe aktardık

class ViTFeatureExtractor(IVisionExtractor):
    """
    Görsel Öznitelik Çıkarıcı:
    SceneGraph içindeki her bir düğümü kırpar, ViT'ten geçirir ve 
    zenginleştirilmiş yeni bir SceneGraph döndürür.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Vision Extractor yükleniyor ({model_name}) - Çalıştırma Cihazı: {self.device}...")
        
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name, use_safetensors=True).to(self.device)
        self.model.eval()
        print("Vision Extractor başarıyla yüklendi ve hazır!")

    def extract_features(self, graph: SceneGraph, image_path: str) -> SceneGraph:
        """
        Arayüzün zorunlu tuttuğu metot. Ham grafı alır, işler ve döndürür.
        """
        yeni_nodelar = []
        
        for node in graph.nodes:
            # 1. Yardımcı sınıfımızla resmi kırp
            cropped_image = ImageCropper.crop_bounding_box(image_path, node.bbox)
            
            # 2. Görüntüyü tensöre çevir
            inputs = self.processor(images=cropped_image, return_tensors="pt").to(self.device)
            
            # 3. ViT modelinden vektörü çek
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_features = outputs.image_embeds
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                feature_tensor = image_features.squeeze(0).cpu()
                
            # 4. Node sınıfı frozen olduğu için replace ile yeni bir nesne oluştur
            guncellenmis_node = replace(node, feature_tensor=feature_tensor)
            yeni_nodelar.append(guncellenmis_node)
            
        # Güncellenmiş düğümlerle yeni grafı döndür
        return replace(graph, nodes=yeni_nodelar)



if __name__ == "__main__":
    # Test için gerekli sahte verileri içe aktaralım
    from core.entities import Node, BoundingBox
    import os

    print("--- Extractor Testi Başlıyor ---")
    
    # 1. Test resminin yolunu belirle (proje klasöründeki test_verified.jpg)
    # Eğer resim farklı bir klasördeyse burayı ona göre güncelle
    test_image_path = "test_verified.jpg" 
    
    if not os.path.exists(test_image_path):
        print(f"HATA: {test_image_path} bulunamadı! Lütfen resmi proje ana dizinine koyun.")
    else:
        # 2. Sahte bir BoundingBox ve Node (Düğüm) oluşturalım
        # Örnek: x_min=50, y_min=50, x_max=200, y_max=200 pikselleri arası
        dummy_bbox = BoundingBox(x_min=50.0, y_min=50.0, x_max=200.0, y_max=200.0)
        dummy_node = Node(node_id=0, label="test_objesi", bbox=dummy_bbox)
        
        # 3. Sahte bir SceneGraph oluşturalım (Kenarlar/Edges şimdilik boş olabilir)
        dummy_graph = SceneGraph(image_id="test_1", nodes=[dummy_node], edges=[])
        
        # 4. Extractor sınıfını başlatalım (RTX 5060 CUDA ile çalışacak)
        extractor = ViTFeatureExtractor()
        
        # 5. İşlemi tetikleyelim
        sonuc_graph = extractor.extract_features(graph=dummy_graph, image_path=test_image_path)
        
        # 6. Sonuçları kontrol edelim
        print("\n--- Test Sonuçları ---")
        islenen_node = sonuc_graph.nodes[0]
        if islenen_node.feature_tensor is not None:
            print("BAŞARILI! Feature Tensor oluşturuldu.")
            print(f"Tensor Boyutu (Shape): {islenen_node.feature_tensor.shape} (Beklenen: torch.Size([512]))")
        else:
            print("BAŞARISIZ! Tensor hala None dönüyor.")