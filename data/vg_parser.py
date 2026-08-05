import sys
from pathlib import Path

# Proje ana dizinini (GraphCLIP) Python'ın arama yoluna ekliyoruz.
# Bu sayede betiği nereden çalıştırırsak çalıştıralım 'core' klasörü sorunsuz bulunur.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
from typing import Dict, List
from core.interfaces import IDataParser #[cite: 3]
from core.entities import SceneGraph, Node, Edge, BoundingBox #[cite: 2]


class VisualGenomeParser(IDataParser): # IDataParser arayüzünü (interface) miras alıyoruz
    """
    Visual Genome veri setindeki ham JSON dosyalarını okuyarak,
    her bir resim için 'SceneGraph' nesnesi oluşturan ayrıştırıcı sınıf.
    """
    def __init__(self, objects_json_path: str, relationships_json_path: str):
        """
        Sınıf ilk çalıştırıldığında devasa JSON dosyalarını belleğe bir kez yükler 
        ve hızlı erişim için sözlük (Dictionary) yapısına dönüştürür.
        """
        print("JSON dosyaları belleğe yükleniyor...")
        
        # 1. objects.json dosyasını oku ve Python listesine çevir
        with open(objects_json_path, 'r', encoding='utf-8') as f:
            objects_data = json.load(f)
            
        # 2. relationships.json dosyasını oku ve Python listesine çevir
        with open(relationships_json_path, 'r', encoding='utf-8') as f:
            relationships_data = json.load(f)
            
        # 3. Arama performansını O(1) yapmak için image_id değerlerini anahtar (key) yapıyoruz.
        # Bu sayede yüzbinlerce resim arasından aradığımız resmi döngüye girmeden anında bulabiliriz.
        self.objects_dict = {str(item['image_id']): item['objects'] for item in objects_data}
        self.relationships_dict = {str(item['image_id']): item['relationships'] for item in relationships_data}
        
        print("JSON verileri başarıyla hazırlandı!")

    def parse_image(self, image_id: str) -> SceneGraph: # interfaces.py'daki zorunlu metot[cite: 3, 7]
        """
        Verilen bir resim ID'si için ham nesne ve ilişki verilerini alıp,
        matematiksel bir graf nesnesi (SceneGraph) inşa eder.
        """
        # Verilen resim ID'sine ait ham nesne ve ilişki listelerini sözlükten çekiyoruz
        raw_objects = self.objects_dict.get(str(image_id), [])
        raw_relationships = self.relationships_dict.get(str(image_id), [])
        
        nodes: List[Node] = []
        edges: List[Edge] = []
        
        # Visual Genome'un kendi özgün 'object_id' değerlerini, bizim grafımızda 
        # 0, 1, 2... şeklinde kullanacağımız düzenli Node indeksleriyle eşleştirmek için yardım sözlüğü
        vg_id_to_node_id = {}
        
        # =========================================================================
        # ADIM 1: DÜĞÜMLERİ (NODES) OLUŞTURMA
        # =========================================================================
        for idx, obj in enumerate(raw_objects):
            # Visual Genome koordinatları x, y, w, h şeklinde verir.
            # Biz bunu x_min, y_min, x_max, y_max formatındaki BoundingBox nesnesine dönüştürüyoruz[cite: 2, 6].
            bbox = BoundingBox(
                x_min=float(obj['x']),
                y_min=float(obj['y']),
                x_max=float(obj['x'] + obj['w']),
                y_max=float(obj['y'] + obj['h'])
            )
            
            # Nesnenin etiketini çekiyoruz (Örn: 'cheese', 'pizza', 'man')
            label = obj.get('names', [obj.get('name', 'unknown')])[0]
            
            # 'entities.py' içinde tanımlanan Node nesnesini üretiyoruz[cite: 2, 6].
            # 'feature_tensor' alanını şimdilik None bırakıyoruz çünkü resim kırpma ve 
            # ViT vektör çıkarma işlemi henüz yapılmadı (bir sonraki adımımız)[cite: 2, 6].
            node = Node(
                node_id=idx,          # 0, 1, 2, 3... şeklinde düzenli ID
                label=label,          # Nesnenin adı
                bbox=bbox,            # Resimdeki koordinat kutusu
                feature_tensor=None   # Vektör henüz boş
            )
            nodes.append(node)
            
            # Haritalamayı kaydediyoruz: (VG Orijinal ID -> Bizim Node ID'miz)
            vg_id_to_node_id[obj['object_id']] = idx
            
        # =========================================================================
        # ADIM 2: KENARLARI (EDGES / İLİŞKİLERİ) OLUŞTURMA
        # =========================================================================
        for rel in raw_relationships:
            subj_id = rel['subject']['object_id'] # İlişkiyi başlatan nesne (Örn: Adam)
            obj_id = rel['object']['object_id']   # İlişkinin hedefi olan nesne (Örn: Pizza)
            
            # Eğer ilişkinin her iki tarafındaki nesne de düğüm listemizde mevcutsa kenarı (Edge) kuruyoruz
            if subj_id in vg_id_to_node_id and obj_id in vg_id_to_node_id:
                edge = Edge(
                    source_id=vg_id_to_node_id[subj_id],  # Başlangıç düğüm indeksi
                    target_id=vg_id_to_node_id[obj_id],    # Hedef düğüm indeksi
                    relation_label=rel['predicate'],      # İlişkinin türü (Örn: 'looking at', 'on')
                    spatial_features=None
                )
                edges.append(edge)
                
        # =========================================================================
        # ADIM 3: NİHAİ SCENEGRAPH PAKETİNİ DÖNDÜRME
        # =========================================================================
        # Oluşturulan tüm düğüm ve kenarları tek bir taşıyıcı nesneye koyup geri döndürüyoruz[cite: 2, 6].
        return SceneGraph(
            image_id=str(image_id),
            nodes=nodes,
            edges=edges
        )

