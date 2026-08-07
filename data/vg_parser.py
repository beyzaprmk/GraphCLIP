import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
from typing import Dict, List
from core.interfaces import IDataParser #[cite: 3]
from core.entities import SceneGraph, Node, Edge, BoundingBox #[cite: 2]


class VisualGenomeParser(IDataParser): # IDataParser arayüzünü (interface) miras alıyoruz
   
    def __init__(self, objects_json_path: str, relationships_json_path: str):
        
        print("JSON dosyaları belleğe yükleniyor...")
        
        # 1. objects.json dosyasını oku ve Python listesine çevir
        with open(objects_json_path, 'r', encoding='utf-8') as f:
            objects_data = json.load(f)
            
        # 2. relationships.json dosyasını oku ve Python listesine çevir
        with open(relationships_json_path, 'r', encoding='utf-8') as f:
            relationships_data = json.load(f)
            
        self.objects_dict = {str(item['image_id']): item['objects'] for item in objects_data}
        self.relationships_dict = {str(item['image_id']): item['relationships'] for item in relationships_data}
        
        print("JSON verileri başarıyla hazırlandı!")

    def parse_image(self, image_id: str) -> SceneGraph: # interfaces.py'daki zorunlu metot[cite: 3, 7]
        
        raw_objects = self.objects_dict.get(str(image_id), [])
        raw_relationships = self.relationships_dict.get(str(image_id), [])
        
        nodes: List[Node] = []
        edges: List[Edge] = []
        
        
        vg_id_to_node_id = {}
        
        for idx, obj in enumerate(raw_objects):
            bbox = BoundingBox(
                x_min=float(obj['x']),
                y_min=float(obj['y']),
                x_max=float(obj['x'] + obj['w']),
                y_max=float(obj['y'] + obj['h'])
            )
            
            # Nesnenin etiketini çekiyoruz (Örn: 'cheese', 'pizza', 'man')
            label = obj.get('names', [obj.get('name', 'unknown')])[0]
           
            node = Node(
                node_id=idx,          # 0, 1, 2, 3... şeklinde düzenli ID
                label=label,          # Nesnenin adı
                bbox=bbox,            # Resimdeki koordinat kutusu
                feature_tensor=None   # Vektör henüz boş
            )
            nodes.append(node)
            
            vg_id_to_node_id[obj['object_id']] = idx
            
        
        for rel in raw_relationships:
            subj_id = rel['subject']['object_id'] # İlişkiyi başlatan nesne (Örn: Adam)
            obj_id = rel['object']['object_id']   # İlişkinin hedefi olan nesne (Örn: Pizza)
            
            if subj_id in vg_id_to_node_id and obj_id in vg_id_to_node_id:
                edge = Edge(
                    source_id=vg_id_to_node_id[subj_id],  # Başlangıç düğüm indeksi
                    target_id=vg_id_to_node_id[obj_id],    # Hedef düğüm indeksi
                    relation_label=rel['predicate'],      # İlişkinin türü (Örn: 'looking at', 'on')
                    spatial_features=None
                )
                edges.append(edge)
       
        return SceneGraph(
            image_id=str(image_id),
            nodes=nodes,
            edges=edges
        )

