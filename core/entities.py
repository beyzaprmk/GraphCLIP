# SceneGraph, Node, Edge sınıfları (Dataclasses)

from dataclasses import dataclass
from typing import List, Tuple
import torch

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

@dataclass
class NodeEntity:
    node_id: int
    label: str
    bbox: BoundingBox
    # Görüntü işlemecisi burayı sonradan dolduracak
    embedding: torch.Tensor = None 

@dataclass
class EdgeEntity:
    source_id: int
    target_id: int
    relation: str

@dataclass
class SceneGraphDTO:
    image_id: str
    image_path: str
    nodes: List[NodeEntity]
    edges: List[EdgeEntity]