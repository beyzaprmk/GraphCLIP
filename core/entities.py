from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float



@dataclass(frozen=True)
class Node:
   
    node_id: int

    label: str

    bbox: BoundingBox

    
    feature_tensor: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class Edge:
    source_id: int

    target_id: int

   
    relation_label: str

   
    relation_id: Optional[int] = None

    spatial_features: Optional[torch.Tensor] = None



@dataclass(frozen=True)
class SceneGraph:
    """
    Bir görüntünün Scene Graph gösterimi.
    """

    image_id: str

    nodes: list[Node]

    edges: list[Edge]


    def get_node_features(self) -> torch.Tensor:
        """
        Node feature tensorlerini
        [N, 512] formatında döndürür.
        """

        if len(self.nodes) == 0:

            raise ValueError(

                f"Image {self.image_id} contains no nodes."

            )

        features = []

        for node in self.nodes:

            if node.feature_tensor is None:

                raise ValueError(

                    f"Node {node.node_id} has no feature tensor."

                )

            features.append(

                node.feature_tensor.float()

            )

        return torch.stack(

            features,

            dim=0

        )


    def get_edge_index(self) -> torch.Tensor:
        """
        PyTorch Geometric edge_index üretir.
        """

        if len(self.edges) == 0:

            return torch.empty(

                (2, 0),

                dtype=torch.long

            )

        sources = [

            edge.source_id

            for edge in self.edges

        ]

        targets = [

            edge.target_id

            for edge in self.edges

        ]

        return torch.tensor(

            [

                sources,

                targets

            ],

            dtype=torch.long

        )


    def get_relation_labels(self) -> list[str]:
        """
        Bütün relation label'larını döndürür.
        """

        return [

            edge.relation_label

            for edge in self.edges

        ]


    def get_relation_ids(self) -> torch.Tensor:
       
        ids = []

        for edge in self.edges:

            if edge.relation_id is None:

                raise ValueError(

                    "Relation IDs have not been assigned."

                )

            ids.append(

                edge.relation_id

            )

        return torch.tensor(

            ids,

            dtype=torch.long

        )