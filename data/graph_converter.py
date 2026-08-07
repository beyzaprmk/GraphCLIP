from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data

from core.entities import SceneGraph


class GraphConverter:
   
    def __init__(self, feature_dir: str, relation_vocab=None):
        self.feature_dir = Path(feature_dir)
        self.relation_vocab = relation_vocab

    def convert(
        self,
        graph: SceneGraph
    ) -> Data:

        x = self._build_node_features(graph.image_id)

        edge_index = self._build_edge_index(graph)

        edge_attr = self._build_edge_attr(graph)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

    def _build_node_features(
        self,
        image_id: str
    ) -> torch.Tensor:
        

        feature_path = self.feature_dir / f"{image_id}.pt"

        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature dosyası bulunamadı: {feature_path}"
            )

        features = torch.load(
            feature_path,
            map_location="cpu"
        )

        if not isinstance(features, torch.Tensor):
            raise TypeError(
                f"{feature_path} bir Tensor içermiyor."
            )

        return features.float()

    def _build_edge_index(
        self,
        graph: SceneGraph
    ) -> torch.Tensor:

        sources = []
        targets = []

        for edge in graph.edges:

            sources.append(edge.source_id)
            targets.append(edge.target_id)

        return torch.tensor(
            [sources, targets],
            dtype=torch.long
        )

    def _build_edge_attr(
        self,
        graph: SceneGraph
    ) -> torch.Tensor:

        relation_ids = []

        for edge in graph.edges:

            relation_ids.append(edge.relation_id)

        return torch.tensor(
            relation_ids,
            dtype=torch.long
        )