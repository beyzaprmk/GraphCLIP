from __future__ import annotations

import torch
from torch_geometric.data import Data

from core.entities import SceneGraph


class GraphConverter:
    """
    SceneGraph -> PyTorch Geometric Data dönüştürücüsü.
    """

    def convert(self, graph: SceneGraph) -> Data:
        

        x = self._build_node_features(graph)

        edge_index = self._build_edge_index(graph)

        edge_attr = self._build_edge_attr(graph)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

    def _build_node_features(
        self,
        graph: SceneGraph
    ) -> torch.Tensor:

        features = []

        for node in graph.nodes:

            if node.feature_tensor is None:
                raise ValueError(
                    f"Node {node.node_id} has no feature tensor."
                )

            features.append(node.feature_tensor)

        return torch.stack(features)

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

        return torch.zeros(
            len(graph.edges),
            dtype=torch.long
        )