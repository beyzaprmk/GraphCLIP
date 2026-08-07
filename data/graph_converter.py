from __future__ import annotations

import torch
from torch_geometric.data import Data

from core.entities import SceneGraph
from relation.vocabulary import RelationVocabulary


class GraphConverter:
    

    def __init__(
        self,
        relation_vocab: RelationVocabulary
    ):

        self.relation_vocab = relation_vocab



    def convert(
        self,
        graph: SceneGraph
    ) -> Data:

        return Data(

            x=self._build_node_features(graph),

            edge_index=self._build_edge_index(graph),

            edge_attr=self._build_edge_attr(graph)

        )


    def _build_node_features(
        self,
        graph: SceneGraph
    ) -> torch.Tensor:

        if len(graph.nodes) == 0:

            raise ValueError(

                f"Image {graph.image_id} contains no nodes."

            )

        features = []

        for node in graph.nodes:

            if node.feature_tensor is None:

                raise ValueError(

                    f"Image {graph.image_id}: "
                    f"Node {node.node_id} has no feature tensor."

                )

            features.append(

                node.feature_tensor.float()

            )

        return torch.stack(

            features,

            dim=0

        )

    def _build_edge_index(
        self,
        graph: SceneGraph
    ) -> torch.Tensor:

        if len(graph.edges) == 0:

            return torch.empty(

                (2, 0),

                dtype=torch.long

            )

        return torch.tensor(

            [

                [

                    edge.source_id

                    for edge in graph.edges

                ],

                [

                    edge.target_id

                    for edge in graph.edges

                ]

            ],

            dtype=torch.long

        )

    def _build_edge_attr(
        self,
        graph: SceneGraph
    ) -> torch.Tensor:

        if len(graph.edges) == 0:

            return torch.empty(

                (0,),

                dtype=torch.long

            )

        relation_ids = []

        for edge in graph.edges:

            relation_label = edge.relation_label.strip().lower()

            relation_id = self.relation_vocab.relation_to_id.get(

                relation_label,

                0

            )

            relation_ids.append(

                relation_id

            )

        return torch.tensor(

            relation_ids,

            dtype=torch.long

        )