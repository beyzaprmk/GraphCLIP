from __future__ import annotations

from pathlib import Path

from model.graph_encoder import (
    RelationEmbedding,
    GraphBackbone,
    GraphEncoder,
)
from model.fusion import FusionHead
from model.graph_clip import GraphCLIP
from relation.vocabulary import RelationVocabulary


class GraphCLIPFactory:
    

    @staticmethod
    def create(
        vocab_path: str | Path,
        model_name: str = "openai/clip-vit-base-patch32",
        node_dim: int = 512,
        relation_dim: int = 64,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> GraphCLIP:

        vocab = RelationVocabulary.load(
            Path(vocab_path)
        )

        num_relations = len(
            vocab.relation_to_id
        )

        relation_embedding = RelationEmbedding(
            num_relations=num_relations,
            embedding_dim=relation_dim,
        )

        backbone = GraphBackbone(
            node_dim=node_dim,
            edge_dim=relation_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        graph_encoder = GraphEncoder(
            relation_embedding=relation_embedding,
            backbone=backbone,
            hidden_dim=hidden_dim,
        )

        fusion_head = FusionHead(
            embedding_dim=hidden_dim,
            dropout=dropout,
        )

        model = GraphCLIP(
            graph_encoder=graph_encoder,
            fusion_head=fusion_head,
            model_name=model_name,
        )

        return model