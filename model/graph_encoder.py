import torch
import torch.nn as nn


class RelationEmbedding(nn.Module):
    """
    Relation ID'lerini öğrenilebilir embedding vektörlerine dönüştürür.

    """

    def __init__(
        self,
        num_relations: int,
        embedding_dim: int = 64
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_relations,
            embedding_dim=embedding_dim
        )

    def forward(
        self,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:

        return self.embedding(edge_attr)


import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class GraphBackbone(nn.Module):

    def __init__(

        self,

        node_dim=512,

        edge_dim=64,

        hidden_dim=512,

        dropout=0.1

    ):

        super().__init__()

        self.conv1 = TransformerConv(

            in_channels=node_dim,

            out_channels=hidden_dim,

            edge_dim=edge_dim

        )

        self.conv2 = TransformerConv(

            in_channels=hidden_dim,

            out_channels=hidden_dim,

            edge_dim=edge_dim

        )

        self.dropout = nn.Dropout(dropout)

    def forward(

        self,

        x,

        edge_index,

        edge_attr

    ):

        x = self.conv1(

            x,

            edge_index,

            edge_attr

        )

        x = F.gelu(x)

        x = self.dropout(x)

        x = self.conv2(

            x,

            edge_index,

            edge_attr

        )

        return x    



from torch_geometric.nn import global_mean_pool
import torch.nn as nn


class GraphEncoder(nn.Module):
    

    def __init__(
        self,
        relation_embedding: RelationEmbedding,
        backbone: GraphBackbone
    ):
        super().__init__()

        self.relation_embedding = relation_embedding
        self.backbone = backbone

    def forward(self, data):

        edge_embedding = self.relation_embedding(
            data.edge_attr
        )

        node_embedding = self.backbone(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=edge_embedding
        )

        graph_embedding = global_mean_pool(
            node_embedding,
            data.batch
        )

        return graph_embedding