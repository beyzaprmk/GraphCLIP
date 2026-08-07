import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionHead(nn.Module):
 

    def __init__(
        self,
        embedding_dim: int = 512,
        dropout: float = 0.1
    ):

        super().__init__()

        self.projection = nn.Sequential(

            nn.Linear(
                embedding_dim,
                embedding_dim
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                embedding_dim,
                embedding_dim
            )

        )

    def forward(
        self,
        graph_embedding: torch.Tensor
    ) -> torch.Tensor:

        fused = self.projection(

            graph_embedding

        )

        fused = F.normalize(

            fused,

            p=2,

            dim=-1

        )

        return fused