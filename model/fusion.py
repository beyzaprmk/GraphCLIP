import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionHead(nn.Module):
    """
    Image ve Graph embedding'lerini tek bir embedding'de birleştirir.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        fusion_dim = embedding_dim * 2

        self.projection = nn.Sequential(

            nn.Linear(fusion_dim, embedding_dim),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(embedding_dim, embedding_dim)

        )

    def forward(
        self,
        image_embedding: torch.Tensor,
        graph_embedding: torch.Tensor
    ) -> torch.Tensor:

        fused = torch.cat(
            [
                image_embedding,
                graph_embedding
            ],
            dim=-1
        )

        fused = self.projection(fused)

        fused = F.normalize(
            fused,
            p=2,
            dim=-1
        )

        return fused