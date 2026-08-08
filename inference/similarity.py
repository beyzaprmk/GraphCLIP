from __future__ import annotations

import torch
import torch.nn.functional as F


class Similarity:

    @staticmethod
    def cosine(
        image_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:

        image_embedding = F.normalize(
            image_embedding,
            dim=-1,
        )

        text_embedding = F.normalize(
            text_embedding,
            dim=-1,
        )

        return image_embedding @ text_embedding.T