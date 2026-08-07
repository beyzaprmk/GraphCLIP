import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    CLIPModel,
    CLIPTokenizer
)

from torch_geometric.data import Data

from model.graph_encoder import GraphEncoder
from model.fusion import FusionHead


class GraphCLIP(nn.Module):
    """
    GraphCLIP modeli.

    Bu sürümde görüntüler eğitim sırasında tekrar
    CLIP Vision Encoder'dan geçirilmez.

    Node feature'ları preprocessing aşamasında
    üretildiği için yalnızca:

        Graph -> GNN
        Text  -> CLIP Text Encoder

    çalıştırılır.
    """

    def __init__(
        self,
        graph_encoder: GraphEncoder,
        fusion_head: FusionHead,
        model_name: str = "openai/clip-vit-base-patch32"
    ):

        super().__init__()

        #
        # Sadece Text Encoder kullanılacak.
        # CLIPModel yüklenmeye devam ediyor çünkü
        # text encoder ve projection katmanlarını
        # buradan alıyoruz.
        #
        self.clip = CLIPModel.from_pretrained(
            model_name
        )

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name
        )

        self.graph_encoder = graph_encoder

        self.fusion_head = fusion_head

    def forward(
        self,
        text: list[str],
        graph_data: Data
    ) -> dict[str, torch.Tensor]:

        device = next(self.parameters()).device

        # ==================================================
        # TEXT ENCODER
        # ==================================================

        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        tokens = {

            key: value.to(device)

            for key, value in tokens.items()

        }

        text_outputs = self.clip.text_model(
            **tokens
        )

        text_embedding = self.clip.text_projection(
            text_outputs.pooler_output
        )

        text_embedding = F.normalize(

            text_embedding,

            dim=-1

        )

        # ==================================================
        # GRAPH ENCODER
        # ==================================================
        graph_data = graph_data.to(device)
        graph_embedding = self.graph_encoder(
            graph_data
        )

        graph_embedding = F.normalize(

            graph_embedding,

            dim=-1

        )

        # ==================================================
        # FUSION
        # ==================================================

        #
        # Yeni mimaride FusionHead artık
        # graph embedding'i son embedding olarak kullanıyor.
        #
        fused_embedding = self.fusion_head(

            graph_embedding=graph_embedding

        )

        fused_embedding = F.normalize(

            fused_embedding,

            dim=-1

        )

        return {

            "graph_embedding": graph_embedding,

            "fused_embedding": fused_embedding,

            "text_embedding": text_embedding

        }