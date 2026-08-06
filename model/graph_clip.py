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
    

    def __init__(
        self,
        graph_encoder: GraphEncoder,
        fusion_head: FusionHead,
        model_name: str = "openai/clip-vit-base-patch32"
    ):

        super().__init__()

        self.clip = CLIPModel.from_pretrained(model_name)

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        self.graph_encoder = graph_encoder

        self.fusion_head = fusion_head

    def forward(
        self,
        image_tensor: torch.Tensor,
        text: list[str],
        graph_data: Data
    ) -> dict[str, torch.Tensor]:

        # Vision Encoder
        

        vision_outputs = self.clip.vision_model(
            pixel_values=image_tensor
        )

        image_embedding = self.clip.visual_projection(
            vision_outputs.pooler_output
        )

        image_embedding = F.normalize(
            image_embedding,
            dim=-1
        )

       
        # Text Encoder
        

        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        tokens = {
            key: value.to(image_tensor.device)
            for key, value in tokens.items()
        }

        text_outputs = self.clip.text_model(**tokens)

        text_embedding = self.clip.text_projection(
            text_outputs.pooler_output
        )

        text_embedding = F.normalize(
            text_embedding,
            dim=-1
        )

    
        # Graph Encoder
        
        graph_data = graph_data.to(image_tensor.device)

        graph_embedding = self.graph_encoder(
            graph_data
        )

        graph_embedding = F.normalize(
            graph_embedding,
            dim=-1
        )

    
        # Fusion
      
        fused_embedding = self.fusion_head(
            image_embedding=image_embedding,
            graph_embedding=graph_embedding
        )

        return {

            "image_embedding": image_embedding,

            "graph_embedding": graph_embedding,

            "fused_embedding": fused_embedding,

            "text_embedding": text_embedding

        }