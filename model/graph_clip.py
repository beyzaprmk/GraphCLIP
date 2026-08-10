from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    CLIPModel,
    CLIPTokenizer,
)
from torch_geometric.data import Data

from model.graph_encoder import (
    GraphEncoder,
    RelationEmbedding,
    GraphBackbone,
)
from model.fusion import FusionHead


class GraphCLIP(nn.Module):
  
    def __init__(
        self,
        graph_encoder: GraphEncoder,
        fusion_head: FusionHead,
        model_name: str = "openai/clip-vit-base-patch32",
    ):
        super().__init__()

        self.model_name = model_name

        self.clip = CLIPModel.from_pretrained(
            model_name,
            use_safetensors=True,
        )

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_name
        )

        self.graph_encoder = graph_encoder
        self.fusion_head = fusion_head

   
    def encode_text(
        self,
        text: list[str],
    ) -> torch.Tensor:

        device = next(self.parameters()).device

        tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
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
            dim=-1,
        )

        return text_embedding

    
    def encode_graph(
        self,
        graph_data: Data,
    ) -> torch.Tensor:

        device = next(self.parameters()).device

        graph_data = graph_data.to(device)

        graph_embedding = self.graph_encoder(
            graph_data
        )

        graph_embedding = F.normalize(
            graph_embedding,
            dim=-1,
        )

        fused_embedding = self.fusion_head(
            graph_embedding=graph_embedding
        )

        fused_embedding = F.normalize(
            fused_embedding,
            dim=-1,
        )

        return fused_embedding

   
    def forward(
        self,
        text: list[str],
        graph_data: Data,
    ) -> dict[str, torch.Tensor]:

        text_embedding = self.encode_text(text)

        device = next(self.parameters()).device

        graph_data = graph_data.to(device)

        graph_embedding = self.graph_encoder(
            graph_data
        )

        graph_embedding = F.normalize(
            graph_embedding,
            dim=-1,
        )

        fused_embedding = self.fusion_head(
            graph_embedding=graph_embedding
        )

        fused_embedding = F.normalize(
            fused_embedding,
            dim=-1,
        )

        return {
            "graph_embedding": graph_embedding,
            "fused_embedding": fused_embedding,
            "text_embedding": text_embedding,
        }

    
    def get_config(self) -> dict:

        relation_embedding = (
            self.graph_encoder.relation_embedding
        )

        backbone = self.graph_encoder.backbone

        fusion_projection = (
            self.fusion_head.projection
        )

        config = {
            "architecture": "GraphCLIP",
            "clip_model": self.model_name,

            "node_dim": backbone.conv1.in_channels,
            "hidden_dim": backbone.conv1.out_channels,

            "edge_dim": (
                relation_embedding.embedding.embedding_dim
            ),

            "num_relations": (
                relation_embedding.embedding.num_embeddings
            ),

            "embedding_dim": (
                fusion_projection[0].in_features
            ),

            "dropout": backbone.dropout.p,

            "fusion_dropout": (
                fusion_projection[2].p
            ),
        }

        return config

   
    def save_pretrained(
        self,
        output_dir: str | Path,
        metadata: dict | None = None,
    ) -> Path:

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = output_dir / "model.pt"
        config_path = output_dir / "config.json"
        metadata_path = output_dir / "metadata.json"

        # Model weights
       

        torch.save(
            self.state_dict(),
            model_path,
        )

       
        # Configuration
       
        config = self.get_config()

        with config_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                config,
                file,
                indent=2,
            )

        # Metadata
       

        default_metadata = {
            "name": output_dir.name,
            "architecture": "GraphCLIP",
            "framework": "pytorch",
            "format_version": "1.0",
        }

        if metadata is not None:
            default_metadata.update(metadata)

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                default_metadata,
                file,
                indent=2,
            )

        print("=" * 60)
        print("GraphCLIP model exported.")
        print(f"Directory : {output_dir}")
        print(f"Model     : {model_path}")
        print(f"Config    : {config_path}")
        print(f"Metadata  : {metadata_path}")
        print("=" * 60)

        return output_dir

  
    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: str | torch.device | None = None,
    ) -> "GraphCLIP":

        model_dir = Path(model_dir)

        if not model_dir.exists():
            raise FileNotFoundError(
                f"GraphCLIP model directory not found: {model_dir}"
            )

        model_path = model_dir / "model.pt"
        config_path = model_dir / "config.json"
        metadata_path = model_dir / "metadata.json"

        # ------------------------------------------------------
        # Validate artifact
        # ------------------------------------------------------

        required_files = [
            model_path,
            config_path,
            metadata_path,
        ]

        missing_files = [
            path.name
            for path in required_files
            if not path.exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                "Invalid GraphCLIP artifact.\n"
                f"Directory: {model_dir}\n"
                f"Missing files: {', '.join(missing_files)}"
            )

        # Load config
       

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            config = json.load(file)

        
        relation_embedding = RelationEmbedding(
            num_relations=config["num_relations"],
            embedding_dim=config["edge_dim"],
        )

        backbone = GraphBackbone(
            node_dim=config["node_dim"],
            edge_dim=config["edge_dim"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
        )

        graph_encoder = GraphEncoder(
            relation_embedding=relation_embedding,
            backbone=backbone,
            hidden_dim=config["hidden_dim"],
        )

        
        fusion_head = FusionHead(
            embedding_dim=config["embedding_dim"],
            dropout=config["fusion_dropout"],
        )

        # ------------------------------------------------------
        # Construct GraphCLIP
        # ------------------------------------------------------

        model = cls(
            graph_encoder=graph_encoder,
            fusion_head=fusion_head,
            model_name=config["clip_model"],
        )

       
        state_dict = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(
            state_dict
        )

       
        if device is not None:
            model = model.to(device)

        model.eval()

        print("=" * 60)
        print("GraphCLIP pretrained model loaded.")
        print(f"Model : {model_dir}")
        print(f"Device: {next(model.parameters()).device}")
        print("=" * 60)

        return model