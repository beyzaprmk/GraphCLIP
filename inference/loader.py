from __future__ import annotations

import json
from pathlib import Path

import torch

from inference.factory import GraphCLIPFactory


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


def _resolve_device(
    device: str | None = None,
) -> torch.device:

    if device is not None:
        return torch.device(device)

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


class CheckpointLoader:
   
    def __init__(
        self,
        device: str | None = None,
    ):

        self.device = _resolve_device(
            device
        )

        
        self.vocab_path = (
            PROJECT_ROOT
            / "artifacts"
            / "graphclip-base"
            / "final_vocab.json"
        )

    def load(
        self,
        checkpoint_path: str | Path,
    ):

        checkpoint_path = Path(
            checkpoint_path
        )

       
        if not checkpoint_path.is_absolute():

            checkpoint_path = (
                PROJECT_ROOT
                / checkpoint_path
            )

        checkpoint_path = checkpoint_path.resolve()

        
        if not checkpoint_path.exists():

            raise FileNotFoundError(
                "GraphCLIP checkpoint not found.\n"
                f"Checkpoint: {checkpoint_path}"
            )

        if not checkpoint_path.is_file():

            raise ValueError(
                "GraphCLIP checkpoint path is not a file.\n"
                f"Path: {checkpoint_path}"
            )

        if checkpoint_path.suffix.lower() != ".pt":

            raise ValueError(
                "Invalid GraphCLIP checkpoint format.\n"
                f"File: {checkpoint_path}\n"
                "Expected a .pt file."
            )

       
        if not self.vocab_path.exists():

            raise FileNotFoundError(
                "Relation vocabulary not found.\n"
                f"Vocabulary: {self.vocab_path}\n\n"
                "The current V1 checkpoint format does not "
                "contain model configuration, so the "
                "GraphCLIP architecture cannot be reconstructed "
                "without final_vocab.json."
            )

        
        model = GraphCLIPFactory.create(
            vocab_path=self.vocab_path,
        )

        
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if not isinstance(
            checkpoint,
            dict,
        ):

            raise ValueError(
                "Invalid GraphCLIP checkpoint.\n"
                "Expected a checkpoint dictionary."
            )

        if "model_state_dict" not in checkpoint:

            raise ValueError(
                "Invalid GraphCLIP checkpoint.\n"
                "Missing key: model_state_dict"
            )

       
        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        
        model = model.to(
            self.device
        )

        
        model.eval()

       
    
        return model

class ArtifactLoader:
   
    REQUIRED_FILES = (
        "model.pt",
        "config.json",
        "metadata.json",
        "final_vocab.json",
    )

    REQUIRED_CONFIG = (
        "architecture",
        "clip_model",
        "node_dim",
        "hidden_dim",
        "edge_dim",
        "num_relations",
        "embedding_dim",
        "dropout",
        "fusion_dropout",
    )

    def __init__(
        self,
        device: str | None = None,
    ):

        self.device = _resolve_device(
            device
        )

    def load(
        self,
        artifact_dir: str | Path,
    ):

        artifact_dir = Path(
            artifact_dir
        )

        # Resolve relative path from project root
       

        if not artifact_dir.is_absolute():

            artifact_dir = (
                PROJECT_ROOT
                / artifact_dir
            )

        artifact_dir = artifact_dir.resolve()

        # VALIDATE ARTIFACT DIRECTORY

        if not artifact_dir.exists():

            raise FileNotFoundError(
                "GraphCLIP artifact not found.\n"
                f"Directory: {artifact_dir}"
            )

        if not artifact_dir.is_dir():

            raise ValueError(
                "GraphCLIP artifact path is not a directory.\n"
                f"Path: {artifact_dir}"
            )

       
        missing_files = [
            filename
            for filename in self.REQUIRED_FILES
            if not (
                artifact_dir / filename
            ).exists()
        ]

        if missing_files:

            raise FileNotFoundError(
                "Invalid GraphCLIP artifact.\n"
                f"Directory: {artifact_dir}\n"
                f"Missing files: "
                f"{', '.join(missing_files)}"
            )

        model_path = (
            artifact_dir / "model.pt"
        )

        config_path = (
            artifact_dir / "config.json"
        )

        vocab_path = (
            artifact_dir / "final_vocab.json"
        )

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            config = json.load(
                file
            )

        if not isinstance(
            config,
            dict,
        ):

            raise ValueError(
                "Invalid GraphCLIP config.\n"
                f"File: {config_path}"
            )

       
        missing_config = [
            key
            for key in self.REQUIRED_CONFIG
            if key not in config
        ]

        if missing_config:

            raise ValueError(
                "Invalid GraphCLIP config.\n"
                f"Missing fields: "
                f"{', '.join(missing_config)}"
            )

        if config["architecture"] != "GraphCLIP":

            raise ValueError(
                "Invalid model architecture: "
                f"{config['architecture']}"
            )

       
        if not vocab_path.exists():

            raise FileNotFoundError(
                "Relation vocabulary not found "
                "inside GraphCLIP artifact.\n"
                f"Expected: {vocab_path}"
            )

        
        model = GraphCLIPFactory.create(
            vocab_path=vocab_path,
            model_name=config["clip_model"],
            node_dim=config["node_dim"],
            relation_dim=config["edge_dim"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
        )

       
        state_dict = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True,
        )

        
        if (
            isinstance(state_dict, dict)
            and "model_state_dict" in state_dict
        ):

            
            state_dict = state_dict[
                "model_state_dict"
            ]

        if not isinstance(
            state_dict,
            dict,
        ):

            raise ValueError(
                "Invalid GraphCLIP model weights.\n"
                f"File: {model_path}"
            )

        
        model.load_state_dict(
            state_dict
        )

        
        model = model.to(
            self.device
        )

        
        model.eval()

        return model



def load_model(
    source: str | Path,
    device: str | None = None,
):

    source = str(
        source
    )

    source_path = Path(
        source
    )

    
    if source_path.exists():

       
        if source_path.is_file():

            if source_path.suffix.lower() != ".pt":

                raise ValueError(
                    "Unsupported GraphCLIP model file.\n"
                    f"File: {source_path}\n\n"
                    "Expected a .pt checkpoint."
                )

            loader = CheckpointLoader(
                device=device,
            )

            return loader.load(
                source_path
            )

       
        if source_path.is_dir():

            loader = ArtifactLoader(
                device=device,
            )

            return loader.load(
                source_path
            )

    print(
        f"[GraphCLIP] Local model not found: "
        f"{source}"
    )

    print(
        f"[GraphCLIP] Trying Hugging Face Hub: "
        f"{source}"
    )

    from model.hub import download_model

    model_dir = download_model(
        repo_id=source,
        cache_dir="artifacts",
    )


    loader = ArtifactLoader(
        device=device,
    )

    return loader.load(
        model_dir
    )