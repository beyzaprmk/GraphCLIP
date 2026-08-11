from __future__ import annotations

import json
from pathlib import Path

import torch

from inference.factory import GraphCLIPFactory


# GraphCLIP repository root
PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)


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


def _resolve_local_path(
    path: str | Path,
) -> Path:

    path = Path(path)

    if path.is_absolute():
        return path.resolve()

   
    cwd_path = (
        Path.cwd() / path
    ).resolve()

    if cwd_path.exists():
        return cwd_path

    
    project_path = (
        PROJECT_ROOT / path
    ).resolve()

    if project_path.exists():
        return project_path

    return cwd_path


def _attach_model_metadata(
    model,
    artifact_dir: Path | None = None,
    vocab_path: Path | None = None,
):

    model._artifact_dir = (
        artifact_dir.resolve()
        if artifact_dir is not None
        else None
    )

    model._vocab_path = (
        vocab_path.resolve()
        if vocab_path is not None
        else None
    )

    return model


class CheckpointLoader:

    def __init__(
        self,
        device: str | None = None,
    ):

        self.device = _resolve_device(
            device
        )

    def _resolve_vocab(
        self,
        checkpoint_path: Path,
    ) -> Path:

        checkpoint_path = (
            checkpoint_path.resolve()
        )

        candidates = [

            # Checkpoint'in bulunduğu klasör
            checkpoint_path.parent
            / "final_vocab.json",

            # Kullanıcının projesindeki artifact
            Path.cwd()
            / "artifacts"
            / "graphclip-base"
            / "final_vocab.json",

            # GraphCLIP repository artifact'i
            PROJECT_ROOT
            / "artifacts"
            / "graphclip-base"
            / "final_vocab.json",

            # Kullanıcının relation resource'u
            Path.cwd()
            / "relation"
            / "resources"
            / "final_vocab.json",

            # GraphCLIP relation resource'u
            PROJECT_ROOT
            / "relation"
            / "resources"
            / "final_vocab.json",
        ]

        for path in candidates:

            if path.exists():
                return path.resolve()

        raise FileNotFoundError(
            "Relation vocabulary not found for "
            "GraphCLIP checkpoint.\n\n"
            "Checked:\n"
            + "\n".join(
                f"  - {path.resolve()}"
                for path in candidates
            )
            + "\n\n"
            "The V1 checkpoint format requires "
            "final_vocab.json."
        )

    def load(
        self,
        checkpoint_path: str | Path,
    ):

        checkpoint_path = _resolve_local_path(
            checkpoint_path
        )

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

        vocab_path = self._resolve_vocab(
            checkpoint_path
        )

        model = GraphCLIPFactory.create(
            vocab_path=vocab_path,
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
            checkpoint["model_state_dict"]
        )

        model = model.to(
            self.device
        )

        model.eval()

        return _attach_model_metadata(
            model=model,
            artifact_dir=vocab_path.parent,
            vocab_path=vocab_path,
        )


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

        artifact_dir = _resolve_local_path(
            artifact_dir
        )

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

            config = json.load(file)

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

            state_dict = (
                state_dict["model_state_dict"]
            )

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

        return _attach_model_metadata(
            model=model,
            artifact_dir=artifact_dir,
            vocab_path=vocab_path,
        )


def load_model(
    source: str | Path,
    device: str | None = None,
):

    source_path = Path(
        source
    )


    if source_path.is_absolute():

        if source_path.exists():

            if source_path.is_file():

                return CheckpointLoader(
                    device=device,
                ).load(
                    source_path
                )

            if source_path.is_dir():

                return ArtifactLoader(
                    device=device,
                ).load(
                    source_path
                )


    if not source_path.is_absolute():

        # First: user's current project
        cwd_path = (
            Path.cwd() / source_path
        ).resolve()

        if cwd_path.exists():

            if cwd_path.is_file():

                return CheckpointLoader(
                    device=device,
                ).load(
                    cwd_path
                )

            if cwd_path.is_dir():

                return ArtifactLoader(
                    device=device,
                ).load(
                    cwd_path
                )

        project_path = (
            PROJECT_ROOT / source_path
        ).resolve()

        if project_path.exists():

            if project_path.is_file():

                return CheckpointLoader(
                    device=device,
                ).load(
                    project_path
                )

            if project_path.is_dir():

                return ArtifactLoader(
                    device=device,
                ).load(
                    project_path
                )

    # HUGGING FACE HUB

    from model.hub import download_model

    model_dir = download_model(
        repo_id=str(source),
        cache_dir="artifacts",
    )

    return ArtifactLoader(
        device=device,
    ).load(
        model_dir
    )