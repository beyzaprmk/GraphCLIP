from __future__ import annotations

from pathlib import Path

import torch
from inference.factory import GraphCLIPFactory

class CheckpointLoader:
   
    def __init__(
        self,
        vocab_path: str | Path,
        device: str | None = None,
    ):

        project_root = (
            Path(__file__).resolve().parent.parent
        )

        vocab_path = Path(vocab_path)

        if not vocab_path.is_absolute():
            vocab_path = project_root / vocab_path

        self.vocab_path = vocab_path.resolve()

        if device is None:

            if torch.backends.mps.is_available():
                self.device = torch.device("mps")

            elif torch.cuda.is_available():
                self.device = torch.device("cuda")

            else:
                self.device = torch.device("cpu")

        else:
            self.device = torch.device(device)
    def load(
        self,
        checkpoint_path: str | Path,
    ):

        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                "GraphCLIP checkpoint not found.\n"
                f"Checkpoint: {checkpoint_path}"
            )

        if not self.vocab_path.exists():

            raise FileNotFoundError(
                "Relation vocabulary not found.\n"
                f"Vocabulary: {self.vocab_path}\n\n"
                "The current V1 checkpoint format requires "
                "final_vocab.json to reconstruct the model."
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

        print("=" * 60)
        print("GraphCLIP checkpoint loaded.")
        print(
            f"Checkpoint : {checkpoint_path}"
        )
        print(
            f"Vocabulary : {self.vocab_path}"
        )
        print(
            f"Device     : {self.device}"
        )
        print("=" * 60)

        return model