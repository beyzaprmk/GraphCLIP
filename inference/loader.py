from __future__ import annotations

from pathlib import Path

import torch


class CheckpointLoader:
    """
    Loads a trained GraphCLIP checkpoint.
    """

    def __init__(
        self,
        model,
        device: str | None = None,
    ):
        self.model = model

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    def load(self, checkpoint_path: str | Path):
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint bulunamadı: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        return self.model