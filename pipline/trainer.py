from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F


class Trainer:
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        device: str | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        checkpoint_dir: str = "checkpoints",
        temperature: float = 0.07
    ):
        self.temperature = temperature
        if device is None:

            if torch.backends.mps.is_available():

                self.device = torch.device("mps")

            elif torch.cuda.is_available():

                self.device = torch.device("cuda")

            else:

                self.device = torch.device("cpu")

        else:

            self.device = torch.device(device)

        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epochs = epochs

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.best_loss = float("inf")



    def _compute_loss(
        self,
        outputs: dict
    ) -> torch.Tensor:
       

        image_features = outputs["fused_embedding"]

        text_features = outputs["text_embedding"]

        batch_size = image_features.size(0)

        # Similarity Matrix

        logits = (image_features @ text_features.T) / self.temperature

        # Ground Truth

        targets = torch.arange(
            batch_size,
            device=self.device
        )

        # Image -> Text

        loss_i = F.cross_entropy(
            logits,
            targets
        )

        # Text -> Image

        loss_t = F.cross_entropy(
            logits.T,
            targets
        )

        loss = (loss_i + loss_t) / 2

        return loss


    

    def train_epoch(
        self
    ) -> float:

        self.model.train()

        running_loss = 0.0

        for i, batch in enumerate(self.train_loader):

            if i % 10 == 0:
                print(f"Batch {i}/{len(self.train_loader)}")

            graph = batch["graph"].to(self.device)

            text = batch["text"]

            self.optimizer.zero_grad()

            outputs = self.model(

                text,

                graph

            )

            loss = self._compute_loss(
                outputs
            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(
            self.train_loader
        )

        return epoch_loss

    def validate(
        self
    ) -> float:
        

        if self.val_loader is None:
            raise ValueError(
                "Validation DataLoader tanımlanmamış."
            )

        self.model.eval()

        running_loss = 0.0

        with torch.no_grad():

            for batch in self.val_loader:

                graph = batch["graph"].to(self.device)

                text = batch["text"]

                outputs = self.model(

                    text,

                    graph

                )
                loss = self._compute_loss(outputs)

                running_loss += loss.item()

        epoch_loss = running_loss / len(self.val_loader)

        return epoch_loss


    def save_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float
    ) -> None:
       

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scheduler_state_dict":
                self.scheduler.state_dict(),

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "best_loss":
                self.best_loss

        }

        checkpoint_path = (
            self.checkpoint_dir /
            f"epoch_{epoch}.pt"
        )

        torch.save(
            checkpoint,
            checkpoint_path
        )

        # En iyi modeli ayrıca kaydet

        if val_loss < self.best_loss:

            self.best_loss = val_loss

            best_path = (
                self.checkpoint_dir /
                "best_model.pt"
            )

            torch.save(
                checkpoint,
                best_path
            )

   

    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ) -> int:
       

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint bulunamadı: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if load_optimizer:
            self.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if load_scheduler:
            self.scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        self.best_loss = checkpoint.get(
            "best_loss",
            float("inf")
        )

        start_epoch = checkpoint.get(
            "epoch",
            0
        ) + 1

        print("=" * 60)
        print("Checkpoint yüklendi.")
        print(f"Dosya      : {checkpoint_path.name}")
        print(f"Epoch      : {start_epoch}")
        print(f"Best Loss  : {self.best_loss:.4f}")
        print("=" * 60)

        return start_epoch  


    
    def fit(self) -> None:
       
        print("=" * 60)
        print("Training started...")
        print(f"Device : {self.device}")
        print(f"Epochs: {self.epochs}")
        print("=" * 60)

        for epoch in range(1, self.epochs + 1):

            train_loss = self.train_epoch()

            if self.val_loader is not None:
                val_loss = self.validate()
            else:
                val_loss = train_loss

            self.scheduler.step()

            self.save_checkpoint(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss
            )

            print(
                f"[{epoch:03d}/{self.epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

        print("=" * 60)
        print("Training completed.")
        print("=" * 60)