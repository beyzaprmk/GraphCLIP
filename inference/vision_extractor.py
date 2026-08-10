from __future__ import annotations
import contextlib
import io
import torch
from PIL import Image
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()
from transformers import (
    CLIPProcessor,
    CLIPVisionModelWithProjection,
)
class InferenceVisionExtractor:
   

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)

        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            self.processor = CLIPProcessor.from_pretrained(
                model_name
            )

        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):

            self.model = (
                CLIPVisionModelWithProjection
                .from_pretrained(
                    model_name,
                    use_safetensors=True,
                )
                .to(self.device)
                .eval()
            )

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _resolve_device(
        device: str | None,
    ) -> str:

        if device is not None:
            return device

        if torch.backends.mps.is_available():
            return "mps"

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    @torch.no_grad()
    def extract(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
       
        if not isinstance(
            image,
            Image.Image,
        ):
            raise TypeError(
                "image bir PIL.Image.Image olmalıdır."
            )

        image = image.convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(
            **inputs
        )

        feature = outputs.image_embeds.squeeze(
            0
        )

        feature = torch.nn.functional.normalize(
            feature,
            p=2,
            dim=0,
        )

        if feature.numel() != 512:
            raise ValueError(
                "CLIP vision embedding boyutu 512 olmalıdır. "
                f"Alınan boyut: {feature.numel()}"
            )

        return feature.cpu()