from __future__ import annotations

import os
from pathlib import Path
from dataclasses import replace

import cv2
import torch

from transformers import (
    CLIPProcessor,
    CLIPVisionModelWithProjection
)

from core.entities import SceneGraph
from core.interfaces import IVisionExtractor

from vision.cropper import ImageCropper


class ViTFeatureExtractor(IVisionExtractor):
    

    def __init__(

        self,

        images_dir: str,

        model_name: str = "openai/clip-vit-base-patch32",

        device: str | None = None

    ):

        self.images_dir = Path(images_dir)

        self.device = self._resolve_device(device)

        print("=" * 60)
        print("Vision Extractor")
        print(f"Model : {model_name}")
        print(f"Device: {self.device}")
        print("=" * 60)

        self.processor = CLIPProcessor.from_pretrained(
            model_name
        )

        self.model = (
            CLIPVisionModelWithProjection
            .from_pretrained(
                model_name,
                use_safetensors=True
            )
            .to(self.device)
            .eval()
        )

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _resolve_device(
        device: str | None
    ) -> str:

        if device is not None:
            return device

        if torch.backends.mps.is_available():
            return "mps"

        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    def extract_features(

        self,

        graph: SceneGraph

    ) -> SceneGraph:

        image_path = (
            self.images_dir /
            f"{graph.image_id}.jpg"
        )

        image = cv2.imread(str(image_path))

        if image is None:

            raise FileNotFoundError(

                f"Image not found: {image_path}"

            )

        enriched_nodes = []

        for node in graph.nodes:

            feature = self._extract_node_feature(

                image=image,

                node=node

            )

            enriched_nodes.append(

                replace(

                    node,

                    feature_tensor=feature

                )

            )

        return replace(

            graph,

            nodes=enriched_nodes

        )

    def _extract_node_feature(

        self,

        image,

        node

    ) -> torch.Tensor:

        cropped = ImageCropper.crop_bounding_box(

            image,

            node.bbox

        )

        inputs = self.processor(

            images=cropped,

            return_tensors="pt"

        )

        inputs = {

            key: value.to(self.device)

            for key, value in inputs.items()

        }

        with torch.no_grad():

            outputs = self.model(

                **inputs

            )

            feature = outputs.image_embeds.squeeze(0)

            feature = torch.nn.functional.normalize(

                feature,

                p=2,

                dim=0

            )

        return feature.cpu()