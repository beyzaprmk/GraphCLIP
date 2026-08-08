from __future__ import annotations

from typing import Iterable

import torch
from PIL import Image
from transformers import (
    OwlViTForObjectDetection,
    OwlViTProcessor,
)


class OWLViTObjectDetector:
  

    def __init__(
        self,
        model_name: str = "google/owlvit-base-patch32",
        device: str | None = None,
        threshold: float = 0.10,
        max_detections_per_query: int = 3,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.max_detections_per_query = (
            max_detections_per_query
        )

        self.device = self._resolve_device(
            device
        )

        self.processor = (
            OwlViTProcessor.from_pretrained(
                self.model_name
            )
        )

        self.model = (
            OwlViTForObjectDetection.from_pretrained(
                self.model_name
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

   
    # Device

    @staticmethod
    def _resolve_device(
        device: str | None,
    ) -> torch.device:

        if device is not None:
            return torch.device(
                device
            )

        if torch.backends.mps.is_available():
            return torch.device(
                "mps"
            )

        if torch.cuda.is_available():
            return torch.device(
                "cuda"
            )

        return torch.device(
            "cpu"
        )

    # Detection

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        queries: Iterable[str],
    ) -> list[dict]:

        if not isinstance(
            image,
            Image.Image,
        ):
            raise TypeError(
                "image bir PIL.Image.Image olmalıdır."
            )

        queries = self._normalize_queries(
            queries
        )

        if not queries:
            return []

        # OWL-ViT text labels'i batch formatında bekler.
        text_labels = [
            self._build_prompt(query)
            for query in queries
        ]

        inputs = self.processor(
            text=[text_labels],
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if isinstance(
                value,
                torch.Tensor,
            )
        }

        outputs = self.model(
            **inputs
        )

        target_sizes = torch.tensor(
            [
                (
                    image.height,
                    image.width,
                )
            ],
            device=self.device,
        )

        results = (
            self.processor
            .post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.threshold,
                text_labels=[text_labels],
            )
        )

        result = results[0]

        boxes = result.get(
            "boxes",
            [],
        )

        scores = result.get(
            "scores",
            [],
        )

        detected_labels = result.get(
            "text_labels",
            [],
        )

        return self._build_detections(
            boxes=boxes,
            scores=scores,
            labels=detected_labels,
            queries=queries,
        )

    # Detection conversion

    def _build_detections(
        self,
        boxes,
        scores,
        labels,
        queries: list[str],
    ) -> list[dict]:

        detections = []

        per_query_count = {
            query: 0
            for query in queries
        }

        for box, score, label in zip(
            boxes,
            scores,
            labels,
        ):

            label = self._clean_label(
                label
            )

            if not label:
                continue

            canonical_label = (
                self._match_query_label(
                    label,
                    queries,
                )
            )

            if canonical_label is None:
                continue

            if (
                per_query_count[
                    canonical_label
                ]
                >= self.max_detections_per_query
            ):
                continue

            box = box.detach().cpu().tolist()

            if len(box) != 4:
                continue

            x1, y1, x2, y2 = box

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "id": len(detections),
                    "label": canonical_label,
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ],
                    "score": float(
                        score.detach().cpu().item()
                    ),
                }
            )

            per_query_count[
                canonical_label
            ] += 1

        return detections

    
    @staticmethod
    def _normalize_queries(
        queries: Iterable[str],
    ) -> list[str]:

        normalized = []

        for query in queries:

            if query is None:
                continue

            query = str(
                query
            ).strip().lower()

            if not query:
                continue

            if query not in normalized:
                normalized.append(
                    query
                )

        return normalized

    @staticmethod
    def _build_prompt(
        query: str,
    ) -> str:

        return (
            f"a photo of a {query}"
        )

    @staticmethod
    def _clean_label(
        label,
    ) -> str:

        if label is None:
            return ""

        return str(
            label
        ).strip().lower()

    @staticmethod
    def _match_query_label(
        detected_label: str,
        queries: list[str],
    ) -> str | None:

        detected_label = (
            detected_label
            .strip()
            .lower()
        )

        for query in queries:

            if detected_label == query:
                return query

            prompt = (
                f"a photo of a {query}"
            )

            if detected_label == prompt:
                return query

        return None