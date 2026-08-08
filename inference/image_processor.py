from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from PIL import Image

from core.entities import (
    BoundingBox,
    Edge,
    Node,
    SceneGraph,
)

from inference.text_analyzer import (
    TextAnalysis,
    TextRelation,
)

@dataclass(frozen=True)
class DetectedObject:
    detector_id: int | str
    label: str
    bbox: BoundingBox


class ImageProcessor:
   
    def __init__(
        self,
        detector,
        feature_extractor,
    ):
        self.detector = detector
        self.feature_extractor = feature_extractor

    def process(
        self,
        image: Image.Image,
        analysis: TextAnalysis,
        image_id: str = "inference",
    ) -> SceneGraph:

        if not isinstance(
            image,
            Image.Image,
        ):
            raise TypeError(
                "image bir PIL.Image.Image olmalıdır."
            )

        if not isinstance(
            analysis,
            TextAnalysis,
        ):
            raise TypeError(
                "analysis bir TextAnalysis nesnesi olmalıdır."
            )

        entities = self._normalize_entities(
            analysis
        )

        if len(entities) == 0:
            raise ValueError(
                "Text analizinden hiçbir entity çıkarılamadı."
            )

        detections = self._detect(
            image=image,
            entities=entities,
        )

        if len(detections) == 0:
            raise ValueError(
                "Görüntüde text ile ilişkili hiçbir nesne bulunamadı."
            )

        nodes, entity_to_node = self._build_nodes(
            image=image,
            detections=detections,
        )

        edges = self._build_edges(
            relations=analysis.relations,
            entity_to_node=entity_to_node,
        )

        return SceneGraph(
            image_id=str(image_id),
            nodes=nodes,
            edges=edges,
        )

    # OBJECT DETECTION
    
    def _detect(
        self,
        image: Image.Image,
        entities: list[str],
    ) -> list[DetectedObject]:

        result = self.detector.detect(
            image=image,
            queries=entities,
        )

        if result is None:
            return []

        detections = []

        for index, item in enumerate(result):

            if isinstance(
                item,
                DetectedObject,
            ):
                detections.append(item)
                continue

            if not isinstance(
                item,
                dict,
            ):
                raise TypeError(
                    "Detector çıktısındaki nesneler "
                    "dict veya DetectedObject olmalıdır."
                )

            detector_id = item.get(
                "id",
                item.get(
                    "object_id",
                    index,
                ),
            )

            label = item.get(
                "label",
                item.get(
                    "name",
                ),
            )

            bbox = item.get(
                "bbox",
            )

            if label is None:
                raise ValueError(
                    "Detector çıktısında label/name bulunamadı."
                )

            if bbox is None:
                raise ValueError(
                    "Detector çıktısında bbox bulunamadı."
                )

            detections.append(
                DetectedObject(
                    detector_id=detector_id,
                    label=str(label).strip().lower(),
                    bbox=self._parse_bbox(
                        bbox
                    ),
                )
            )

        return detections

    @staticmethod
    def _parse_bbox(
        bbox,
    ) -> BoundingBox:

        if isinstance(
            bbox,
            BoundingBox,
        ):
            return bbox

        if isinstance(
            bbox,
            dict,
        ):

            if {
                "x_min",
                "y_min",
                "x_max",
                "y_max",
            }.issubset(bbox):

                return BoundingBox(
                    x_min=float(
                        bbox["x_min"]
                    ),
                    y_min=float(
                        bbox["y_min"]
                    ),
                    x_max=float(
                        bbox["x_max"]
                    ),
                    y_max=float(
                        bbox["y_max"]
                    ),
                )

            if {
                "x",
                "y",
                "w",
                "h",
            }.issubset(bbox):

                x = float(
                    bbox["x"]
                )

                y = float(
                    bbox["y"]
                )

                w = float(
                    bbox["w"]
                )

                h = float(
                    bbox["h"]
                )

                return BoundingBox(
                    x_min=x,
                    y_min=y,
                    x_max=x + w,
                    y_max=y + h,
                )

        if isinstance(
            bbox,
            (list, tuple),
        ):

            if len(bbox) != 4:
                raise ValueError(
                    "bbox dört değer içermelidir."
                )

            x1, y1, x2, y2 = bbox

            return BoundingBox(
                x_min=float(x1),
                y_min=float(y1),
                x_max=float(x2),
                y_max=float(y2),
            )

        raise TypeError(
            "Desteklenmeyen bbox formatı."
        )

    def _build_nodes(
        self,
        image: Image.Image,
        detections: list[DetectedObject],
    ) -> tuple[
        list[Node],
        dict[str, list[int]],
    ]:

        nodes = []

        entity_to_node: dict[
            str,
            list[int],
        ] = {}

        for node_id, detection in enumerate(
            detections
        ):

            crop = self._crop(
                image=image,
                bbox=detection.bbox,
            )

            feature = self._extract_feature(
                crop
            )

            if feature.ndim != 1:
                feature = feature.flatten()

            if feature.numel() != 512:
                raise ValueError(
                    "Node feature boyutu 512 olmalıdır. "
                    f"Alınan boyut: {feature.numel()}"
                )

            feature = feature.float()

            label = (
                detection.label
                .strip()
                .lower()
            )

            nodes.append(
                Node(
                    node_id=node_id,
                    label=label,
                    bbox=detection.bbox,
                    feature_tensor=feature,
                )
            )

            entity_to_node.setdefault(
                label,
                [],
            ).append(
                node_id
            )

        return nodes, entity_to_node

    # FEATURE EXTRACTION

    def _extract_feature(
        self,
        crop: Image.Image,
    ) -> torch.Tensor:

        if hasattr(
            self.feature_extractor,
            "extract",
        ):

            feature = (
                self.feature_extractor.extract(
                    crop
                )
            )

        elif callable(
            self.feature_extractor
        ):

            feature = self.feature_extractor(
                crop
            )

        else:

            raise TypeError(
                "Feature extractor .extract(image) "
                "metoduna veya callable bir arayüze "
                "sahip olmalıdır."
            )

        if not isinstance(
            feature,
            torch.Tensor,
        ):

            feature = torch.as_tensor(
                feature
            )

        return feature.detach().cpu()

    # CROPPING

    @staticmethod
    def _crop(
        image: Image.Image,
        bbox: BoundingBox,
    ) -> Image.Image:

        width, height = image.size

        x1 = max(
            0,
            int(bbox.x_min),
        )

        y1 = max(
            0,
            int(bbox.y_min),
        )

        x2 = min(
            width,
            int(bbox.x_max),
        )

        y2 = min(
            height,
            int(bbox.y_max),
        )

        if x2 <= x1 or y2 <= y1:

            x2 = min(
                x1 + 1,
                width,
            )

            y2 = min(
                y1 + 1,
                height,
            )

        return image.crop(
            (
                x1,
                y1,
                x2,
                y2,
            )
        )

    # EDGE CONSTRUCTION

    def _build_edges(
        self,
        relations: Iterable[TextRelation],
        entity_to_node: dict[str, list[int]],
    ) -> list[Edge]:

        edges = []

        for relation in relations:

            subject = self._normalize_entity(
                relation.subject
            )

            object_label = self._normalize_entity(
                relation.object
            )

            subject_nodes = (
                entity_to_node.get(
                    subject,
                    [],
                )
            )

            object_nodes = (
                entity_to_node.get(
                    object_label,
                    [],
                )
            )

            if not subject_nodes:
                continue

            if not object_nodes:
                continue

            for source_id in subject_nodes:

                for target_id in object_nodes:

                    if source_id == target_id:
                        continue

                    edges.append(
                        Edge(
                            source_id=source_id,
                            target_id=target_id,
                            relation_label=(
                                relation.canonical_relation
                                .strip()
                                .lower()
                            ),
                            relation_id=(
                                relation.relation_id
                            ),
                            spatial_features=None,
                        )
                    )

        return edges

    # ENTITY MATCHING

    @staticmethod
    def _normalize_entities(
        analysis: TextAnalysis,
    ) -> list[str]:

        entities = []

        for entity in analysis.entities:

            value = (
                entity.text
                .strip()
                .lower()
            )

            if not value:
                continue

            if value not in entities:
                entities.append(
                    value
                )

        return entities

    @staticmethod
    def _normalize_entity(
        value: str,
    ) -> str:

        return " ".join(
            value.strip()
            .lower()
            .split()
        )