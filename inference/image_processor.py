from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
from PIL import Image

from core.entities import (
    BoundingBox,
    Edge,
    Node,
    SceneGraph,
)


@dataclass(frozen=True)
class DetectedObject:
    detector_id: int | str
    label: str
    bbox: BoundingBox


@dataclass(frozen=True)
class DetectedRelation:
    
    source_id: int | str
    target_id: int | str
    relation_label: str


class ImageProcessor:
    
    def __init__(
        self,
        detector: Any,
        feature_extractor: Any,
    ):
        self.detector = detector
        self.feature_extractor = feature_extractor

    def process(
        self,
        image: Image.Image,
        text_entities: Iterable[str],
        image_id: str = "inference",
    ) -> SceneGraph:

        if not isinstance(image, Image.Image):
            raise TypeError(
                "image bir PIL.Image.Image olmalıdır."
            )

        entities = self._normalize_entities(
            text_entities
        )

        if len(entities) == 0:
            raise ValueError(
                "En az bir text entity gereklidir."
            )

        detections = self._detect(
            image=image,
            entities=entities,
        )

        if len(detections) == 0:
            raise ValueError(
                "Görüntüde istenen nesneler bulunamadı."
            )

        nodes, id_mapping = self._build_nodes(
            image=image,
            detections=detections,
        )

        relations = self._get_relations(
            image=image,
            detections=detections,
            id_mapping=id_mapping,
            entities=entities,
        )

        edges = self._build_edges(
            relations=relations,
        )

        return SceneGraph(
            image_id=str(image_id),
            nodes=nodes,
            edges=edges,
        )

    def _detect(
        self,
        image: Image.Image,
        entities: list[str],
    ) -> list[DetectedObject]:

        if hasattr(self.detector, "detect"):
            result = self.detector.detect(
                image=image,
                queries=entities,
            )

        elif callable(self.detector):
            result = self.detector(
                image,
                entities,
            )

        else:
            raise TypeError(
                "Detector .detect(image, queries) metoduna "
                "veya callable bir arayüze sahip olmalıdır."
            )

        return self._parse_detections(
            result
        )

    def _parse_detections(
        self,
        result: Any,
    ) -> list[DetectedObject]:

        if result is None:
            return []

        detections = []

        for index, item in enumerate(result):

            if isinstance(item, DetectedObject):
                detections.append(item)
                continue

            if not isinstance(item, dict):
                raise TypeError(
                    "Detector çıktısındaki her nesne "
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
                    "Detector çıktısında 'label' veya 'name' "
                    "bulunamadı."
                )

            if bbox is None:
                raise ValueError(
                    "Detector çıktısında 'bbox' bulunamadı."
                )

            detections.append(
                DetectedObject(
                    detector_id=detector_id,
                    label=str(label),
                    bbox=self._parse_bbox(bbox),
                )
            )

        return detections

    @staticmethod
    def _parse_bbox(
        bbox: Any,
    ) -> BoundingBox:

        if isinstance(bbox, BoundingBox):
            return bbox

        if isinstance(bbox, dict):

            if {
                "x_min",
                "y_min",
                "x_max",
                "y_max",
            }.issubset(bbox):

                return BoundingBox(
                    x_min=float(bbox["x_min"]),
                    y_min=float(bbox["y_min"]),
                    x_max=float(bbox["x_max"]),
                    y_max=float(bbox["y_max"]),
                )

            if {
                "x",
                "y",
                "w",
                "h",
            }.issubset(bbox):

                x = float(bbox["x"])
                y = float(bbox["y"])
                w = float(bbox["w"])
                h = float(bbox["h"])

                return BoundingBox(
                    x_min=x,
                    y_min=y,
                    x_max=x + w,
                    y_max=y + h,
                )

        if isinstance(bbox, (list, tuple)):

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
        dict[int | str, int],
    ]:

        nodes = []
        id_mapping = {}

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
                    f"Node feature boyutu 512 olmalıdır. "
                    f"Alınan boyut: {feature.numel()}"
                )

            feature = feature.float()

            id_mapping[
                detection.detector_id
            ] = node_id

            nodes.append(
                Node(
                    node_id=node_id,
                    label=detection.label,
                    bbox=detection.bbox,
                    feature_tensor=feature,
                )
            )

        return nodes, id_mapping

    def _extract_feature(
        self,
        crop: Image.Image,
    ) -> torch.Tensor:

        if hasattr(
            self.feature_extractor,
            "extract",
        ):

            feature = self.feature_extractor.extract(
                crop
            )

        elif callable(
            self.feature_extractor
        ):

            feature = self.feature_extractor(
                crop
            )

        else:

            raise TypeError(
                "Feature extractor .extract(image) metoduna "
                "veya callable bir arayüze sahip olmalıdır."
            )

        if not isinstance(
            feature,
            torch.Tensor,
        ):

            feature = torch.as_tensor(
                feature
            )

        return feature.detach().cpu()

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
            (x1, y1, x2, y2)
        )

    def _get_relations(
        self,
        image: Image.Image,
        detections: list[DetectedObject],
        id_mapping: dict[int | str, int],
        entities: list[str],
    ) -> list[DetectedRelation]:

        if not hasattr(
            self.detector,
            "detect_relations",
        ):

            return []

        raw_relations = self.detector.detect_relations(
            image=image,
            objects=detections,
            entities=entities,
        )

        relations = []

        for item in raw_relations or []:

            if isinstance(
                item,
                DetectedRelation,
            ):

                relations.append(item)
                continue

            if not isinstance(
                item,
                dict,
            ):

                raise TypeError(
                    "Relation çıktısı dict veya "
                    "DetectedRelation olmalıdır."
                )

            source_id = item.get(
                "source_id",
                item.get(
                    "subject_id",
                ),
            )

            target_id = item.get(
                "target_id",
                item.get(
                    "object_id",
                ),
            )

            relation_label = item.get(
                "relation_label",
                item.get(
                    "predicate",
                ),
            )

            if (
                source_id is None
                or target_id is None
                or relation_label is None
            ):

                continue

            if source_id not in id_mapping:
                continue

            if target_id not in id_mapping:
                continue

            relations.append(
                DetectedRelation(
                    source_id=id_mapping[source_id],
                    target_id=id_mapping[target_id],
                    relation_label=str(
                        relation_label
                    ),
                )
            )

        return relations

    @staticmethod
    def _build_edges(
        relations: list[DetectedRelation],
    ) -> list[Edge]:

        edges = []

        for relation in relations:

            edges.append(
                Edge(
                    source_id=int(
                        relation.source_id
                    ),
                    target_id=int(
                        relation.target_id
                    ),
                    relation_label=(
                        relation.relation_label
                        .strip()
                        .lower()
                    ),
                    relation_id=None,
                )
            )

        return edges

    @staticmethod
    def _normalize_entities(
        entities: Iterable[str],
    ) -> list[str]:

        normalized = []

        for entity in entities:

            if entity is None:
                continue

            entity = str(entity).strip().lower()

            if not entity:
                continue

            if entity not in normalized:
                normalized.append(entity)

        return normalized