from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from relation.vocabulary import RelationVocabulary
from data.graph_converter import GraphConverter

from inference.loader import load_model
from inference.text_analyzer import TextAnalyzer
from inference.object_detector import OWLViTObjectDetector
from inference.vision_extractor import InferenceVisionExtractor
from inference.image_processor import ImageProcessor
from inference.predictor import GraphCLIPPredictor
from inference.similarity import Similarity


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)


def _get_synset_path() -> Path:

    synset_path = (
        PROJECT_ROOT
        / "relation"
        / "resources"
        / "relationship_synsets.json"
    )

    if not synset_path.exists():

        raise FileNotFoundError(
            "GraphCLIP relation synset resource "
            "was not found.\n"
            f"Expected: {synset_path}"
        )

    return synset_path.resolve()


class GraphCLIPInference:

    def __init__(
        self,
        model,
        device: str | None = None,
        vision_model_name: str = (
            "openai/clip-vit-base-patch32"
        ),
        owlvit_model_name: str = (
            "google/owlvit-base-patch32"
        ),
        detection_threshold: float = 0.10,
        max_detections_per_query: int = 3,
    ):

        self.model = model

        self.device = (
            torch.device(device)
            if device is not None
            else next(
                model.parameters()
            ).device
        )

        # RESOLVE VOCABULARY

        vocab_path = getattr(
            model,
            "_vocab_path",
            None,
        )

        if vocab_path is None:

            raise ValueError(
                "The loaded GraphCLIP model does not "
                "contain relation vocabulary information."
            )

        vocab_path = Path(
            vocab_path
        ).resolve()

        if not vocab_path.exists():

            raise FileNotFoundError(
                "GraphCLIP relation vocabulary not found.\n"
                f"Expected: {vocab_path}"
            )

        self.vocab_path = vocab_path

        # GRAPH CONVERTER

        relation_vocab = (
            RelationVocabulary.load(
                vocab_path
            )
        )

        graph_converter = GraphConverter(
            relation_vocab=relation_vocab
        )
        # PREDICTOR

        self.predictor = GraphCLIPPredictor(
            model=model,
            graph_converter=graph_converter,
            device=str(self.device),
        )

        # TEXT ANALYZER

        self.synset_path = (
            _get_synset_path()
        )

        self.text_analyzer = TextAnalyzer(
            vocab_path=vocab_path,
            synset_path=self.synset_path,
        )

        # OBJECT DETECTOR

        self.object_detector = (
            OWLViTObjectDetector(
                model_name=owlvit_model_name,
                device=str(self.device),
                threshold=detection_threshold,
                max_detections_per_query=(
                    max_detections_per_query
                ),
            )
        )

        # VISION FEATURE EXTRACTOR

        self.vision_extractor = (
            InferenceVisionExtractor(
                model_name=vision_model_name,
                device=str(self.device),
            )
        )

        # IMAGE PROCESSOR

        self.image_processor = ImageProcessor(
            detector=self.object_detector,
            feature_extractor=self.vision_extractor,
        )

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path,
        device: str | None = None,
        vision_model_name: str = (
            "openai/clip-vit-base-patch32"
        ),
        owlvit_model_name: str = (
            "google/owlvit-base-patch32"
        ),
        detection_threshold: float = 0.10,
        max_detections_per_query: int = 3,
    ):

        model = load_model(
            source=source,
            device=device,
        )

        return cls(
            model=model,
            device=device,
            vision_model_name=(
                vision_model_name
            ),
            owlvit_model_name=(
                owlvit_model_name
            ),
            detection_threshold=(
                detection_threshold
            ),
            max_detections_per_query=(
                max_detections_per_query
            ),
        )

    @torch.no_grad()
    def predict(
        self,
        image: str | Path | Image.Image,
        text: str,
        image_id: str = "inference",
    ) -> dict:

        image = self._load_image(
            image
        )

        analysis = (
            self.text_analyzer.analyze(
                text
            )
        )

        if not analysis.entities:

            raise ValueError(
                "Kullanıcı text'inden "
                "entity çıkarılamadı."
            )

        scene_graph = (
            self.image_processor.process(
                image=image,
                analysis=analysis,
                image_id=image_id,
            )
        )

        image_embedding = (
            self.predictor.encode_graph(
                scene_graph
            )
        )

        text_embedding = (
            self._encode_text(
                text
            )
        )

        similarity = Similarity.cosine(
            image_embedding=image_embedding,
            text_embedding=text_embedding,
        )

        return {
            "similarity": similarity,
            "image_embedding": image_embedding,
            "text_embedding": text_embedding,
            "scene_graph": scene_graph,
            "text_analysis": analysis,
        }

    @torch.no_grad()
    def _encode_text(
        self,
        text: str,
    ) -> torch.Tensor:

        if not text or not text.strip():

            raise ValueError(
                "Text boş olamaz."
            )

        tokens = self.model.tokenizer(
            [text],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        tokens = {
            key: value.to(
                self.device
            )
            for key, value in tokens.items()
        }

        text_outputs = (
            self.model.clip.text_model(
                **tokens
            )
        )

        text_embedding = (
            self.model.clip.text_projection(
                text_outputs.pooler_output
            )
        )

        text_embedding = (
            torch.nn.functional.normalize(
                text_embedding,
                dim=-1,
            )
        )

        return text_embedding

    @staticmethod
    def _load_image(
        image: str | Path | Image.Image,
    ) -> Image.Image:

        if isinstance(
            image,
            Image.Image,
        ):

            return image.convert(
                "RGB"
            )

        image_path = Path(
            image
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Görüntü bulunamadı: "
                f"{image_path}"
            )

        return Image.open(
            image_path
        ).convert(
            "RGB"
        )