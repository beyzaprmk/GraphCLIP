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


class GraphCLIPInference:

    def __init__(
        self,
        model,
        graph_converter,
        vocab_path: str | Path,
        synset_path: str | Path,
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

        self.predictor = GraphCLIPPredictor(
            model=model,
            graph_converter=graph_converter,
            device=device,
        )

        self.device = self.predictor.device

       
        self.text_analyzer = TextAnalyzer(
            vocab_path=vocab_path,
            synset_path=synset_path,
        )


        self.object_detector = OWLViTObjectDetector(
            model_name=owlvit_model_name,
            device=str(self.device),
            threshold=detection_threshold,
            max_detections_per_query=(
                max_detections_per_query
            ),
        )

       
        self.vision_extractor = InferenceVisionExtractor(
            model_name=vision_model_name,
            device=str(self.device),
        )

        
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

       
        source_path = Path(
            source
        )

       
        if source_path.exists():

            if source_path.is_dir():

                vocab_path = (
                    source_path
                    / "final_vocab.json"
                )

            else:

               
                vocab_path = (
                    PROJECT_ROOT
                    / "artifacts"
                    / "graphclip-base"
                    / "final_vocab.json"
                )

        
        else:

            from model.hub import download_model

            model_dir = download_model(
                repo_id=str(source),
                cache_dir="artifacts",
            )

            vocab_path = (
                Path(model_dir)
                / "final_vocab.json"
            )

        vocab_path = Path(
            vocab_path
        ).resolve()

        
        if not vocab_path.exists():

            raise FileNotFoundError(
                "GraphCLIP relation vocabulary not found.\n"
                f"Expected: {vocab_path}\n\n"
                "The inference pipeline requires "
                "final_vocab.json."
            )

       
        synset_path = (
            PROJECT_ROOT
            / "relation"
            / "resources"
            / "relationship_synsets.json"
        )

        if not synset_path.exists():

            raise FileNotFoundError(
                "Relation synset file not found.\n"
                f"Expected: {synset_path}\n\n"
                "The inference text analyzer requires "
                "relationship_synsets.json."
            )

       
        relation_vocab = (
            RelationVocabulary.load(
                vocab_path
            )
        )

        
        graph_converter = GraphConverter(
            relation_vocab=relation_vocab
        )

        
        return cls(
            model=model,
            graph_converter=graph_converter,
            vocab_path=vocab_path,
            synset_path=synset_path,
            device=device,
            vision_model_name=vision_model_name,
            owlvit_model_name=owlvit_model_name,
            detection_threshold=detection_threshold,
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

       
        analysis = self.text_analyzer.analyze(
            text
        )

        if not analysis.entities:

            raise ValueError(
                "Kullanıcı text'inden entity çıkarılamadı."
            )

       
        scene_graph = self.image_processor.process(
            image=image,
            analysis=analysis,
            image_id=image_id,
        )

      
        image_embedding = (
            self.predictor.encode_graph(
                scene_graph
            )
        )

       
        text_embedding = self._encode_text(
            text
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
            key: value.to(self.device)
            for key, value in tokens.items()
        }

        text_outputs = self.model.clip.text_model(
            **tokens
        )

        text_embedding = (
            self.model.clip.text_projection(
                text_outputs.pooler_output
            )
        )

        text_embedding = torch.nn.functional.normalize(
            text_embedding,
            dim=-1,
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
                f"Görüntü bulunamadı: {image_path}"
            )

        return Image.open(
            image_path
        ).convert(
            "RGB"
        )