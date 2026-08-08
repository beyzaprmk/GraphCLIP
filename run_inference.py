from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from relation.vocabulary import RelationVocabulary
from data.graph_converter import GraphConverter

from inference.factory import GraphCLIPFactory
from inference.loader import CheckpointLoader
from inference.inference import GraphCLIPInference


VOCAB_PATH = (
    BASE_DIR
    / "relation"
    / "resources"
    / "final_vocab.json"
)

SYNSET_PATH = (
    BASE_DIR
    / "relation"
    / "resources"
    / "relationship_synsets.json"
)

CHECKPOINT_PATH = (
    BASE_DIR
    / "checkpoints"
    / "best_model.pt"
)


def main():

    print("=" * 60)
    print("GraphCLIP Inference")
    print("=" * 60)

    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"Relation vocabulary bulunamadı:\n{VOCAB_PATH}"
        )

    if not SYNSET_PATH.exists():
        raise FileNotFoundError(
            f"Relation synset dosyası bulunamadı:\n{SYNSET_PATH}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı:\n{CHECKPOINT_PATH}"
        )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print("GraphCLIP modeli oluşturuluyor...")

    model = GraphCLIPFactory.create(
        vocab_path=VOCAB_PATH,
        model_name="openai/clip-vit-base-patch32",
    )

    # Checkpoint
    
    print("Checkpoint yükleniyor...")

    loader = CheckpointLoader(
        model=model
    )

    model = loader.load(
        CHECKPOINT_PATH
    )

    device = loader.device

    # Graph Converter

    relation_vocab = RelationVocabulary.load(
        VOCAB_PATH
    )

    graph_converter = GraphConverter(
        relation_vocab=relation_vocab
    )
    # Inference Pipeline

    pipeline = GraphCLIPInference(
        model=model,
        graph_converter=graph_converter,
        vocab_path=VOCAB_PATH,
        synset_path=SYNSET_PATH,
        device=str(device),
        vision_model_name="openai/clip-vit-base-patch32",
        owlvit_model_name="google/owlvit-base-patch32",
        detection_threshold=0.10,
        max_detections_per_query=3,
    )

    # User Input
    image_path = input(
        "\nGörüntü yolu: "
    ).strip()

    text = input(
        "Metin: "
    ).strip()

    if not image_path:
        raise ValueError(
            "Görüntü yolu boş olamaz."
        )

    if not text:
        raise ValueError(
            "Metin boş olamaz."
        )

    # Inference
    print("\nInference başlatılıyor...")
    print("-" * 60)

    result = pipeline.predict(
        image=image_path,
        text=text,
        image_id=Path(image_path).stem,
    )

    # Results
    similarity = result["similarity"]

    print("\n" + "=" * 60)
    print("INFERENCE SONUCU")
    print("=" * 60)

    print(
        f"Similarity: "
        f"{similarity.item():.4f}"
    )

    print("\nEntities:")

    for entity in result[
        "text_analysis"
    ].entities:

        print(
            f"  - {entity.text}"
        )

    print("\nRelations:")

    for relation in result[
        "text_analysis"
    ].relations:

        print(
            f"  - "
            f"{relation.subject} "
            f"-- {relation.canonical_relation} "
            f"--> {relation.object} "
            f"(id={relation.relation_id})"
        )

    print("\nScene Graph:")

    scene_graph = result[
        "scene_graph"
    ]

    print(
        f"  Nodes: {len(scene_graph.nodes)}"
    )

    print(
        f"  Edges: {len(scene_graph.edges)}"
    )

    for node in scene_graph.nodes:

        print(
            f"  Node {node.node_id}: "
            f"{node.label} "
            f"bbox={node.bbox}"
        )

    for edge in scene_graph.edges:

        print(
            f"  Edge: "
            f"{edge.source_id} "
            f"-- {edge.relation_label} "
            f"--> {edge.target_id} "
            f"(id={edge.relation_id})"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()