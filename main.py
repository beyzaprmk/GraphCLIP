from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from relation.build import build_relation_vocabulary
from relation.vocabulary import RelationVocabulary

from data.graph_converter import GraphConverter
from data.graph_dataset import GraphDataset

from model.graph_encoder import (
    RelationEmbedding,
    GraphBackbone,
    GraphEncoder,
)

from model.fusion import FusionHead
from model.graph_clip import GraphCLIP

from pipline.trainer import Trainer

BATCH_SIZE = 8
EPOCHS = 20

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

VALIDATION_SPLIT = 0.20
RANDOM_STATE = 42

NUM_WORKERS = 4

RELATION_EMBEDDING_DIM = 64
GRAPH_NODE_DIM = 512
GRAPH_HIDDEN_DIM = 512

DROPOUT = 0.1


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "dataset"

IMAGES_DIR = DATA_DIR / "images"

GRAPH_DIR = DATA_DIR / "processed_data"

CAPTION_FILE = DATA_DIR / "captions.json"

# Canonical relation vocabulary
RELATION_VOCAB = (
    BASE_DIR
    / "relation"
    / "resources"
    / "final_vocab.json"
)

# Portable artifact vocabulary
ARTIFACT_DIR = (
    BASE_DIR
    / "artifacts"
    / "graphclip-base"
)

ARTIFACT_VOCAB = (
    ARTIFACT_DIR
    / "final_vocab.json"
)

CHECKPOINT_DIR = (
    BASE_DIR
    / "checkpoints"
)

CAPTION_SCRIPT = (
    BASE_DIR
    / "create_caption.py"
)


def prepare_captions() -> None:

    if CAPTION_FILE.exists():

        print(
            f"Captions found: {CAPTION_FILE}"
        )

        return

    if not CAPTION_SCRIPT.exists():

        raise FileNotFoundError(
            "\n"
            "Training captions were not found.\n\n"
            f"Expected:\n"
            f"  {CAPTION_FILE}\n\n"
            f"Caption generation script was also not found:\n"
            f"  {CAPTION_SCRIPT}\n"
        )

    print("=" * 60)
    print("captions.json not found.")
    print("Running create_caption.py...")
    print("=" * 60)

    result = subprocess.run(
        [
            sys.executable,
            str(CAPTION_SCRIPT),
        ],
        cwd=str(BASE_DIR),
    )

    if result.returncode != 0:

        raise RuntimeError(
            "create_caption.py failed.\n"
            "Training cannot continue."
        )

    if not CAPTION_FILE.exists():

        raise FileNotFoundError(
            "create_caption.py completed, but "
            "captions.json was not created.\n"
            f"Expected: {CAPTION_FILE}"
        )

    print(
        f"Captions created: {CAPTION_FILE}"
    )


def prepare_relation_vocabulary() -> Path:
    
    if ARTIFACT_VOCAB.exists():

        print(
            f"Using artifact vocabulary:\n"
            f"  {ARTIFACT_VOCAB}"
        )

        # Keep the repository vocabulary synchronized.
        RELATION_VOCAB.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not RELATION_VOCAB.exists():

            shutil.copy2(
                ARTIFACT_VOCAB,
                RELATION_VOCAB,
            )

        return ARTIFACT_VOCAB

    if RELATION_VOCAB.exists():

        print(
            f"Using relation vocabulary:\n"
            f"  {RELATION_VOCAB}"
        )

        # Make sure the artifact contains the
        # same vocabulary.

        ARTIFACT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            RELATION_VOCAB,
            ARTIFACT_VOCAB,
        )

        return RELATION_VOCAB

    print("=" * 60)
    print("Relation vocabulary not found.")
    print("Building relation vocabulary...")
    print("=" * 60)

    RELATION_VOCAB.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    build_relation_vocabulary()

    if not RELATION_VOCAB.exists():

        raise FileNotFoundError(
            "\n"
            "Relation vocabulary generation completed, "
            "but final_vocab.json was not found.\n\n"
            f"Expected:\n"
            f"  {RELATION_VOCAB}\n"
        )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        RELATION_VOCAB,
        ARTIFACT_VOCAB,
    )

    print(
        f"Relation vocabulary created:\n"
        f"  {RELATION_VOCAB}"
    )

    print(
        f"Artifact vocabulary created:\n"
        f"  {ARTIFACT_VOCAB}"
    )

    return RELATION_VOCAB


def validate_dataset() -> None:

    missing = []

    if not DATA_DIR.exists():
        missing.append(str(DATA_DIR))

    if not IMAGES_DIR.exists():
        missing.append(str(IMAGES_DIR))

    if not GRAPH_DIR.exists():
        missing.append(str(GRAPH_DIR))

    if missing:

        raise FileNotFoundError(
            "\n"
            "GraphCLIP training data is not ready.\n\n"
            "Missing required directories:\n"
            + "\n".join(
                f"  - {path}"
                for path in missing
            )
            + "\n\n"
            "Prepare the Visual Genome dataset "
            "before starting training."
        )

    graph_files = list(
        GRAPH_DIR.glob("*.pt")
    )

    if not graph_files:

        raise FileNotFoundError(
            "\n"
            "No processed SceneGraph files were found.\n\n"
            f"Expected directory:\n"
            f"  {GRAPH_DIR}\n\n"
            "Prepare the processed dataset before training."
        )


def main() -> None:

    print("=" * 60)
    print("GraphCLIP Training")
    print("=" * 60)

   
    validate_dataset()

    prepare_captions()

    
    vocabulary_path = (
        prepare_relation_vocabulary()
    )


    relation_vocab = (
        RelationVocabulary.load(
            vocabulary_path
        )
    )

    num_relations = len(
        relation_vocab.relation_to_id
    )

    print(
        f"Relations: {num_relations:,}"
    )

    image_ids = sorted(
        int(path.stem)
        for path in GRAPH_DIR.glob("*.pt")
        if path.stem.isdigit()
    )

    if not image_ids:

        raise ValueError(
            "No valid SceneGraph image IDs were found."
        )

    
    print("=" * 60)
    print("Validating SceneGraphs...")
    print("=" * 60)

    valid_image_ids = []

    empty_graphs = 0
    invalid_graphs = 0

    for image_id in tqdm(
        image_ids,
        desc="Validating",
    ):

        graph_path = (
            GRAPH_DIR
            / f"{image_id}.pt"
        )

        try:

            scene_graph = torch.load(
                graph_path,
                map_location="cpu",
                weights_only=False,
            )

        except Exception:

            invalid_graphs += 1
            continue

        if not hasattr(
            scene_graph,
            "nodes",
        ):

            invalid_graphs += 1
            continue

        if len(
            scene_graph.nodes
        ) == 0:

            empty_graphs += 1
            continue

        valid_image_ids.append(
            image_id
        )

    image_ids = valid_image_ids

    if len(image_ids) < 2:

        raise ValueError(
            "Not enough valid SceneGraphs for training."
        )

    print(
        f"Valid SceneGraphs : {len(image_ids):,}"
    )

    print(
        f"Empty SceneGraphs : {empty_graphs:,}"
    )

    print(
        f"Invalid SceneGraphs: {invalid_graphs:,}"
    )

    
    with CAPTION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        raw_captions = json.load(
            file
        )

    captions = {
        int(key): str(value)
        for key, value in raw_captions.items()
    }

    if not captions:

        raise ValueError(
            "captions.json is empty. "
            "Text supervision is required for training."
        )

    
    image_ids = [
        image_id
        for image_id in image_ids
        if image_id in captions
    ]

    if len(image_ids) < 2:

        raise ValueError(
            "Not enough SceneGraphs have "
            "corresponding captions."
        )

    print(
        f"Usable image-caption pairs: "
        f"{len(image_ids):,}"
    )

   
    train_ids, val_ids = train_test_split(
        image_ids,
        test_size=VALIDATION_SPLIT,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("=" * 60)
    print("Dataset split")
    print("=" * 60)

    print(
        f"Train      : {len(train_ids):,}"
    )

    print(
        f"Validation : {len(val_ids):,}"
    )

    
    graph_converter = GraphConverter(
        relation_vocab=relation_vocab
    )

    # ------------------------------------------------------
    # Training dataset
    # ------------------------------------------------------

    train_dataset = GraphDataset(
        image_ids=train_ids,
        graph_dir=str(GRAPH_DIR),
        graph_converter=graph_converter,
        images_dir=str(IMAGES_DIR),
        captions=captions,
        image_transform=None,
    )

   
    val_dataset = GraphDataset(
        image_ids=val_ids,
        graph_dir=str(GRAPH_DIR),
        graph_converter=graph_converter,
        images_dir=str(IMAGES_DIR),
        captions=captions,
        image_transform=None,
    )

   
    pin_memory = torch.cuda.is_available()

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": pin_memory,
    }

    if NUM_WORKERS > 0:

        loader_kwargs[
            "persistent_workers"
        ] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

  
    relation_embedding = RelationEmbedding(
        num_relations=num_relations,
        embedding_dim=RELATION_EMBEDDING_DIM,
    )

    graph_backbone = GraphBackbone(
        node_dim=GRAPH_NODE_DIM,
        edge_dim=RELATION_EMBEDDING_DIM,
        hidden_dim=GRAPH_HIDDEN_DIM,
        dropout=DROPOUT,
    )

    graph_encoder = GraphEncoder(
        relation_embedding=relation_embedding,
        backbone=graph_backbone,
        hidden_dim=GRAPH_HIDDEN_DIM,
    )

    fusion_head = FusionHead(
        embedding_dim=GRAPH_HIDDEN_DIM,
        dropout=DROPOUT,
    )

    model = GraphCLIP(
        graph_encoder=graph_encoder,
        fusion_head=fusion_head,
    )

    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        epochs=EPOCHS,
        checkpoint_dir=str(
            CHECKPOINT_DIR
        ),
    )

    
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.fit()

    print("=" * 60)
    print("Training finished.")
    print(
        f"Checkpoints: {CHECKPOINT_DIR}"
    )
    print("=" * 60)

if __name__ == "__main__":
    main()