from pathlib import Path

import torch
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




BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "dataset"

IMAGES_DIR = DATA_DIR / "images"

GRAPH_DIR = DATA_DIR / "processed_data"

RELATION_VOCAB = (
    BASE_DIR
    / "relation"
    / "resources"
    / "final_vocab.json"
)

CHECKPOINT_DIR = (
    BASE_DIR
    / "checkpoints"
)


def main():


    if not RELATION_VOCAB.exists():

        print("=" * 60)
        print("Relation vocabulary bulunamadı.")
        print("Vocabulary oluşturuluyor...")
        print("=" * 60)

        build_relation_vocabulary()

    relation_vocab = RelationVocabulary.load(
        RELATION_VOCAB
    )


    image_ids = sorted(

        int(path.stem)

        for path in GRAPH_DIR.glob("*.pt")

    )
    print("=" * 60)
    print("SceneGraph dosyaları doğrulanıyor...")
    print("=" * 60)

    valid_image_ids = []

    empty_graphs = 0

    for image_id in image_ids:

        graph_path = GRAPH_DIR / f"{image_id}.pt"

        try:

            scene_graph = torch.load(

                graph_path,

                map_location="cpu",

                weights_only=False

            )

        except Exception:

            continue

        if len(scene_graph.nodes) == 0:

            empty_graphs += 1

            continue

        valid_image_ids.append(image_id)

    image_ids = valid_image_ids

    print(f"Kullanılabilir SceneGraph : {len(image_ids):,}")
    print(f"Boş SceneGraph          : {empty_graphs:,}")
    print("=" * 60)


    graph_converter = GraphConverter(
        relation_vocab=relation_vocab
    )

    

    train_ids, val_ids = train_test_split(

        image_ids,

        test_size=0.20,

        shuffle=True,

        random_state=42

    )

    captions = {}

    train_dataset = GraphDataset(

        image_ids=train_ids,

        graph_dir=str(GRAPH_DIR),

        graph_converter=graph_converter,

        images_dir=str(IMAGES_DIR),

        captions=captions,

        image_transform=None

    )

    val_dataset = GraphDataset(

        image_ids=val_ids,

        graph_dir=str(GRAPH_DIR),

        graph_converter=graph_converter,

        images_dir=str(IMAGES_DIR),

        captions=captions,

        image_transform=None

    )

   
    PIN_MEMORY = torch.cuda.is_available()

    train_loader = DataLoader(

        train_dataset,

        batch_size=32,

        shuffle=True,

        num_workers=8,

        pin_memory=PIN_MEMORY,

        persistent_workers=True

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=32,

        shuffle=False,

        num_workers=8,

        pin_memory=PIN_MEMORY,

        persistent_workers=True

    )

   
    relation_embedding = RelationEmbedding(

        num_relations=len(
            relation_vocab.relation_to_id
        ),

        embedding_dim=64

    )

    graph_backbone = GraphBackbone(

        node_dim=512,

        edge_dim=64,

        hidden_dim=512

    )

    graph_encoder = GraphEncoder(

        relation_embedding=relation_embedding,

        backbone=graph_backbone

    )

    fusion_head = FusionHead()

    model = GraphCLIP(

        graph_encoder=graph_encoder,

        fusion_head=fusion_head

    )

    
    trainer = Trainer(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        learning_rate=1e-4,

        weight_decay=1e-4,

        epochs=1,

        checkpoint_dir=str(CHECKPOINT_DIR)

    )

    trainer.fit()


if __name__ == "__main__":

    main()