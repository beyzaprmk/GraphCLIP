from pathlib import Path

from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from data.vg_parser import VisualGenomeParser
from data.graph_converter import GraphConverter
from data.graph_dataset import GraphDataset

from model.relation_vocab import RelationVocabulary
from model.graph_encoder import GraphEncoder
from model.fusion import FusionHead
from model.graph_clip import GraphCLIP

from pipline.trainer import Trainer


# PATHS

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "visual_genome"

OBJECTS_JSON = DATA_DIR / "objects.json"

RELATIONSHIPS_JSON = DATA_DIR / "relationships.json"

IMAGES_DIR = DATA_DIR / "images"

FEATURES_DIR = DATA_DIR / "features"

RELATION_VOCAB = (
    BASE_DIR
    / "resources"
    / "final_vocab.json"
)

CHECKPOINT_DIR = (
    BASE_DIR
    / "checkpoints"
)


# PARSER

parser = VisualGenomeParser(

    objects_json_path=str(OBJECTS_JSON),

    relationships_json_path=str(
        RELATIONSHIPS_JSON
    )

)

# RELATION VOCAB

relation_vocab = RelationVocabulary(
    str(RELATION_VOCAB)
)

# GRAPH CONVERTER

graph_converter = GraphConverter(

    feature_dir=str(FEATURES_DIR),

    relation_vocab=relation_vocab

)


# IMAGE IDS

image_ids = sorted(

    parser.objects_dict.keys(),

    key=int

)


# TRAIN / VALIDATION SPLIT

train_ids, val_ids = train_test_split(

    image_ids,

    test_size=0.20,

    random_state=42,

    shuffle=True

)


# CAPTIONS


captions = {}


# DATASETS

train_dataset = GraphDataset(

    image_ids=train_ids,

    parser=parser,

    graph_converter=graph_converter,

    images_dir=str(IMAGES_DIR),

    captions=captions,

    image_transform=None

)

val_dataset = GraphDataset(

    image_ids=val_ids,

    parser=parser,

    graph_converter=graph_converter,

    images_dir=str(IMAGES_DIR),

    captions=captions,

    image_transform=None

)


# DATALOADERS

train_loader = DataLoader(

    train_dataset,

    batch_size=32,

    shuffle=True,

    num_workers=4,

    pin_memory=True

)

val_loader = DataLoader(

    val_dataset,

    batch_size=32,

    shuffle=False,

    num_workers=4,

    pin_memory=True

)


# GRAPH ENCODER

graph_encoder = GraphEncoder(

    num_relations=len(relation_vocab),

    node_dim=512,

    relation_dim=64,

    hidden_dim=512

)


# FUSION

fusion_head = FusionHead(

    embedding_dim=512,

    hidden_dim=512

)


# MODEL

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

    epochs=30,

    checkpoint_dir=str(CHECKPOINT_DIR)

)



if __name__ == "__main__":

    trainer.fit()