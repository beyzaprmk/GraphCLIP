<div align="center">

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-3C2179?logo=pyg&logoColor=white)](https://pyg.org/)
[![Transformers](https://img.shields.io/badge/Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![CLIP](https://img.shields.io/badge/CLIP-412991?logo=openai&logoColor=white)](https://openai.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Visual Genome](https://img.shields.io/badge/Visual%20Genome-Dataset-blue)](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)

</div>

# GraphCLIP

GraphCLIP is a graph-enhanced vision-language model for semantic and relational image-text queries. The project includes both model training and inference pipelines and supports local checkpoints, portable artifacts, and Hugging Face pretrained models.

## Model Architecture

```text
Image
  │
  ▼
OWL-ViT
  │
  ▼
Object Detection
  │
  ▼
CLIP Vision Encoder
  │
  ▼
Scene Graph
  │
  ▼
Relation Embedding
  │
  ▼
Graph TransformerConv
  │
  ▼
Attentional Aggregation
  │
  ▼
Fusion Head
  │
  ▼
GraphCLIP Embedding
  │
  ├──────────────► Cosine Similarity ◄──────────────┐
  │                                                  │
Text ───────────────► CLIP Text Encoder ────────────┘
```

GraphCLIP combines CLIP, OWL-ViT, PyTorch Geometric, and a Graph Neural Network to represent objects and their relationships as a scene graph.

## Dataset

The model was trained using the [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html) dataset, including its images, object annotations, relationships, and region descriptions.

## Installation

Clone the repository:

```bash
git clone https://github.com/beyzaprmk/GraphCLIP.git
cd GraphCLIP
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip3 install -r requirements.txt
```

## Model Download & Upload

Download a pretrained model:

```bash
python3 scripts/export_model.py --repo prmkkbb/graphclip-base
```

The model is downloaded into:

```text
artifacts/graphclip-base/
```

Upload a local model:

```bash
python3 scripts/upload_model.py --repo YOUR_USERNAME/graphclip-base
```

## Using a Pretrained Model

A pretrained model can be loaded directly from Hugging Face:

```python
from inference.loader import load_model

model = load_model(
    "prmkkbb/graphclip-base"
)
```

It can also be loaded from a local artifact:

```python
model = load_model(
    "artifacts/graphclip-base"
)
```

## Using a Training Checkpoint

A model produced by the training pipeline can also be loaded directly:

```python
from inference.loader import load_model

model = load_model(
    "checkpoints/best_model.pt"
)
```

This allows the training output to be used directly for inference without exporting it first.

## Inference

```python
from inference.loader import load_model
from inference.inference import GraphCLIPInference

model = load_model(
    "prmkkbb/graphclip-base"
)

engine = GraphCLIPInference(
    model=model,
    graph_converter=graph_converter,
    vocab_path="artifacts/graphclip-base/final_vocab.json",
    synset_path="path/to/synset.json",
)

result = engine.predict(
    image="path/to/image.jpg",
    text="a flower next to a vase",
)

print(result["similarity"])
```

Inference does not require the original training dataset.

## Training

GraphCLIP can also be trained from scratch using the included training pipeline.
```bash
python3 main.py
```
The training process creates checkpoints under:
checkpoints/
├── best_model.pt
└── epoch_*.pt

Training uses a CLIP-style bidirectional contrastive loss with AdamW and cosine annealing.

## Model Artifacts

A portable model artifact contains:

```text
artifacts/graphclip-base/
├── model.pt
├── config.json
├── metadata.json
└── final_vocab.json
```

## Relational Queries

GraphCLIP is designed for queries involving multiple objects and their relationships, such as:

```text
"flower next to vase"
"person holding a cup"
"car near a tree"
"person beside a bicycle"
```

The inference pipeline extracts query entities, detects relevant objects with OWL-ViT, constructs a scene graph, encodes it with the Graph Neural Network, and compares it with the CLIP text embedding.