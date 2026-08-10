# GraphCLIP Artifacts

This directory is used for local GraphCLIP model artifacts.

Artifacts are portable model packages intended for:

- loading pretrained GraphCLIP models;
- sharing trained models;
- exporting models before uploading them to Hugging Face Hub;
- storing locally downloaded pretrained models.

## Artifact Structure

A GraphCLIP artifact contains the files required to reconstruct a trained
GraphCLIP model:

```text
graphclip-base/
├── model.pt
├── config.json
└── metadata.json