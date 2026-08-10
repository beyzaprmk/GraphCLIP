from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def upload_model(
    artifact_dir: str | Path,
    repo_id: str,
) -> None:
    
    artifact_dir = Path(artifact_dir)

    if not artifact_dir.exists():
        raise FileNotFoundError(
            f"Artifact directory not found: {artifact_dir}"
        )

    if not artifact_dir.is_dir():
        raise NotADirectoryError(
            f"Artifact path is not a directory: {artifact_dir}"
        )

   
    required_files = [
        "model.pt",
        "config.json",
        "metadata.json",
    ]

    missing_files = [
        filename
        for filename in required_files
        if not (artifact_dir / filename).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Invalid GraphCLIP artifact.\n"
            f"Directory: {artifact_dir}\n"
            f"Missing files: {', '.join(missing_files)}"
        )

    # ----------------------------------------------------------
    # Hugging Face API
    # ----------------------------------------------------------

    api = HfApi()

    print("=" * 60)
    print("GraphCLIP Hugging Face upload")
    print(f"Artifact : {artifact_dir}")
    print(f"Repository: {repo_id}")
    print("=" * 60)

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
    )

   
    api.upload_folder(
        folder_path=str(artifact_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload GraphCLIP model artifact",
    )

    print("=" * 60)
    print("GraphCLIP model uploaded successfully.")
    print(f"Repository: https://huggingface.co/{repo_id}")
    print("=" * 60)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Upload a GraphCLIP artifact to Hugging Face Hub."
    )

    parser.add_argument(
        "--artifact",
        default="artifacts/graphclip-base",
        help=(
            "Path to the GraphCLIP artifact directory. "
            "Default: artifacts/graphclip-base"
        ),
    )

    parser.add_argument(
        "--repo",
        required=True,
        help=(
            "Hugging Face repository ID. "
            "Example: username/graphclip-base"
        ),
    )

    args = parser.parse_args()

    upload_model(
        artifact_dir=args.artifact,
        repo_id=args.repo,
    )


if __name__ == "__main__":
    main()