from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_ARTIFACTS_DIR = (
    BASE_DIR / "artifacts"
)


def download_model(
    repo_id: str,
    output_dir: Path,
    force: bool = False,
) -> None:

    if output_dir.exists():

        if not force:
            raise FileExistsError(
                "Model artifact already exists:\n"
                f"{output_dir}\n\n"
                "Use --force to replace it."
            )

        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("Downloading GraphCLIP model")
    print(f"Repository : {repo_id}")
    print(f"Destination: {output_dir}")
    print("=" * 60)

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(output_dir),
    )

    print("=" * 60)
    print("GraphCLIP model downloaded successfully.")
    print(f"Artifact: {output_dir}")
    print("=" * 60)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Download a pretrained GraphCLIP model "
            "from Hugging Face Hub."
        )
    )

    parser.add_argument(
        "--repo",
        required=True,
        help=(
            "Hugging Face model repository ID.\n"
            "Example: username/graphclip-base"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Local artifact directory.\n"
            "Default: artifacts/<model-name>"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace an existing local artifact."
        ),
    )

    args = parser.parse_args()

    model_name = args.repo.split("/")[-1]

    if args.output is None:
        output_dir = (
            DEFAULT_ARTIFACTS_DIR
            / model_name
        )
    else:
        output_dir = args.output

    download_model(
        repo_id=args.repo,
        output_dir=output_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()