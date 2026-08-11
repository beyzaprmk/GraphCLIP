from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

DEFAULT_ARTIFACTS_DIR = (
    PROJECT_ROOT / "artifacts"
)


def download_model(
    repo_id: str,
    output_dir: str | Path | None = None,
    force: bool = False,
) -> Path:

    repo_name = repo_id.split("/")[-1]

    if output_dir is None:

        output_dir = (
            DEFAULT_ARTIFACTS_DIR
            / repo_name
        )

    output_dir = Path(
        output_dir
    ).resolve()

    if output_dir.exists():

        if not force:

            return output_dir

        shutil.rmtree(
            output_dir
        )

    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(output_dir),
    )

    return output_dir