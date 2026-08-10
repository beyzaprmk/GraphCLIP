from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def download_model(
    repo_id: str,
    cache_dir: str | Path = "artifacts",
) -> Path:
   
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_name = repo_id.split("/")[-1]

    local_dir = cache_dir / model_name

    local_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("GraphCLIP model download")
    print(f"Repository : {repo_id}")
    print(f"Destination: {local_dir}")
    print("=" * 60)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
    )

    return local_dir