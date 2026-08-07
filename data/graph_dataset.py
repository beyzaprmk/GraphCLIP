from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from torch.utils.data import Dataset

from data.graph_converter import GraphConverter


class GraphDataset(Dataset):
    

    def __init__(

        self,

        image_ids: list[int],

        graph_dir: str,

        graph_converter: GraphConverter,

        images_dir: str,

        captions: dict[int, str],

        image_transform: Callable | None = None

    ):

        self.image_ids = image_ids

        self.graph_dir = Path(graph_dir)

        self.graph_converter = graph_converter

        self.images_dir = Path(images_dir)

        self.captions = captions

        self.image_transform = image_transform

    def __len__(self) -> int:

        return len(self.image_ids)

    def __getitem__(
        self,
        index: int
    ) -> dict:

        image_id = self.image_ids[index]

        graph = self._load_graph(
            image_id
        )

        caption = self._load_caption(
            image_id
        )

        return {

            "graph": graph,

            "text": caption,

            "image_id": image_id

        }
    def _load_image(

        self,

        image_id: int

    ):

        image_path = (

            self.images_dir /

            f"{image_id}.jpg"

        )

        image = Image.open(

            image_path

        ).convert("RGB")

        if self.image_transform is not None:

            image = self.image_transform(

                image

            )

        return image

    def _load_graph(

        self,

        image_id: int

    ):

        graph_path = (

            self.graph_dir /

            f"{image_id}.pt"

        )

        if not graph_path.exists():

            raise FileNotFoundError(

                f"SceneGraph bulunamadı: {graph_path}"

            )

        try:

            scene_graph = torch.load(

                graph_path,

                map_location="cpu",

                weights_only=False

            )

        except Exception as e:

            raise RuntimeError(

                f"SceneGraph yüklenemedi: {graph_path}"

            ) from e

        return self.graph_converter.convert(

            scene_graph

        )

    def _load_caption(

        self,

        image_id: int

    ) -> str:

        return self.captions.get(

            image_id,

            ""

        )