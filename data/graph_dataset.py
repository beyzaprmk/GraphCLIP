from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

from torch_geometric.data import Data

from core.entities import SceneGraph
from data.vg_parser import VisualGenomeParser
from data.graph_converter import GraphConverter


class GraphDataset(Dataset):
   

    def __init__(
        self,
        image_ids: list[int],
        parser: VisualGenomeParser,
        graph_converter: GraphConverter,
        images_dir: str,
        captions: dict[int, str],
        image_transform: Callable | None = None,
    ):

        self.image_ids = image_ids

        self.parser = parser
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

        image = self._load_image(image_id)

        caption = self._load_caption(image_id)

        graph = self._load_graph(image_id)

        return {

            "image": image,

            "text": caption,

            "graph": graph,

            "image_id": image_id

        }

    def _load_image(
        self,
        image_id: int
    ):

        image_path = self.images_dir / f"{image_id}.jpg"

        image = Image.open(image_path).convert("RGB")

        if self.image_transform is not None:
            image = self.image_transform(image)

        return image

    def _load_caption(
        self,
        image_id: int
    ) -> str:

        return self.captions[image_id]

    def _load_graph(
        self,
        image_id: int
    ) -> Data:

        scene_graph: SceneGraph = self.parser.parse_image(
            str(image_id)
        )

        pyg_graph = self.graph_converter.convert(
            scene_graph
        )

        return pyg_graph