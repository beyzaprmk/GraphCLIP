from __future__ import annotations

import torch

from core.entities import SceneGraph
from data.graph_converter import GraphConverter


class GraphCLIPPredictor:

    def __init__(
        self,
        model,
        graph_converter: GraphConverter,
        device: str | None = None,
    ):
        self.model = model
        self.graph_converter = graph_converter

        if device is None:

            if torch.backends.mps.is_available():
                self.device = torch.device("mps")

            elif torch.cuda.is_available():
                self.device = torch.device("cuda")

            else:
                self.device = torch.device("cpu")

        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_graph(
        self,
        scene_graph: SceneGraph,
    ) -> torch.Tensor:

        graph_data = self.graph_converter.convert(
            scene_graph
        )

        graph_data = graph_data.to(
            self.device
        )

        embedding = self.model.encode_graph(
            graph_data
        )

        return embedding