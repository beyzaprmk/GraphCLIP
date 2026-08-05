import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dataclasses import replace
import cv2
import torch
from core.entities import SceneGraph
from core.interfaces import IVisionExtractor
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from vision.cropper import ImageCropper


class ViTFeatureExtractor(IVisionExtractor):
  """Görsel Öznitelik Çıkarıcı:

  SceneGraph içindeki her bir düğümü kırpar, ViT'ten geçirir ve
  zenginleştirilmiş yeni bir SceneGraph döndürür.
  """

  def __init__(
      self,
      images_dir: str = "data/visual_genome/images",
      model_name: str = "openai/clip-vit-base-patch32",
      device: str = None,
  ):
    self.images_dir = images_dir
    self.device = device if device else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(
        f"Vision Extractor yükleniyor ({model_name}) - Çalıştırma Cihazı:"
        f" {self.device}..."
    )

    self.processor = CLIPProcessor.from_pretrained(model_name)
    self.model = (
        CLIPVisionModelWithProjection.from_pretrained(
            model_name, use_safetensors=True
        )
        .to(self.device)
        .eval()
    )
    print("Vision Extractor başarıyla yüklendi ve hazır!")

  def extract_features(self, graph: SceneGraph) -> SceneGraph:
    """Arayüzün zorunlu tuttuğu tek parametreli metot."""
    image_path = os.path.join(self.images_dir, f"{graph.image_id}.jpg")

    # Disk I/O darboğazını önlemek için görüntü tek seferde belleğe okunur
    img_mat = cv2.imread(image_path)
    if img_mat is None:
      raise FileNotFoundError(f"Görüntü okunamadı: {image_path}")

    yeni_nodelar = []

    for node in graph.nodes:
      cropped_image = ImageCropper.crop_bounding_box(img_mat, node.bbox)
      inputs = self.processor(images=cropped_image, return_tensors="pt").to(
          self.device
      )

      with torch.no_grad():
        outputs = self.model(**inputs)
        image_features = outputs.image_embeds
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )
        feature_tensor = image_features.squeeze(0).cpu()

      guncellenmis_node = replace(node, feature_tensor=feature_tensor)
      yeni_nodelar.append(guncellenmis_node)

    return replace(graph, nodes=yeni_nodelar)

