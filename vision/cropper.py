import sys
from pathlib import Path

# Proje ana dizinini Python'ın arama yoluna ekliyoruz
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
from PIL import Image
from core.entities import BoundingBox


class ImageCropper:
    """
    Görüntü İşleme Yardımcısı:
    Visual Genome BoundingBox koordinatlarını kullanarak 
    orijinal resimden nesne parçalarını kırpar ve PIL formatına çevirir.
    """
    @staticmethod
    def crop_bounding_box(img_input, bbox: BoundingBox) -> Image.Image:
        """
        Args:
            img_input: Orijinal .jpg resminin dosya yolu (str) VEYA önceden okunmuş NumPy matrisi.
            bbox: Kırpılacak nesnenin (x_min, y_min, x_max, y_max) koordinatları.
        """
        # Eğer dışarıdan dosya yolu geldiyse oku, NumPy matrisi geldiyse doğrudan kullan
        if isinstance(img_input, str):
            img = cv2.imread(img_input)
            if img is None:
                raise FileNotFoundError(f"Görüntü okunamadı: {img_input}")
        else:
            img = img_input
        
        height, width, _ = img.shape

        x1 = max(0, int(bbox.x_min))
        y1 = max(0, int(bbox.y_min))
        x2 = min(width, int(bbox.x_max))
        y2 = min(height, int(bbox.y_max))

        if x2 <= x1 or y2 <= y1:
            x2 = min(x1 + 1, width)
            y2 = min(y1 + 1, height)

        cropped_bgr = img[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_rgb)

        return pil_img