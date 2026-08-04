import sys
from pathlib import Path

# Proje ana dizinini (GraphCLIP) Python'ın arama yoluna ekliyoruz
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
from PIL import Image
from core.entities import BoundingBox #[cite: 6]


class ImageCropper:
    """
    Görüntü İşleme Yardımcısı:
    Visual Genome BoundingBox koordinatlarını kullanarak 
    orijinal resimden nesne parçalarını kırpar ve PIL formatına çevirir.
    """
    @staticmethod
    def crop_bounding_box(image_path: str, bbox: BoundingBox) -> Image.Image:
        """
        Args:
            image_path: Orijinal .jpg resminin yolu.
            bbox: Kırpılacak nesnenin (x_min, y_min, x_max, y_max) koordinatları.
            
        Returns:
            PIL.Image: Kırpılmış ve RGB formatına dönüştürülmüş resim nesnesi.
        """
        # 1. OpenCV ile resmi diskten oku (BGR formatında gelir)
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Görüntü okunamadı: {image_path}")
        
        # Görüntünün matris boyutlarını al (yükseklik, genişlik)
        height, width, _ = img.shape

        # 2. Koordinatları piksel sınırları içinde kalacak şekilde tamsayıya (int) çevir
        x1 = max(0, int(bbox.x_min))
        y1 = max(0, int(bbox.y_min))
        x2 = min(width, int(bbox.x_max))
        y2 = min(height, int(bbox.y_max))

        # Hatalı/Bozuk kutu kontrolü (Genişlik veya yükseklik 0 veya negatifse)
        if x2 <= x1 or y2 <= y1:
            # En az 1x1 piksellik güvenli bir kırpma alanı oluştur
            x2 = max(x1 + 1, width)
            y2 = max(y1 + 1, height)

        # 3. OpenCV Numpy slicing ile görüntüyü kes: [y_başlangıç:y_bitiş, x_başlangıç:x_bitiş]
        cropped_bgr = img[y1:y2, x1:x2]

        # 4. ViT modellerinin anlayacağı RGB formatına ve PIL Image objesine dönüştür
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped_rgb)

        return pil_img


