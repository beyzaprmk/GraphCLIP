import os
import torch
import time  # Donanımı soğutmak için ekledik
from tqdm import tqdm

# Kendi yazdığın ve arayüzlere uyan modülleri çağırıyoruz
from data.vg_parser import VisualGenomeParser
from vision.extractor import ViTFeatureExtractor
from pipline.orchestrator import GraphCLIPPipeline

# Klasör ve Dosya Yolları
DATASET_DIR = "dataset/test_images"  
OBJECTS_JSON = "dataset/objects.json"
RELATIONSHIPS_JSON = "dataset/relationships.json"
OUTPUT_DIR = "processed_data"  # İşlenmiş .pt dosyalarının kaydedileceği klasör

def main():
    # Çıktı klasörü yoksa oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("--- Sistem Ayağa Kaldırılıyor ---")
    
    # 1. Parser ve Extractor'ı başlat (Model burada GPU'ya yüklenecek)
    parser = VisualGenomeParser(
        objects_json_path=OBJECTS_JSON,
        relationships_json_path=RELATIONSHIPS_JSON
    )
    
    extractor = ViTFeatureExtractor(
        images_dir=DATASET_DIR, 
        model_name="openai/clip-vit-base-patch32"
    )
    
    # 2. Arkadaşının yazdığı Orchestrator'ı (Pipeline) kur
    pipeline = GraphCLIPPipeline(parser=parser, extractor=extractor)

    # 3. İşlenecek tüm resim ID'lerini parser sözlüğünden çek   
    image_ids = list(parser.objects_dict.keys())

    
    # # DUMAN TESTİ: Sadece ilk 10 resmi alıyoruz
    # all_image_ids = list(parser.objects_dict.keys())
    # image_ids = all_image_ids[:10]

    print(f"Toplam {len(image_ids)} resim tespit edildi. İşlem başlıyor...\n")

    # 4. İşleme Döngüsü (tqdm ile ilerleme çubuğu ekledik)
    for img_id in tqdm(image_ids, desc="Graf Tensörleri Üretiliyor"):
        out_file = os.path.join(OUTPUT_DIR, f"{img_id}.pt")
        
        # HAYAT KURTARAN DOKUNUŞ: Dosya zaten varsa atla. 
        # Bu sayede işlem yarıda kesilirse scripti tekrar çalıştırdığında kaldığı yerden başlar.
        if os.path.exists(out_file):
            continue
            
        try:
            # İşlemi orchestrator üzerinden tetikle
            rich_graph = pipeline.prepare_data(img_id)
            
            # Oluşan zenginleştirilmiş grafı (içindeki 512'lik tensörlerle birlikte) .pt olarak kaydet
            torch.save(rich_graph, out_file)


            # DONANIM KORUMASI: Her resimden sonra minik bir mola
            time.sleep(0.05)
        except Exception as e:
            # Resim bozuksa, yoksa veya bir hata çıkarsa sistemi çökertme, hatayı bas ve devam et
            print(f"\nHATA - Resim ID {img_id} işlenemedi. Sebep: {str(e)}")
            continue

    print("\n--- Veri Ön İşleme Başarıyla Tamamlandı! ---")
    print(f"Bütün .pt dosyaları '{OUTPUT_DIR}' klasöründe hazır.")

if __name__ == "__main__":
    main()