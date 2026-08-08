import json
from tqdm import tqdm

INPUT_FILE = "dataset/region_descriptions.json"

OUTPUT_FILE = "dataset/captions.json"

print(f"{INPUT_FILE} okunuyor, bu biraz sürebilir...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

captions_dict = {}

print("Basit captions.json sözlüğü oluşturuluyor...")

for image_data in tqdm(raw_data):
    image_id = str(image_data["id"])
    

    phrases = [region["phrase"] for region in image_data["regions"]]
    combined_caption = ". ".join(phrases)
    
    captions_dict[image_id] = combined_caption

print("Dosya kaydediliyor...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(captions_dict, f, ensure_ascii=False, indent=2)

print("İşlem tamam! dataset klasörüne captions.json başarıyla oluşturuldu.")