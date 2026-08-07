import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from vocabulary import RelationVocabulary

BASE_DIR = Path(__file__).resolve().parent

SYNSET_PATH = BASE_DIR / "resources" / "relationship_synsets.json"
RELATIONSHIP_PATH = BASE_DIR / "resources" / "relationships.json"
OUTPUT_PATH = BASE_DIR / "resources" / "final_vocab.json"


def build_relation_vocabulary() -> None:

    print("1. Loading relationship synsets...")

    vocab = RelationVocabulary(SYNSET_PATH)

    print("2. Reading Visual Genome relationships...")

    total_relationships = 0

    with open(RELATIONSHIP_PATH, "r", encoding="utf-8") as f:

        dataset = json.load(f)

    for image in dataset:

        for relation in image.get("relationships", []):

            predicate = relation.get("predicate")

            if not predicate:
                continue

            vocab.observe(predicate)

            total_relationships += 1

    print(f"   Total predicates : {total_relationships:,}")

    print("3. Building vocabulary...")

    vocab.build()

    print("4. Saving vocabulary...")

    vocab.save(OUTPUT_PATH)

    print("\nBuild completed successfully.")
    print(f"Unique synsets : {len(vocab.relation_to_id):,}")
    print(f"Output         : {OUTPUT_PATH}")


if __name__ == "__main__":
    build_relation_vocabulary()