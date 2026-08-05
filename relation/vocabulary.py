from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class RelationVocabulary:
    

    def __init__(self, synset_path: Path):

        self.synset_map = self._load_synsets(synset_path)

        self.counter = Counter()

        self.relation_to_id = {}
        self.id_to_relation = {}

        self._built = False

    @staticmethod
    def _clean(predicate: str) -> str:
        return predicate.lower().strip()

    @staticmethod
    def _load_synsets(path: Path) -> dict[str, str]:

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            key.lower().strip(): value.strip()
            for key, value in data.items()
        }

    def observe(self, predicate: str) -> None:
        """
        Called once for every predicate in relationships.json.
        Only canonical WordNet synsets are counted.
        """

        predicate = self._clean(predicate)

        synset = self.synset_map.get(predicate)

        # Predicate synset dosyasında yoksa
        # vocabulary'ye EKLEME.
        if synset is None:
            return

        self.counter[synset] += 1

    def build(self) -> None:
        """
        Assign deterministic IDs.
        """

        ordered = sorted(self.counter.keys())

        self.relation_to_id = {
            relation: idx
            for idx, relation in enumerate(ordered)
        }

        self.id_to_relation = {
            idx: relation
            for relation, idx in self.relation_to_id.items()
        }

        self._built = True

    def encode(self, predicate: str) -> int:
        """
        Runtime:
        wearing -> wear.v.01 -> id
        """

        if not self._built:
            raise RuntimeError("Vocabulary has not been built.")

        predicate = self._clean(predicate)

        synset = self.synset_map.get(predicate)

        if synset is None:
            raise KeyError(f"Unknown predicate: {predicate}")

        return self.relation_to_id[synset]

    def decode(self, relation_id: int) -> str:

        if not self._built:
            raise RuntimeError("Vocabulary has not been built.")

        return self.id_to_relation[relation_id]

    def save(self, output_path: Path) -> None:

        if not self._built:
            raise RuntimeError("Vocabulary has not been built.")

        data = {
            "metadata": {
                "relation_count": len(self.relation_to_id)
            },
            "relations": {
                relation: {
                    "id": relation_id,
                    "count": self.counter[relation]
                }
                for relation, relation_id in self.relation_to_id.items()
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False
            )

    @classmethod
    def load(cls, vocab_path: Path):

        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        obj = cls.__new__(cls)

        obj.synset_map = {}

        obj.counter = Counter()

        obj.relation_to_id = {
            relation: info["id"]
            for relation, info in data["relations"].items()
        }

        obj.id_to_relation = {
            idx: relation
            for relation, idx in obj.relation_to_id.items()
        }

        obj._built = True

        return obj