from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import spacy

from relation.vocabulary import RelationVocabulary


@dataclass(frozen=True)
class TextEntity:
    text: str


@dataclass(frozen=True)
class TextRelation:
    subject: str
    predicate: str
    object: str
    canonical_relation: str
    relation_id: int


@dataclass(frozen=True)
class TextAnalysis:
    entities: list[TextEntity]
    relations: list[TextRelation]


class TextAnalyzer:
    

    def __init__(
        self,
        vocab_path: str | Path,
        synset_path: str | Path,
        model_name: str = "en_core_web_sm",
    ):
        self.vocab_path = Path(vocab_path)
        self.synset_path = Path(synset_path)

        self.vocab = RelationVocabulary.load(
            self.vocab_path
        )

        self.nlp = self._load_spacy_model(
            model_name
        )

        self.synset_map = self._load_synset_map(
            self.synset_path
        )

   
    def analyze(
        self,
        text: str,
    ) -> TextAnalysis:

        if not text or not text.strip():
            raise ValueError(
                "Analiz edilecek text boş olamaz."
            )

        document = self.nlp(
            text.strip()
        )

        entities = self._extract_entities(
            document
        )

        relations = self._extract_relations(
            document
        )

        return TextAnalysis(
            entities=entities,
            relations=relations,
        )

    @staticmethod
    def _load_spacy_model(
        model_name: str,
    ):
        try:
            return spacy.load(
                model_name
            )
        except OSError as exc:
            raise RuntimeError(
                f"spaCy modeli bulunamadı: {model_name}\n"
                f"Kurulum için:\n"
                f"python -m spacy download {model_name}"
            ) from exc

 
    def _extract_entities(
        self,
        document,
    ) -> list[TextEntity]:

        entities: list[TextEntity] = []
        seen: set[str] = set()

        for chunk in document.noun_chunks:

            entity = self._clean_entity(
                chunk.text
            )

            if not entity:
                continue

            key = entity.lower()

            if key in seen:
                continue

            seen.add(key)

            entities.append(
                TextEntity(
                    text=entity
                )
            )

        return entities

    @staticmethod
    def _clean_entity(
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            return ""

        value = " ".join(
            value.split()
        )

        return value.lower()

    def _extract_relations(
        self,
        document,
    ) -> list[TextRelation]:

        relations: list[TextRelation] = []
        seen: set[
            tuple[str, str, str]
        ] = set()

        for token in document:

            candidates = []

            if token.pos_ in {
                "VERB",
                "AUX",
            }:

                subject = self._find_subject(
                    token
                )

                object_phrase = self._find_direct_object(
                    token
                )

                if (
                    subject is not None
                    and object_phrase is not None
                ):

                    candidates.append(
                        (
                            subject,
                            self._verb_phrase(
                                token
                            ),
                            object_phrase,
                        )
                    )

            if token.dep_ == "prep":

                object_phrase = self._find_prepositional_object(
                    token
                )

                if object_phrase is None:
                    continue

                subject = self._find_subject(
                    token.head
                )

                if subject is None:

                    subject = self._entity_from_head(
                        token.head
                    )

                if subject is None:
                    continue

                candidates.append(
                    (
                        subject,
                        self._preposition_phrase(
                            token
                        ),
                        object_phrase,
                    )
                )

            for (
                subject,
                predicate,
                object_phrase,
            ) in candidates:

                relation = self._resolve_relation(
                    subject=subject,
                    predicate=predicate,
                    object_phrase=object_phrase,
                )

                if relation is None:
                    continue

                key = (
                    relation.subject,
                    relation.canonical_relation,
                    relation.object,
                )

                if key in seen:
                    continue

                seen.add(key)
                relations.append(
                    relation
                )

        return relations

    @staticmethod
    def _find_subject(
        token,
    ) -> str | None:

        for child in token.children:

            if child.dep_ in {
                "nsubj",
                "nsubjpass",
                "csubj",
            }:

                return TextAnalyzer._subtree_text(
                    child
                )

        return None

    @staticmethod
    def _find_direct_object(
        token,
    ) -> str | None:

        for child in token.children:

            if child.dep_ in {
                "dobj",
                "obj",
                "attr",
            }:

                return TextAnalyzer._subtree_text(
                    child
                )

        return None

    @staticmethod
    def _find_prepositional_object(
        prep_token,
    ) -> str | None:

        for child in prep_token.children:

            if child.dep_ in {
                "pobj",
                "obj",
            }:

                return TextAnalyzer._subtree_text(
                    child
                )

        return None

    @staticmethod
    def _entity_from_head(
        token,
    ) -> str | None:

        if token.pos_ in {
            "NOUN",
            "PROPN",
        }:

            return TextAnalyzer._subtree_text(
                token
            )

        return None

    @staticmethod
    def _subtree_text(
        token,
    ) -> str:

        tokens = sorted(
            list(token.subtree),
            key=lambda item: item.i,
        )

        words = [
            item.text
            for item in tokens
            if not item.is_punct
        ]

        return " ".join(
            words
        ).strip().lower()

    @staticmethod
    def _verb_phrase(
        token,
    ) -> str:

        return token.lemma_.strip().lower()

    @staticmethod
    def _preposition_phrase(
        token,
    ) -> str:

        return token.text.strip().lower()

    def _resolve_relation(
        self,
        subject: str,
        predicate: str,
        object_phrase: str,
    ) -> TextRelation | None:

        canonical = self._canonicalize_relation(
            predicate
        )

        if canonical is None:
            return None

        relation_id = self.vocab.relation_to_id.get(
            canonical
        )

        if relation_id is None:
            return None

        return TextRelation(
            subject=subject,
            predicate=predicate,
            object=object_phrase,
            canonical_relation=canonical,
            relation_id=relation_id,
        )

    def _canonicalize_relation(
        self,
        predicate: str,
    ) -> str | None:

        predicate = self._normalize_phrase(
            predicate
        )

        if not predicate:
            return None

        # Önce doğrudan canonical relation kontrolü.
        if predicate in self.vocab.relation_to_id:
            return predicate

        # relationship_synsets.json içindeki phrase
        # eşleşmesini kullan.
        synset = self.synset_map.get(
            predicate
        )

        if synset is None:
            return None

        synset = self._normalize_phrase(
            synset
        )

        if synset not in self.vocab.relation_to_id:
            return None

        return synset

   
    def _load_synset_map(
        self,
        path: Path,
    ) -> dict[str, str]:

        if not path.exists():
            raise FileNotFoundError(
                f"Relation synset dosyası bulunamadı: {path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        mapping: dict[str, str] = {}

        self._collect_synset_pairs(
            data,
            mapping,
        )

        return mapping

    def _collect_synset_pairs(
        self,
        value: Any,
        mapping: dict[str, str],
    ) -> None:

        if isinstance(
            value,
            dict,
        ):

           
            if (
                len(value) == 1
                and all(
                    isinstance(k, str)
                    and isinstance(v, str)
                    for k, v in value.items()
                )
            ):

                for key, target in value.items():

                    mapping[
                        self._normalize_phrase(key)
                    ] = self._normalize_phrase(
                        target
                    )

            for key, child in value.items():

                if isinstance(
                    child,
                    str,
                ):

                    normalized_key = (
                        self._normalize_phrase(
                            key
                        )
                    )

                    normalized_child = (
                        self._normalize_phrase(
                            child
                        )
                    )

                    if normalized_key:
                        mapping[
                            normalized_key
                        ] = normalized_child

                else:

                    self._collect_synset_pairs(
                        child,
                        mapping,
                    )

            return

        if isinstance(
            value,
            list,
        ):

            for item in value:
                self._collect_synset_pairs(
                    item,
                    mapping,
                )

    
    @staticmethod
    def _normalize_phrase(
        value: str,
    ) -> str:

        value = value.strip().lower()

        value = " ".join(
            value.split()
        )

        return value