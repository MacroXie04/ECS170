"""Leakage-aware loaders for the instructor's Stage 4 text datasets.

The IMDb test directory is never repartitioned. Jokes are split by normalized
content so copies of a joke cannot occur in more than one partition. All
vocabularies are fitted on the resulting training partition only.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from local_code.base_class.dataset import dataset

PAD, UNK, BOS, EOS = range(4)
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
_WORD_OR_PUNCT = re.compile(r"[^\W_]+(?:'[^\W_]+)*|[^\w\s]", re.UNICODE)
_HTML_TAG = re.compile(r"<[^>]*>")


def tokenize(text: str) -> list[str]:
    """Lowercase and remove HTML, retaining negation, contractions and punctuation."""
    text = unicodedata.normalize("NFKC", html.unescape(str(text)))
    text = _HTML_TAG.sub(" ", text).replace("’", "'").replace("‘", "'")
    return _WORD_OR_PUNCT.findall(text.lower())


def detokenize(tokens: Iterable[str]) -> str:
    """Join the tokenizer's word/punctuation sequence into readable prose."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%\)\]\}])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return text.strip()


class Vocabulary:
    """A deterministic vocabulary with stable special-token indices."""

    PAD, UNK, BOS, EOS = PAD, UNK, BOS, EOS

    def __init__(self, tokens: Sequence[str] = SPECIAL_TOKENS):
        supplied = list(tokens)
        if supplied[:4] == list(SPECIAL_TOKENS):
            supplied = supplied[4:]
        regular = list(dict.fromkeys(t for t in supplied if t not in SPECIAL_TOKENS))
        self.tokens = list(SPECIAL_TOKENS) + regular
        self.stoi = {token: index for index, token in enumerate(self.tokens)}

    @classmethod
    def build(cls, texts: Iterable[str], max_vocab: int = 20_000) -> "Vocabulary":
        if max_vocab < len(SPECIAL_TOKENS):
            raise ValueError("max_vocab must be at least 4, including special tokens")
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))
        ordered = sorted(counts, key=lambda word: (-counts[word], word))
        return cls(ordered[:max_vocab - len(SPECIAL_TOKENS)])

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        encoded = [self.stoi.get(token, UNK) for token in tokenize(text)]
        return ([BOS] if bos else []) + encoded + ([EOS] if eos else [])

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        words = []
        for value in ids:
            index = int(value)
            if index < 0 or index >= len(self.tokens):
                raise ValueError(f"Token index outside vocabulary: {index}")
            if skip_special and index in (PAD, BOS, EOS):
                continue
            # Preserve <unk>: deleting unknown words would conceal OOV behavior.
            words.append(self.tokens[index])
        return detokenize(words)

    def to_dict(self) -> dict:
        return {"tokens": self.tokens}

    @classmethod
    def from_dict(cls, value: dict) -> "Vocabulary":
        return cls(value["tokens"])


def _digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _normalized(text: str) -> str:
    return " ".join(tokenize(text))


def _partition_metadata(partitions: dict[str, list[dict]], vocab: Vocabulary) -> dict:
    split_ids = {name: [row["id"] for row in records] for name, records in partitions.items()}
    split_sets = {name: set(ids) for name, ids in split_ids.items()}
    normalized = {name: {_normalized(row["text"]) for row in records}
                  for name, records in partitions.items()}
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    return {
        "counts": {name: len(records) for name, records in partitions.items()},
        "split_ids": split_ids,
        "split_id_sha256": {name: _digest(ids) for name, ids in split_ids.items()},
        "id_overlap_counts": {f"{a}_{b}": len(split_sets[a] & split_sets[b]) for a, b in pairs},
        "normalized_text_overlap_counts": {f"{a}_{b}": len(normalized[a] & normalized[b]) for a, b in pairs},
        "vocabulary_size": len(vocab),
        "vocabulary_sha256": _digest(vocab.tokens),
        "vocabulary_fit_split": "train",
        "token_counts": {name: sum(len(row["tokens"]) for row in rows) for name, rows in partitions.items()},
        "unknown_token_counts": {name: sum(row["tokens"].count(UNK) for row in rows) for name, rows in partitions.items()},
    }


class Dataset_Loader(dataset):
    def __init__(self, data_dir, seed: int = 1704, max_vocab: int = 20_000,
                 max_length: int = 256, validation_fraction: float = 0.1):
        super().__init__("Stage 4 text datasets", "IMDb sentiment and short-joke generation")
        self.data_dir = Path(data_dir).expanduser().resolve()
        if not (self.data_dir / "text_classification").exists() and (self.data_dir / "stage_4_data").is_dir():
            self.data_dir = self.data_dir / "stage_4_data"
        if max_length < 1:
            raise ValueError("max_length must be positive")
        if max_vocab < 4:
            raise ValueError("max_vocab must include all four special tokens")
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be strictly between zero and one")
        self.seed = int(seed)
        self.max_vocab = int(max_vocab)
        self.max_length = int(max_length)
        self.validation_fraction = float(validation_fraction)

    def load(self):
        return self.load_classification()

    def load_classification(self) -> dict:
        root = self.data_dir / "text_classification"
        official = {"train": [], "test": []}
        source_hash = hashlib.sha256()
        for split in official:
            for label_name, label in (("neg", 0), ("pos", 1)):
                paths = sorted((root / split / label_name).glob("*.txt"))
                if not paths:
                    raise FileNotFoundError(f"No review text files in {root / split / label_name}")
                for path in paths:
                    payload = path.read_bytes()
                    identifier = path.relative_to(root).as_posix()
                    source_hash.update(identifier.encode("utf-8") + b"\0" + payload + b"\0")
                    official[split].append({"id": identifier,
                                            "text": payload.decode("utf-8", errors="replace"),
                                            "label": label})
        rng = random.Random(self.seed)
        partitions = {"train": [], "validation": [], "test": official["test"]}
        # Group only official-training texts. Test content never influences the
        # validation choice, even when the original dataset contains duplicates.
        groups = defaultdict(list)
        for row in official["train"]:
            groups[_normalized(row["text"])].append(row)
        label_counts = Counter(row["label"] for row in official["train"])
        if any(label_counts[label] < 2 for label in (0, 1)):
            raise ValueError("Each training class needs at least two examples")
        targets = {label: min(label_counts[label] - 1,
                             max(1, round(label_counts[label] * self.validation_fraction)))
                   for label in (0, 1)}
        remaining = dict(targets)
        group_keys = sorted(groups)
        rng.shuffle(group_keys)
        for key in group_keys:
            group = groups[key]
            group_labels = Counter(row["label"] for row in group)
            # Whole groups enter validation only when they fit both class
            # budgets. Singleton groups normally fill the remaining slots.
            use_validation = all(group_labels[label] <= remaining[label] for label in (0, 1))
            if use_validation:
                partitions["validation"].extend(group)
                for label in (0, 1):
                    remaining[label] -= group_labels[label]
            else:
                partitions["train"].extend(group)
        if not partitions["validation"]:
            raise ValueError("No complete text group fits the validation budget; increase validation_fraction")
        rng.shuffle(partitions["train"])
        rng.shuffle(partitions["validation"])
        vocab = Vocabulary.build((r["text"] for r in partitions["train"]), self.max_vocab)
        truncated_counts = {}
        for name, rows in partitions.items():
            truncated_counts[name] = 0
            for row in rows:
                ids = vocab.encode(row["text"])
                truncated_counts[name] += len(ids) > self.max_length
                row["tokens"] = ids[:self.max_length] or [UNK]
        metadata = _partition_metadata(partitions, vocab)
        metadata.update({
            "task": "classification", "dataset": "IMDb", "seed": self.seed,
            "source_sha256": source_hash.hexdigest(),
            "official_split_counts": {name: len(rows) for name, rows in official.items()},
            "split_policy": "Seeded normalized-text groups from official train; greedy per-class validation budgets; official test unchanged",
            "official_train_normalized_group_count": len(groups),
            "official_train_duplicate_normalized_rows": len(official["train"]) - len(groups),
            "validation_target_label_counts": {str(label): target for label, target in targets.items()},
            "validation_fraction": self.validation_fraction,
            "max_length": self.max_length, "max_vocab": self.max_vocab,
            "truncated_counts": truncated_counts,
            "label_mapping": {"negative": 0, "positive": 1},
            "label_counts": {name: dict(Counter(str(r["label"]) for r in rows)) for name, rows in partitions.items()},
            "cleaning": "HTML decoding/removal, Unicode normalization, lowercasing; keep negation, contractions and punctuation",
        })
        return {**partitions, "vocab": vocab, "metadata": metadata}

    def load_generation(self) -> dict:
        path = self.data_dir / "text_generation" / "data"
        with path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or not {"ID", "Joke"}.issubset(reader.fieldnames):
                raise ValueError("Expected a CSV with ID and Joke columns")
            rows = [{"id": f"joke/{row['ID']}", "text": row["Joke"]} for row in reader]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Joke IDs must be unique")
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            normalized = _normalized(row["text"])
            if not normalized:
                raise ValueError(f"Joke {row['id']} has no text after normalization")
            groups[normalized].append(row)
        keys = sorted(groups)
        if len(keys) < 3:
            raise ValueError("At least three distinct jokes are required for train/validation/test")
        rng = random.Random(self.seed)
        rng.shuffle(keys)
        n_holdout = min((len(keys) - 1) // 2, max(1, round(len(keys) * 0.1)))
        group_splits = {"test": keys[:n_holdout], "validation": keys[n_holdout:2*n_holdout],
                        "train": keys[2*n_holdout:]}
        partitions = {name: [row for key in group_splits[name] for row in groups[key]]
                      for name in ("train", "validation", "test")}
        vocab = Vocabulary.build((r["text"] for r in partitions["train"]), self.max_vocab)
        for records in partitions.values():
            for row in records:
                # A joke is one independent sequence. Never truncate or join jokes.
                row["tokens"] = vocab.encode(row["text"], bos=True, eos=True)
        metadata = _partition_metadata(partitions, vocab)
        metadata.update({
            "task": "generation", "dataset": "Short jokes", "seed": self.seed,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_count": len(rows), "normalized_group_count": len(groups),
            "duplicate_normalized_rows": len(rows) - len(groups), "dropped_rows": 0,
            "split_policy": "80/10/10 by shuffled normalized-content groups; duplicates kept together",
            "group_counts": {name: len(keys) for name, keys in group_splits.items()},
            "split_fractions_requested": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "max_vocab": self.max_vocab, "truncated_count": 0,
            "max_sequence_length_including_bos_eos": max(len(r["tokens"]) for r in rows),
            "cleaning": "HTML decoding/removal, Unicode normalization, lowercasing; keep contractions and punctuation",
        })
        return {**partitions, "vocab": vocab, "metadata": metadata}
