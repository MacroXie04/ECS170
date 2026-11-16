import csv
import html
import re
from collections import Counter

import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

from ..base_class.dataset import dataset

PAD, UNK, EOS = 0, 1, 2
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?|[.!?,;:]")


def tokenize(text):
    text = html.unescape(text).lower().replace("’", "'")
    text = re.sub(r"<[^>]+>", " ", text)
    return TOKEN.findall(text)


class Dataset_Loader(dataset):
    def __init__(self, root):
        super().__init__("Stage 4", "IMDb reviews and short jokes")
        self.root = root

    def reviews(self, split, smoke):
        rows = []
        for name, label in [("neg", 0), ("pos", 1)]:
            folder = self.root / "text_classification" / split / name
            files = sorted(folder.glob("*.txt"))
            if len(files) != 12500:
                raise ValueError(f"{split}/{name}: expected 12500 files, found {len(files)}")
            for path in (files[:64] if smoke else files):
                rows.append((path.read_text(encoding="utf-8"), label))
        return rows
        # one comment is one row.

    def load(self, task="classify", smoke=False):
        if task == "classify":
            original = self.reviews("train", smoke)

            labels = []
            for review in original:
                labels.append(review[1])
                # The only label we have are 0(neg) and 1(pos), labels are a list
                # that we created to save the label of all rows(the output list in function "reviews").

            train, val = train_test_split(
                original,
                test_size=0.1,
                random_state=999,
                stratify=labels,
                # stratify to make sure the distribution of pos/neg in train/validify set
                # are the same(it's based on the distribution of how many pos/neg we provide as data).
            )

            splits = {"train": train, "val": val, "test": self.reviews("test", smoke)}

            stream = []
            for review in train:
                text = review[0]
                tokens = tokenize(text)
                stream.extend(tokens[:200])

            cap = 20000
            # 200 is the maximum number of token we get from each row, 20000 is the maximum size of our finalized vocabulary.
        else:
            path = self.root / "text_generation" / "data"
            rows = []
            with path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    joke = row["Joke"]
                    tokens = tokenize(joke)
                    rows.append(tuple(tokens))

            if len(rows) != 1622:
                raise ValueError(f"Expected 1622 jokes, found {len(rows)}")

            long_enough = []
            for joke_tokens in rows:
                if len(joke_tokens) >= 4:
                    long_enough.append(joke_tokens)
            unique = list(dict.fromkeys(long_enough))

            # Keep the test jokes separate from the validation used to select an epoch.
            train, held_out = train_test_split(unique, test_size=0.2, random_state=999)
            val, test = train_test_split(held_out, test_size=0.5, random_state=999)
            if smoke:
                train, val, test = train[:128], val[:32], test[:32]
            splits = {"train": train, "val": val, "test": test}

            stream = []
            for joke_tokens in train:
                stream.extend(joke_tokens)

            cap = None

        vocab = {"<pad>": PAD, "<unk>": UNK, "<eos>": EOS}
        for word, count in Counter(stream).most_common(cap): # "cap", which is 20000 for classification task,limits how many words we pick through most_common method: in total there are much more than 20000 words in "stream", we just choose the 20000 most common ones.
            if count >= 2:
                vocab[word] = len(vocab)

        encoded = {}

        for split_name in splits:
            split_rows = splits[split_name]
            encoded_rows = []

            if task == "classify":
                for review in split_rows:
                    text = review[0]
                    label = review[1]
                    tokens = tokenize(text)[:200]

                    token_ids = []
                    for token in tokens:
                        token_id = vocab.get(token, UNK)
                        token_ids.append(token_id)

                    if len(token_ids) == 0:
                        token_ids.append(UNK)

                    encoded_review = (token_ids, label)
                    encoded_rows.append(encoded_review)

            else:
                for joke_tokens in split_rows:
                    token_ids = []
                    for token in joke_tokens:
                        token_id = vocab.get(token, UNK)
                        token_ids.append(token_id)

                    token_ids.append(EOS)
                    encoded_rows.append(token_ids)

            encoded[split_name] = encoded_rows

        training_jokes = []
        if task == "generate":
            training_jokes = train

        return encoded, vocab, training_jokes


def classify_batch(rows):
    xs, ys = zip(*rows)
    ids = pad_sequence(
        [torch.tensor(x) for x in xs], batch_first=True, padding_value=PAD,
    )
    return ids, torch.tensor(ys, dtype=torch.long)


def generate_batch(rows):
    xs = [torch.tensor(row[:-1]) for row in rows]
    ys = [torch.tensor(row[1:]) for row in rows]
    for y in ys:
        # The caller supplies three tokens, so loss starts at the fourth token.
        y[:2] = PAD
    return (
        pad_sequence(xs, batch_first=True, padding_value=PAD),
        pad_sequence(ys, batch_first=True, padding_value=PAD),
    )
