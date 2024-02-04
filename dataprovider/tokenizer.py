import collections
import os
import logging
from typing import List, Optional, Tuple

from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class NLTKWordTokenizer:
    def __init__(
        self, vocab=None, vocab_file=None, unk_token="<UNK>", pad_token="[PAD]"
    ):
        if vocab_file is not None:
            if not os.path.isfile(vocab_file):
                raise ValueError(
                    f"Can't find a vocabulary file at path '{vocab_file}'."
                )

            self.vocab = load_vocab(vocab_file)

        if vocab is not None:
            self.vocab = vocab
            self.vocab.insert(0, pad_token)
            self.vocab.append(unk_token)
            self.vocab = collections.OrderedDict(
                [(ids, tok) for tok, ids in enumerate(self.vocab)]
            )

        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )

        self.unk_token = unk_token
        self.pad_token = pad_token

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join(tokens).strip()
        return out_string

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        index = 0
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "")
                + VOCAB_FILES_NAMES["vocab_file"],
            )
        else:
            vocab_file = (
                filename_prefix + "-" if filename_prefix else ""
            ) + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {vocab_file}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!"
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

    # def fit_on_texts(self, texts):

    def tokenize(self, text):
        tokens = word_tokenize(text)
        tokens = [
            word if word in self.vocab else self.vocab[self.unk_token]
            for word in tokens
        ]

        return tokens

    def __call__(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def convert_tokens_to_ids(self, tokens):
        return [self._convert_token_to_id(token) for token in tokens]
