import pandas as pd
import torch
from torch.utils.data import Dataset

from dataprovider.tokenizer import NLTKWordTokenizer
from .text_process_nltk_utils import (
    doc_word_freq,
)


class TwitterHatredSpeech(Dataset):
    def __init__(self, csv_file, transform=None):
        dtype = {"id": int, "label": int, "tweet": str}
        self.df = pd.read_csv(csv_file, header=0, dtype=dtype)
        self.transform = transform

        self.tweets = self.df["tweet"]
        if "label" in self.df.columns:
            self.labels = self.df["label"]
        self.ids = self.df["id"]

        self._process()

    def _process(self):
        if self.transform is not None:
            self.tweets = self.transform(self.tweets)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if type(idx) is int:
            idx = [idx]

        if "label" in self.df.columns:
            return {
                "tweet": self.tweets[idx],
                "label": self.labels[idx],
                "id": self.ids[idx],
            }
        else:
            return {
                "tweet": self.tweets[idx],
                "id": self.ids[idx],
            }


class TwitterHatredSpeechForWordEmbedding(Dataset):
    def __init__(self, data, vocab_size=None, vocab=None):
        self.data = data
        if vocab is None:
            word_freq = doc_word_freq(self.data[:]["tweet"], vocab_size=vocab_size)
            vocab = [x[0] for x in word_freq]

        self.tokenizer = NLTKWordTokenizer(vocab=vocab)

        self.input_idss = self.data.tweets.apply(self.tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if type(idx) is int:
            idx = [idx]

        if "label" in self.data.df.columns:
            return {
                "input_ids": self.input_idss[idx],
                "label": self.data.labels[idx],
                "id": self.data.ids[idx],
            }
        else:
            return {
                "input_ids": self.input_idss[idx],
                "id": self.data.ids[idx],
            }
