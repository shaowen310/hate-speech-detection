import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class TwitterHatredSpeech(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file, header=0)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.df.iloc[idx].to_dict("list")

        if self.transform:
            sample = self.transform(sample)

        return sample
