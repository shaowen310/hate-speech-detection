import pandas as pd
import torch
from torch.utils.data import Dataset


class TwitterHatredSpeech(Dataset):
    def __init__(self, csv_file, transform=None):
        dtype = {"id": int, "label": int, "tweet": str}
        self.df = pd.read_csv(csv_file, header=0, dtype=dtype)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            iter(idx)
        except TypeError:
            if type(idx) is not slice:
                idx = [idx]

        sample = self.df.iloc[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
