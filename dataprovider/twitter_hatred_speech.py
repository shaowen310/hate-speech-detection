import os

import pandas as pd


class TwitterHatredSpeech:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def train_split(self):
        df_train = pd.read_csv(os.path.join(self.data_dir, "train.csv"), header=0)
        return df_train

    def test_split(self):
        df_test = pd.read_csv(os.path.join(self.data_dir, "test.csv"), header=0)
        return df_test
