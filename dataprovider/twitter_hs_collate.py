import torch
from torch.nn.utils.rnn import pad_sequence


def collate_batch(batch):
    (labels, input_idss) = ([], [])

    labels = [x["label"] for x in batch]
    input_idss = [torch.tensor(x["input_ids"], dtype=torch.int64) for x in batch]

    labels = torch.tensor(labels, dtype=torch.int64)

    input_idss = pad_sequence(input_idss, batch_first=True, padding_value=0)

    return {
        "input_ids": input_idss,
        "label": labels,
    }
