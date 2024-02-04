from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataprovider.twitter_hatred_speech import (
    TwitterHatredSpeech,
    TwitterHatredSpeechForWordEmbedding,
)
from dataprovider.twitter_hs_process import transform_fn_text_cleanup
from dataprovider.twitter_hs_collate import collate_batch
from model import LSTMForSequenceClassification

## data process
transform_fn = transform_fn_text_cleanup()
data_raw = TwitterHatredSpeech(
    "./data/twitter_hatred_speech/train.csv",
    transform=transform_fn,
)
data = TwitterHatredSpeechForWordEmbedding(
    data_raw,
    vocab_size=10000,
)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


train_loader = DataLoader(train_data, batch_size=16, collate_fn=collate_batch)
test_loader = DataLoader(test_data, batch_size=16, collate_fn=collate_batch)

model = LSTMForSequenceClassification(
    hidden_size=128, vocab_size=10002, embedding_dim=256
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
for epoch in range(10):
    for b_i, batch in enumerate(train_loader):
        # Forward pass: Compute predicted y by passing x to the model
        loss, _ = model(batch["input_ids"], label=batch["label"])

        if b_i % 100 == 99:
            print(f"Epoch {epoch} Batch {b_i} Loss {loss.item()}")

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_labels = []
        test_preds = []
        for batch in test_loader:
            logits = model(batch["input_ids"])
            test_preds.extend(torch.argmax(F.softmax(logits, dim=1), dim=1).tolist())
            test_labels.extend(batch["label"].tolist())
        accu = accuracy_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds)
        confus = confusion_matrix(test_labels, test_preds)

        print(f"Epoch {epoch} Accuracy {accu} f1 {f1}")
        print(f"Confusion matrix\n{confus}")
