import torch
import lightning as L

from dataprovider import TwitterHatredSpeechForWordEmbedding
from dataprovider.twitter_hs_process import transform_fn_text_cleanup
from model import LSTMForSequenceClassification

## data process
transform_fn = transform_fn_text_cleanup()
train_data = TwitterHatredSpeechForWordEmbedding(
    "./data/twitter_hatred_speech/test.csv",
    transform=transform_fn,
)

print(train_data[0]["input_ids"])
print(train_data[0:2]["input_ids"])


train_loader = torch.utils.data.DataLoader(TwitterHatredSpeechForWordEmbedding)
model = LSTMForSequenceClassification(
    hidden_size=128, vocab_size=30000, embedding_dim=256
)

trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=model, train_dataloaders=train_loader)
