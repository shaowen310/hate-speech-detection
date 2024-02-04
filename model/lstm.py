import torch.nn as nn
import torch.nn.functional as F


class LSTMForSequenceClassification(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_labels=2,
        vocab_size=None,
        embedding_dim=None,
        embedding_weight=None,
        recurrent_dropout=0.1,
        dropout=0.1,
        num_lstm_layers=2,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.dropout_rate = dropout

        if embedding_weight is not None:
            self.embedding_dim = embedding_weight.size(dim=-1)
            self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        else:
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            self.embedding_dim,
            hidden_size,
            num_layers=num_lstm_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, x, label=None):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = F.dropout(lstm_out, p=self.dropout_rate)
        logits = self.fc(lstm_out)

        if label is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), label.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
            return loss, logits
        else:
            return logits
