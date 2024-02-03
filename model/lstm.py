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
        self.lstm = nn.LSTM(self.embedding_dim, hidden_size, dropout=recurrent_dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, x, labels=None):
        output = self.embedding(x)
        output, _ = self.lstm(output)
        output = F.dropout(output, p=self.dropout_rate)
        logits = self.classifier(output)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + logits

        return outputs  # (loss), logits


# model = Sequential()
#     model.add(Embedding(len(word_index) + 1,
#                      300,
#                      weights=[embedding_matrix],
#                      input_length=max_len,
#                      trainable=False))

#     model.add(LSTM(100))
#     # dropout=0.3, recurrent_dropout=0.3
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
