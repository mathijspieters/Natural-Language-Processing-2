import torch
import torch.nn as nn

class RNNLM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, latent_size, num_layers, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hidden_size,
            num_layers=num_layers)
        self.hidden2out = nn.Linear(hidden_size, latent_size)
        self.act = nn.Softmax(dim=-1)

    def forward(self, input):
        out = self.embedding(input)
        out, _ = self.rnn(out)
        out = self.hidden2out(out)
        out = self.act(out)
        return out
