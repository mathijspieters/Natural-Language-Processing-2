import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, z_dim):
        super().__init__()

        self.net = nn.Sequential(nn.Embedding(vocab_size, emb_size),
                      nn.GRU(input_size=vocab_size, hidden_size=hidden_size,
                          num_layers=1),
                      nn.Linear(hidden_size, z_dim),
                      nn.Softmax(dim=-1))

        # self.embedding = nn.Embedding(vocab_size, emb_size)
        # self.rnn = nn.GRU(input_size=vocab_size, hidden_size=hidden_size,
        #     num_layers=1)
        # self.hidden2out = nn.Linear(hidden_size, z_dim)
        # self.act = nn.Softmax(dim=-1)

    def forward(self, input):
        out = self.net(input)
        return out
