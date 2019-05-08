import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size=256, latent_size=20, num_layers=1):
        super().__init__()

        self.z2hidden = nn.Linear(latent_size, 2*hidden_size)
        self.z_act = nn.Tanh()

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hidden_size,
            num_layers=num_layers)
        self.hidden2out = nn.Linear(hidden_size, latent_size)
        self.act = nn.Softmax(dim=-1)

    def forward(self, input, z):
        hidden = self.z_act(self.z2hidden(z))
        hidden = hidden.chunk(2, dim=-1)

        out = self.embedding(input)
        out, _ = self.rnn(out, hidden)
        out = self.hidden2out(out)
        out = self.act(out)
        return out
