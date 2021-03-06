import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, latent_size, num_layers, pad_idx):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedder = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hidden_size,
            num_layers=num_layers, bidirectional=True)
        self.rnn2hidden = nn.Linear(2*hidden_size, hidden_size)
        self.hidden2mu = nn.Linear(hidden_size, latent_size)
        self.hidden2sigma = nn.Linear(hidden_size, latent_size)

        self.act = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(self, input, lengths):
        out = self.embedder(input)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths)
        out, _ = self.rnn(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)

        fn = out[-1, :, 0:int(self.hidden_size)]
        b1 = out[-1, :, int(self.hidden_size):2*self.hidden_size]

        h = torch.cat([fn, b1], dim=-1)
        h = self.tanh(self.rnn2hidden(h))

        mu = self.hidden2mu(h)
        #sigma = self.act(self.hidden2sigma(h))
        sigma = self.hidden2sigma(h)
        return mu, sigma
