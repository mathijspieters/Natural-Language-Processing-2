import torch
import torch.nn as nn
import random

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, latent_size, num_layers, pad_idx, sos_idx, word_dropout, device):
        super().__init__()

        self.emb_size = emb_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.z2hidden = nn.Linear(latent_size, num_layers*hidden_size)
        self.z_act = nn.Tanh()

        self.embedder = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hidden_size,
            num_layers=num_layers)
        self.hidden2out = nn.Linear(hidden_size, vocab_size)

        self.sos_idx = sos_idx
        self.word_dropout = word_dropout
        self.device = device


    def forward(self, x, z, seq_len):
        # [seq, batch]
        #x0 = torch.ones(1, z.size(0))*self.sos_idx

        # hidden = [batch_size, num_layers*hidden_size]
        hidden = self.z_act(self.z2hidden(z))
        # hidden = [batch_size, num_layers, hidden_size]
        hidden = hidden.reshape(hidden.size(0), self.num_layers, self.hidden_size)
        # hidden = [num_layers, batch_size, hidden_size]
        hidden = hidden.transpose(0, 1).contiguous()

        out = self.embedder(x)
        # for i in range(x.shape[0]):
        #     for j in range(x.shape[1]):
        #         if random.random() < self.word_dropout:
        #             x[i][j] = torch.zeros(self.emb_size, device=self.device)

        out = nn.utils.rnn.pack_padded_sequence(out, seq_len)
        out, _ = self.rnn(out, hidden)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)

        out = self.hidden2out(out)

        return out


    def generate(self, x, z, seq_len):
        hidden = self.z_act(self.z2hidden(z))
        # hidden = [batch_size, num_layers, hidden_size]
        hidden = hidden.reshape(hidden.size(0), self.num_layers, self.hidden_size)
        # hidden = [num_layers, batch_size, hidden_size]
        hidden = hidden.transpose(0, 1).contiguous()
        sent = None

        out = x

        for _ in range(seq_len):
            input_ = self.embedder(out)
            out, hidden = self.rnn(input_, hidden)
            out = self.hidden2out(out)
            out = out.argmax(dim=-1)
            if sent is None:
                sent = out
            else:
                sent = torch.cat([sent, out], dim=0)
        return sent
