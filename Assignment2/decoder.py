import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, latent_size, num_layers, pad_idx, sos_idx):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.z2hidden = nn.Linear(latent_size, num_layers*hidden_size)
        self.z_act = nn.Tanh()

        self.embedder = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hidden_size,
            num_layers=num_layers)
        self.hidden2out = nn.Linear(hidden_size, vocab_size)
        self.act = nn.Softmax(dim=-1)

        self.sos_idx = sos_idx


    def forward(self, z, seq_len):
        # [seq, batch]
        x0 = torch.ones(1, z.size(0))*self.sos_idx

        # hidden = [batch_size, num_layers*hidden_size]
        hidden = self.z_act(self.z2hidden(z))
        # hidden = [batch_size, num_layers, hidden_size]
        hidden = hidden.reshape(hidden.size(0), self.num_layers, self.hidden_size)
        # hidden = [num_layers, batch_size, hidden_size]
        hidden = hidden.transpose(0, 1)

        sent = None

        for t in range(seq_len):
            out = self.embedder(x0.long())
            out, hidden = self.rnn(out, hidden)
            out = self.hidden2out(out)
            out = self.act(out)
            x0 = torch.ones(1, z.size(0))*out.argmax(dim=-1).float()
            if sent is None:
                sent = out
            else:
                sent = torch.cat([sent, out], dim=0)
        return sent