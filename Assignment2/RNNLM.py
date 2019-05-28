import torch
import torch.nn as nn

class RNNLM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, pad_idx, device):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden2out = nn.Linear(hidden_size, vocab_size)

        self.device = device
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, lengths):
        seq_len, batch_size = input.size()

        out = self.embedding(input)
        out = nn.utils.rnn.pack_padded_sequence(out, lengths)
        out, _ = self.rnn(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        out = out.contiguous()
        out = out.view(seq_len * batch_size, self.hidden_size)
        out = self.hidden2out(out)
        out = out.view(seq_len, batch_size, self.vocab_size)
        return out

    def sample(self, SOS, seq_len, batch_size=8, sample=False):
        with torch.no_grad():
            out = torch.ones(batch_size, 1, dtype=torch.long).fill_(SOS).to(self.device)

            input_ = self.embedding(out)
            out, hidden = self.rnn(input_)
            out = self.hidden2out(out)
            if sample:
                out = out.reshape(batch_size, -1)
                softmax = self.softmax(out)
                out = softmax.multinomial(1)
            else:
                out = out.argmax(dim=-1)
            sent = out

            for _ in range(seq_len-1):
                input_ = self.embedding(out)
                out, hidden = self.rnn(input_, hidden)
                out = self.hidden2out(out)
                if sample:
                    out = out.reshape(batch_size, -1)
                    softmax = self.softmax(out)
                    out = softmax.multinomial(1)
                else:
                    out = out.argmax(dim=-1)
                sent = torch.cat([sent, out], dim=1)

        return sent
