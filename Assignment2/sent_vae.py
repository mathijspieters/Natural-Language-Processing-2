import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class SentVAE(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, latent_size, num_layers, pad_idx, sos_idx):
        super().__init__()

        self.encoder = Encoder(vocab_size, emb_size, hidden_size, latent_size, num_layers, pad_idx)
        self.decoder = Decoder(vocab_size, emb_size, hidden_size, latent_size, num_layers, pad_idx, sos_idx)

    def forward(self, input):
        mu, sigma = self.encoder(input)
        eps = torch.normal(torch.zeros_like(mu))
        z = mu + sigma*eps
        out = self.decoder(z, input.size(0))
        return out
