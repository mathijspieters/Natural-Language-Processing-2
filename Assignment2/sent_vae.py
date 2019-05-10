import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class SentVAE(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, latent_size, num_layers, pad_idx, sos_idx):
        super().__init__()
        self.latent_size = latent_size
        self.SOS = sos_idx

        self.encoder = Encoder(vocab_size, emb_size, hidden_size, latent_size, num_layers, pad_idx)
        self.decoder = Decoder(vocab_size, emb_size, hidden_size, latent_size, num_layers, pad_idx, sos_idx)

    def forward(self, input, lengths):
        mu, sigma = self.encoder(input, lengths)
        eps = torch.normal(torch.zeros_like(mu))
        z = mu + sigma*eps
        out = self.decoder(input, z, lengths)
        return out, mu, sigma

    def sample(self, n_samples=10, std=1):
        with torch.no_grad():
            z = std*torch.normal(torch.zeros((n_samples, self.latent_size)))
            start_input = torch.ones(1, n_samples, dtype=torch.long).fill_(self.SOS)

            out = self.decoder.generate(start_input, z, 30)

        return out
