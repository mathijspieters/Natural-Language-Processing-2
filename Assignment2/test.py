import torch
from decoder import Decoder
from RNNLM import RNNLM
from encoder import Encoder
from sent_vae import SentVAE

vocab_size = 100
emb_size = 50
hidden_size = 30
latent_size = 10

seq_len = 3
batch_size = 2

henk = SentVAE(vocab_size, emb_size, hidden_size, latent_size)

input = torch.randint(0, vocab_size, (seq_len, batch_size))
b = henk(input)

print(b.size())
print(b)
