import torch
from decoder import Decoder
from RNNLM import RNNLM
from encoder import Encoder

vocab_size = 100
emb_size = 50
hidden_size = 30
latent_size = 10

seq_len = 3
batch_size = 2

henk = Decoder(vocab_size, emb_size, hidden_size, latent_size)

input = torch.randint(0, vocab_size, (seq_len, batch_size)).long()
z = torch.normal(torch.zeros(batch_size, latent_size), 1)
print(input)
b = henk(input, z)

print(b.sum(-1))
