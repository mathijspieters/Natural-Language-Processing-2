import torch
from decoder import Decoder

vocab_size = 100
emb_size = 50
hidden_size = 30
latent_size = 10

henk = Decoder(vocab_size, emb_size, hidden_size, latent_size)

a = torch.randint(0, vocab_size, (3, 2)).long()
print(a)
b = henk(a)

print(b.sum(-1))
