import torch
from decoder import Decoder
from RNNLM import RNNLM
from encoder import Encoder
from sent_vae import SentVAE

# vocab_size = 100
# emb_size = 50
# hidden_size = 30
# latent_size = 10
# num_layers = 1
# pad_idx = 0
# sos_idx = 1
#
# seq_len = 3
# batch_size = 2
#
# henk = SentVAE(vocab_size, emb_size, hidden_size, latent_size, num_layers, pad_idx, sos_idx)
#
# input = torch.randint(0, vocab_size, (seq_len, batch_size))
# b = henk(input)
#
# print(b.size())
# print(b)

x = 2
y = 3
z = 5

a = torch.rand(x, y, z)
b = torch.randint(0, z, (x, y)).unsqueeze(-1)

print(a)
print(b)
print(a.gather(dim=-1, index=b).size())
