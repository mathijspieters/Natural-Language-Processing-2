class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size=256, z_dim=20, num_layers=1):
        super().__init__()

        self.embedder = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(input_size=emb_size, hidden_size=hidden_size,
            num_layers=num_layers, bidirectional=True)
        self.rnn2hidden = nn.Linear(2*hidden_size, hidden_size)
        self.hidden2mean = nn.Linear(hidden_size, z_dim)
        self.hidden2sigma = nn.Linear(hidden_size, z_dim)

        self.act = nn.Softplus()

    def forward(self, input):
        out = self.embedder(input)
        
