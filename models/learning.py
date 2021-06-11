from torch import nn

class ContextRNN(nn.Module):
    def __init__(self):
        super(ContextRNN, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)


    def forward(self):
        self.gru()

model = ContextRNN
model()

