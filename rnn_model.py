import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNN(nn.Module):
    def __init__(self, vocabulary_size, hidden_size = 256):
        super().__init__()
        self.hidden_size = hidden_size                                                               #  hidden_size : size of hidden state vector
        self.vocabulary_size = vocabulary_size                                                       # vocabulary_size : number of unique characters

        self.Wxh = nn.Parameter(torch.randn(vocabulary_size, hidden_size) * 0.01)                    # Input -> Hidden weights
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)                        # Hidden -> Hidden weights
        self.Why = nn.Parameter(torch.randn(hidden_size, vocabulary_size) * 0.01)                    # Hidden -> Output weights

        self.bh = nn.Parameter(torch.zeros(hidden_size))                                             # Bias terms
        self.by = nn.Parameter(torch.zeros(vocabulary_size))

    def forward(self, x, h=None):
        batch_size, seq_length = x.shape
        if h is None:                                                                                # x is a tensor of shape (batch_size, seq_length)
           h = torch.zeros(batch_size, self.hidden_size)                                             # Initializing hidden state
        outputs = []

        for t in range(seq_length):
            xt = x[:,t]                                                                              # Index of current character
            xt = nn.functional.one_hot(xt, num_classes=self.vocabulary_size).float()                 # conversion to one-hot vector
            h = torch.tanh(xt @ self.Wxh + h @ self.Whh + self.bh)                                   # hidden state update

            y = h @ self.Why + self.by                                                               # output layer
            outputs.append(y.unsqueeze(1))

        outputs = torch.cat(outputs, dim = 1)
        return outputs, h