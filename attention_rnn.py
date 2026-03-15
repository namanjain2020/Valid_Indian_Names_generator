import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionRNN(nn.Module):
    def __init__(self, vocabulary_size, hidden_size = 256):
        super().__init__()
        self.hidden_size = hidden_size                                            # size of hidden state vector
        self.vocabulary_size = vocabulary_size                                    # number of characters in vocabulary

        self.Wxh = nn.Parameter(torch.randn(vocabulary_size, hidden_size)*0.01)   # input → hidden weight matrix
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size)*0.01)       # hidden → hidden weight matrix.

        self.fc = nn.Linear(hidden_size*2, vocabulary_size)                       # final output layer
        self.bh = nn.Parameter(torch.zeros(hidden_size))                          # bias for hidden layer


    def forward(self, x):
        batch, sequence = x.shape
        h = torch.zeros(batch, self.hidden_size)                                  # initial hidden state
        hidden_states = []                                                        # store all hidden states for attention

        outputs = []                                                              # store outputs

        for t in range(sequence):
            xt = F.one_hot(x[:,t], num_classes=self.vocabulary_size).float()      # convert character index to one-hot vector
            h = torch.tanh(xt @ self.Wxh + h @ self.Whh + self.bh)                # RNN hidden state update
            hidden_states.append(h)                                               # save hidden state
        hidden_states = torch.stack(hidden_states, dim=1)

        for t in range(sequence):
            ht = hidden_states[:, t]                                              # current hidden state
            scores = torch.bmm(hidden_states, ht.unsqueeze(2)).squeeze(2)         # compute attention scores between ht and all hidden states
            weights = torch.softmax(scores, dim=1)                                # normalize scores to get attention weights

            context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)   # context vector
            combined = torch.cat([ht, context], dim=1)                            # combine current state and context

            y = self.fc(combined)                                                 # compute output
            outputs.append(y.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)                                       # concatenate outputs across sequence

        return outputs, None