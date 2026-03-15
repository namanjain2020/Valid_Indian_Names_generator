import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # input gate
        self.Wxi = nn.Linear(input_size, hidden_size)
        self.Whi = nn.Linear(hidden_size, hidden_size)

        # forget gate
        self.Wxf = nn.Linear(input_size, hidden_size)
        self.Whf = nn.Linear(hidden_size, hidden_size)

        # output gate
        self.Wxo = nn.Linear(input_size, hidden_size)
        self.Who = nn.Linear(hidden_size, hidden_size)

        # candidate cell state
        self.Wxc = nn.Linear(input_size, hidden_size)
        self.Whc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h, c):
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h))                  # gate calculations
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        o = torch.sigmoid(self.Wxo(x) + self.Who(h))
        g = torch.tanh(self.Wxc(x) + self.Whc(h))

        c= f*c + i*g                                                  # update cell state
        h = o*torch.tanh(c)                                           # update hidden state

        return h, c


class BiLSTM(nn.Module):
    def __init__(self, vocabulary_size, hidden_size = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocabulary_size = vocabulary_size

        self.forward_cell = LSTMCell(vocabulary_size, hidden_size)    # forward LSTM
        self.backward_cell = LSTMCell(vocabulary_size, hidden_size)   # backward LSTM
        self.fc = nn.Linear(hidden_size * 2, vocabulary_size)         # output layer

    def forward(self, x):
        batch, sequence = x.shape

        x = F.one_hot(x, num_classes=self.vocabulary_size).float()    # convert character indices → one-hot

        hf = torch.zeros(batch, self.hidden_size)                     # initializing forward hidden state
        cf = torch.zeros(batch, self.hidden_size)                     # initializing forward cell state
        hb = torch.zeros(batch, self.hidden_size)                     # initializing backward hidden state
        cb = torch.zeros(batch, self.hidden_size)                     # initializing backward cell state

        forward_outputs = []
        backward_outputs = []

        for t in range(sequence):                                     # forward pass
            hf, cf = self.forward_cell(x[:, t], hf, cf)               # update hidden and cell state
            forward_outputs.append(hf)                                # store hidden state

        for t in reversed(range(sequence)):                           # backward pass
            hb, cb = self.backward_cell(x[:, t], hb, cb)
            backward_outputs.insert(0, hb)                            # insert at beginning to align with forward states

        outputs = []

        for t in range(sequence):                                                     # Combine forward and backward outputs
            combined = torch.cat([forward_outputs[t], backward_outputs[t]], dim=1)    # concatenate hidden states
            y = self.fc(combined)                                                     # project to vocabulary size
            outputs.append(y.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)                                           # convert list to tensor

        return outputs, None