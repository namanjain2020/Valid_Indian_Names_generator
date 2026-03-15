import torch
import torch.nn as nn
import torch.optim as optim
from rnn_model import VanillaRNN
from attention_rnn import AttentionRNN
from bilstm_model import BiLSTM

with open("TrainingNames.txt") as f:
     names = [line.strip().lower() for line in f]

names = ["<" + name + ">" for name in names]

chars = sorted(list(set("".join(names))))
char2idx = {c:i for i,c in enumerate(chars)}
idx2char = {i:c for c,i in char2idx.items()}
vocabulary_size = len(chars)

def encoding(name):
    return [char2idx[c] for c in name]

MODEL = "attention"

if MODEL == "rnn":
    model = VanillaRNN(vocabulary_size)

elif MODEL == "bilstm":
    model = BiLSTM(vocabulary_size)

elif MODEL == "attention":
    model = AttentionRNN(vocabulary_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.002)
epochs = 25

model.train()

for epoch in range(epochs):
    total_loss = 0
    for name in names:
        sequence = encoding(name)
        x = torch.tensor(sequence[:-1]).unsqueeze(0)
        y = torch.tensor(sequence[1:]).unsqueeze(0)

        optimizer.zero_grad()
        output, _ = model(x)
        loss = criterion(output.view(-1,vocabulary_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch number : {epoch + 1} and Loss : {total_loss}")

torch.save(model.state_dict(), MODEL + "_model.pth")
print("Your model has been saved successfully")