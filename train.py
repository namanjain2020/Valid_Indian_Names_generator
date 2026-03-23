import torch
import torch.nn as nn
import torch.optim as optim
from rnn_model import VanillaRNN
from attention_rnn import AttentionRNN
from bilstm_model import BiLSTM

with open("TrainingNames.txt") as f:
     names = [line.strip().lower() for line in f]     # remove spaces + lowercase

names = ["<" + name + ">" for name in names]          # Add start "<" and end ">" tokens to each name

chars = sorted(list(set("".join(names))))             # Get all unique characters
char2idx = {c:i for i,c in enumerate(chars)}          # Create character-to-index mappings
idx2char = {i:c for c,i in char2idx.items()}          # Create index-to-character mappings
vocabulary_size = len(chars)                          # total unique characters

def encoding(name):                                   # Convert a name (string) into list of indices
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

model.train()                                         # set model to training mode
for epoch in range(epochs):
    total_loss = 0
    for name in names:
        sequence = encoding(name)                     # convert name to indices
        x = torch.tensor(sequence[:-1]).unsqueeze(0)  # Input sequence (all except last char)
        y = torch.tensor(sequence[1:]).unsqueeze(0)   # Target sequence (all except first char)

        optimizer.zero_grad()                          # clear previous gradients
        output, _ = model(x)                           # Forward pass
        loss = criterion(output.view(-1,vocabulary_size), y.view(-1))   # Reshape output for loss computation
        loss.backward()                                                 # Backpropagation
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch number : {epoch + 1} and Loss : {total_loss}")

torch.save(model.state_dict(), MODEL + "_model.pth")                   # Saving the trained model
print("Your model has been saved successfully")
