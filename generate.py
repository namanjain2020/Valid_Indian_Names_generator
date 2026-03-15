import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_rnn import AttentionRNN
from bilstm_model import BiLSTM
from rnn_model import VanillaRNN

MODEL = 'rnn'  

with open("TrainingNames.txt") as f:
    names = [line.strip().lower() for line in f]

names = ["<"+name+">" for name in names]
chars = sorted(list(set("".join(names))))

char2idx = {c:i for i,c in enumerate(chars)}
idx2char = {i:c for c,i in char2idx.items()}
vocabulary_size = len(chars)

if MODEL == "rnn":
    model = VanillaRNN(vocabulary_size)

elif MODEL == "bilstm":
    model = BiLSTM(vocabulary_size)

elif MODEL == "attention":
    model = AttentionRNN(vocabulary_size)

model.load_state_dict(torch.load(MODEL + "_model.pth",weights_only=True))
model.eval()

def generate_name(max_len=20):
    name = ""
    if MODEL == "rnn":
        char = torch.tensor([[char2idx["<"]]])
        hidden = None

        for _ in range(max_len):
            output, hidden = model(char, hidden)
            probs = F.softmax(output[0,-1]/0.9, dim=0)
            idx = torch.multinomial(probs,1).item()
            c = idx2char[idx]
            
            if c == ">":
                break

            name += c
            char = torch.tensor([[idx]])

    elif MODEL == "attention":
         sequence = [char2idx["<"]]
         for _ in range(max_len): 
            x = torch.tensor([sequence])
            output, _ = model(x)
            logits = output[0,-1] / 0.7

            k = 9
            values, indices = torch.topk(logits, k)
            probs = F.softmax(values, dim=0)
            idx = indices[torch.multinomial(probs,1)].item()
            c = idx2char[idx]

            if len(name) == 0 and c == ">":
                continue
            if c == ">":
                break

            name += c
            sequence.append(idx)
            
    else:
        prefix = "<a" 
        sequence = [char2idx[c] for c in prefix]
        name = "a"
        for _ in range(max_len):
            x = torch.tensor([sequence])
            output, _ = model(x)
            logits = output[0,-1] / 1.0
            
            k = 8
            values, indices = torch.topk(logits, k)
            probs = F.softmax(values, dim=0)
            idx = indices[torch.multinomial(probs,1)].item()
            c = idx2char[idx]

            if len(name) < 2 and c == ">":
                continue

            if c == ">":
                break

            name += c
            sequence.append(idx)
                
    return name

print("\nGenerated Names\n")

for i in range(20):
    print(generate_name())