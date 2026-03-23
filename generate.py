import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_rnn import AttentionRNN
from bilstm_model import BiLSTM
from rnn_model import VanillaRNN

MODEL = 'rnn'  

with open("TrainingNames.txt") as f:
    names = [line.strip().lower() for line in f]

names = ["<"+name+">" for name in names]                                           # Add start "<" and end ">" tokens
chars = sorted(list(set("".join(names))))                                          # Create character vocabulary

char2idx = {c:i for i,c in enumerate(chars)}                                       # Create character to index mappings
idx2char = {i:c for c,i in char2idx.items()}                                       # Create index to character mappings
vocabulary_size = len(chars)

if MODEL == "rnn":
    model = VanillaRNN(vocabulary_size)

elif MODEL == "bilstm":
    model = BiLSTM(vocabulary_size)

elif MODEL == "attention":
    model = AttentionRNN(vocabulary_size)

model.load_state_dict(torch.load(MODEL + "_model.pth",weights_only=True))          # Load trained weights
model.eval()                                                                       # set model to evaluation mode

def generate_name(max_len=20):
    name = ""
    if MODEL == "rnn":
        char = torch.tensor([[char2idx["<"]]])                                    # start token
        hidden = None

        for _ in range(max_len):
            output, hidden = model(char, hidden)
            probs = F.softmax(output[0,-1]/0.9, dim=0)                            # Apply temperature scaling for randomness
            idx = torch.multinomial(probs,1).item()                               # Sample next character
            c = idx2char[idx]
            
            if c == ">":                                                          # Stop if end token reached
                break

            name += c
            char = torch.tensor([[idx]])                                          # feed predicted char as next input

    elif MODEL == "attention":
         sequence = [char2idx["<"]]                                               
         for _ in range(max_len): 
            x = torch.tensor([sequence])
            output, _ = model(x)
            logits = output[0,-1] / 0.7                                          # Lower temperature → more confident predictions

            k = 9                                                                # Top-k sampling 
            values, indices = torch.topk(logits, k)
            probs = F.softmax(values, dim=0)
            idx = indices[torch.multinomial(probs,1)].item()
            c = idx2char[idx]

            if len(name) == 0 and c == ">":                                      # Avoid empty output
                continue
            if c == ">":
                break

            name += c
            sequence.append(idx)
 
    else:                                                                        # BiLSTM
        prefix = "<a"                                                            # start with 'a' to guide generation
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
