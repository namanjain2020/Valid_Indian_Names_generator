import torch
import torch.nn.functional as F
from rnn_model import VanillaRNN
from bilstm_model import BiLSTM
from attention_rnn import AttentionRNN

with open("TrainingNames.txt") as f:
    train_names = [line.strip().lower() for line in f]

train_set = set(train_names)
names = ["<"+n+">" for n in train_names]
chars = sorted(list(set("".join(names))))

char2idx = {c:i for i,c in enumerate(chars)}
idx2char = {i:c for c,i in char2idx.items()}
vocabulary_size = len(chars)

def generate_name(model):
    name = ""
    if isinstance(model, VanillaRNN):
        char = torch.tensor([[char2idx["<"]]])
        hidden = None

        for _ in range(20):
            output, hidden = model(char, hidden)
            probs = F.softmax(output[0,-1] / 0.9, dim=0)
            idx = torch.multinomial(probs,1).item()
            c = idx2char[idx]

            if c == ">":
                break

            name += c
            char = torch.tensor([[idx]])

    else:
        sequence = [char2idx["<"]]
        for _ in range(15):
            x = torch.tensor([sequence])
            output, _ = model(x)
            probs = F.softmax(output[0,-1] / 0.9, dim=0)
            idx = torch.multinomial(probs,1).item()
            c = idx2char[idx]

            if c == ">":
                break

            name += c
            sequence.append(idx)
    return name

def evaluate(model):
    generated = []

    for _ in range(1000):
        generated.append(generate_name(model))

    novel = [n for n in generated if n not in train_set]
    novelty = len(novel) / len(generated)                      # Novelty
    diversity = len(set(generated)) / len(generated)           # Diversity

    return generated[:20], novelty, diversity

models = {
    "RNN": VanillaRNN(vocabulary_size),
    "BiLSTM": BiLSTM(vocabulary_size),
    "Attention": AttentionRNN(vocabulary_size)
}

with open("results.txt", "w") as f:
    for name, model in models.items():
        model.load_state_dict(torch.load(name.lower() + "_model.pth", weights_only=True))
        model.eval()
        samples, novelty, diversity = evaluate(model)
        f.write("================================\n")
        f.write("Model: " + name + "\n\n")
        f.write("Generated Samples:\n")

        for s in samples:
            f.write(s + "\n")

        f.write("\nNovelty Rate: " + str(novelty) + "\n")
        f.write("Diversity: " + str(diversity) + "\n\n")

print("Results saved to results.txt")