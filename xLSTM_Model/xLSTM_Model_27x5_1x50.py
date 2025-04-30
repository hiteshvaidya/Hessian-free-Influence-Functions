import math
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader

# Tokenizer and Vocabulary
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for line in data_iter:
        yield tokenizer(line)

vocab = build_vocab_from_iterator(yield_tokens(PennTreebank(split='train')), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Prepare Indexed Data
tokens = [token for line in PennTreebank(split='train') for token in tokenizer(line)]
indexed_data = [vocab[token] for token in tokens]

# Data Preparation
class PTBDataset(Dataset):
    def __init__(self, data, seq_len=30):
        self.seq_len = seq_len
        self.samples = [
            (data[i:i+seq_len], data[i+1:i+1+seq_len])
            for i in range(len(data) - seq_len)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

train_dataset = PTBDataset(indexed_data, seq_len=30)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model Definition
class xLSTMCellPyTorch(nn.Module):
    def __init__(self, input_size, hidden_size, use_layernorm=False):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.hidden_size = hidden_size
        self.W_x = nn.Linear(input_size, 4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size)
        self.layer_norm = nn.LayerNorm(4 * hidden_size) if use_layernorm else nn.Identity()

    def forward(self, x_t, h_prev, c_prev):
        gates = self.W_x(x_t) + self.W_h(h_prev)
        gates = self.layer_norm(gates)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t

class xLSTMPyTorch(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, use_layernorm=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm_cell = xLSTMCellPyTorch(embed_size, hidden_size, use_layernorm)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        h = torch.zeros(batch_size, self.lstm_cell.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.lstm_cell.hidden_size, device=x.device)
        outputs = []
        embedded = self.embedding(x)
        for t in range(seq_len):
            x_t = embedded[:, t, :]
            h, c = self.lstm_cell(x_t, h, c)
            logits = self.output_layer(h)
            outputs.append(logits.unsqueeze(1))
        return torch.cat(outputs, dim=1)

# Evaluation Function
def evaluate_during_training(model, data_loader, vocab_size):
    model.eval()
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total_tokens = 0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = criterion(output.reshape(-1, vocab_size), y_batch.reshape(-1))
            total_loss += loss.item()
            predictions = output.argmax(dim=2)
            correct += (predictions == y_batch).sum().item()
            total_tokens += y_batch.numel()
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total_tokens
    perplexity = math.exp(avg_loss)
    return accuracy, perplexity

# Training Function with Grid Search
def benchmark_pytorch_with_loader(train_loader, vocab_size, embed_size, hidden_size, epochs=10, lr=0.01):
    best_accuracy = 0.0
    best_model_state = None
    best_metrics = {}

    model = xLSTMPyTorch(vocab_size, embed_size, hidden_size, use_layernorm=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = criterion(output.reshape(-1, vocab_size), y_batch.reshape(-1))
            optimizer.zero_grad()
            loss.backward(create_graph=True)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        accuracy, perplexity = evaluate_during_training(model, train_loader, vocab_size)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | Perplexity: {perplexity:.2f}", flush=True)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict()
            best_metrics = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "accuracy": accuracy,
                "perplexity": perplexity,
                "embed_size": embed_size,
                "hidden_size": hidden_size,
                "learning_rate": lr
            }
            filename = f"best_xlstm_embed{embed_size}_hidden{hidden_size}_lr{lr}.pth"
            torch.save(best_model_state, filename)
            print(f"## Saved best model so far to: {filename}")

    training_time = time.time() - start_time

    print("\n## Best Model Performance in This Run")
    print(f"Epoch: {best_metrics['epoch']}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Perplexity: {best_metrics['perplexity']:.2f}")
    print(f"Params: embed_size={best_metrics['embed_size']}, hidden_size={best_metrics['hidden_size']}, lr={best_metrics['learning_rate']}")

    return model, training_time, best_metrics

def evaluate_model(model, data_loader, vocab_size):
    model.eval()
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total_tokens = 0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = criterion(output.reshape(-1, vocab_size), y_batch.reshape(-1))
            total_loss += loss.item()
            predictions = output.argmax(dim=2)
            correct += (predictions == y_batch).sum().item()
            total_tokens += y_batch.numel()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total_tokens
    perplexity = math.exp(avg_loss)

    print(f"Evaluation - Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | Perplexity: {perplexity:.2f}")
    return avg_loss, accuracy, perplexity

# Main Grid Search Logic
vocab_size = len(vocab)
embed_sizes = [64, 128, 256]
hidden_sizes = [128, 256, 512]
learning_rates = [0.001, 0.005, 0.01]
overall_best_accuracy = 0.0
overall_best_config = {}
overall_best_model_name = ""

print("\n## Starting Grid Search...\n")
for embed in embed_sizes:
    for hidden in hidden_sizes:
        for lr in learning_rates:
            print(f"\n🔍 Testing config: embed_size={embed}, hidden_size={hidden}, lr={lr}")
            model, training_time, best_metrics = benchmark_pytorch_with_loader(
                train_loader=train_loader,
                vocab_size=vocab_size,
                embed_size=embed,
                hidden_size=hidden,
                epochs=5,
                lr=lr
            )
            if best_metrics["accuracy"] > overall_best_accuracy:
                overall_best_accuracy = best_metrics["accuracy"]
                overall_best_config = best_metrics
                overall_best_model_name = f"best_xlstm_embed{embed}_hidden{hidden}_lr{lr}.pth"

print("\n## Best Overall Configuration:")
print(f"Model file: {overall_best_model_name}")
print(f"Epoch: {overall_best_config['epoch']}")
print(f"Accuracy: {overall_best_config['accuracy']:.4f}")
print(f"Perplexity: {overall_best_config['perplexity']:.2f}")
print(f"Params: embed_size={overall_best_config['embed_size']}, hidden_size={overall_best_config['hidden_size']}, lr={overall_best_config['learning_rate']}")

# Continue training the best overall model for 50 more epochs
print("\n## Continuing training of best model for 50 more epochs...")

# Reload best model
best_model = xLSTMPyTorch(
    vocab_size=overall_best_config['embed_size'], 
    embed_size=overall_best_config['embed_size'],
    hidden_size=overall_best_config['hidden_size'],
    use_layernorm=True
)
best_model.load_state_dict(torch.load(overall_best_model_name))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model.to(device)

# Optimizer and loss for continued training
optimizer = torch.optim.Adam(best_model.parameters(), lr=overall_best_config["learning_rate"])
criterion = nn.CrossEntropyLoss()

for epoch in range(1, 51):
    best_model.train()
    total_loss = 0

    for x_batch, y_batch in tqdm(train_loader, desc=f"Continued Epoch {epoch}"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = best_model(x_batch)
        loss = criterion(output.reshape(-1, vocab_size), y_batch.reshape(-1))

        optimizer.zero_grad()
        loss.backward(create_graph=True)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    acc, ppl = evaluate_during_training(best_model, train_loader, vocab_size)
    print(f"## Epoch {epoch} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | Perplexity: {ppl:.2f}")

# Save the final model
torch.save(best_model.state_dict(), "final_continued_model.pth")
print("## Final model saved as: final_continued_model.pth")
