from data.dataloader import *
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_beta_schedule(T=1000):
    betas = np.linspace(0.0001, 0.02, T)  # Linearly increasing betas
    return betas


def q_sample(x_0, t, betas):
    noise = torch.randn_like(x_0)
    alpha_t = 1.0 - betas[t]
    alpha_cumprod_t = torch.tensor(np.cumprod(alpha_t)).to(device)
    x_0 = x_0.to(device)
    noise = noise.to(device)
    x_t = torch.sqrt(alpha_cumprod_t) * x_0 + torch.sqrt(1.0 - alpha_cumprod_t) * noise
    return x_t, noise


# Define the Denoising Model
class TransformerDenoiseModel(nn.Module):
    def __init__(self, feature_size, num_layers=6, nhead=8, dim_feedforward=2048):
        super().__init__()
        self.feature_size = feature_size
        self.positional_encoding = PositionalEncoding(feature_size)
        encoder_layers = TransformerEncoderLayer(feature_size, nhead, dim_feedforward, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(feature_size, feature_size)

    def forward(self, x, t):
        # print(x.size())
        # print(self.positional_encoding(t, x.size(0)).size())
        x = x.to(device)
        x = x + self.positional_encoding(t, x.size(0), x.size(1)).to(device)
        x = self.transformer_encoder(x)
        return self.fc_out(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, t, batch_size, feature_size):
        return self.pe[t, :].expand(batch_size, feature_size, -1)


def train_model(model, dataloader, val_dataloader, betas, T=1000, num_epochs=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = float("inf")
    no_improvement = 0
    for epoch in range(num_epochs):
        train_loss = 0
        loop = tqdm(dataloader, leave=True)
        for x_0 in loop:
            t = np.random.randint(0, T)
            x_t, noise = q_sample(x_0, t, betas)
            x_t = x_t.to(device)
            noise = noise.to(device)
            noise_pred = model(x_t, t)
            loss = (noise_pred - noise).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())
            train_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(dataloader)}")

        val_loss = validate_epoch(model, val_dataloader, betas, T)
        print(f"Val Loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"model_{epoch}.pth")
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement == 5:
                print(f"Early stopping at epoch {epoch}")
                break


def validate_epoch(model, loader, betas, T):
    model.eval()
    losses = []
    with torch.no_grad():
        for x_0 in loader:
            mask = torch.randint(0, 2, (x_0.size(0), x_0.size(1))).bool()  # Random mask
            mask = mask.to(device)
            x_pred = torch.rand_like(x_0).to(device)
            x_0 = x_0.to(device)
            for t in range(T - 1, -1, -1):
                noise_pred = model(x_pred, t)
                x_pred = x_pred - betas[t] * noise_pred
                x_t_1, _ = q_sample(x_0, t - 1, betas)
                x_pred = x_pred * mask.unsqueeze(-1) + x_t_1 * (~mask).unsqueeze(-1)
            loss = (x_pred - x_0).pow(2).mean()
            losses.append(loss.item())
    return np.mean(losses)


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


set_seed(1205)

# Load JSON data
with open("../data/date_dataset.json", "r") as file:
    data = json.load(file)

# Convert JSON data to DataFrame
df = pd.DataFrame.from_dict(data, orient="index")

# Initialize BERT components
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

#  Create dataset and dataloader
dataset = DateDataset(df, tokenizer, bert_model)

val_size = 50
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize and train the model
hidden_size = bert_model.config.hidden_size
model = TransformerDenoiseModel(feature_size=bert_model.config.hidden_size)
model.to(device)
model.double()
betas = linear_beta_schedule(T=1000)  # Define this function as provided in earlier steps
train_model(model, train_loader, val_loader, betas, T=1000, num_epochs=100)
