from data.dataloader import *
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import torch.backends.cudnn as cudnn
import random
from modules import TextModule

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


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


class GaussianNamesModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_model = TextModule(config)
        self.model = TransformerDenoiseModel(feature_size=config["d_model"])
        self.betas = linear_beta_schedule(T=1000)  # Define this function as provided in earlier steps
        self.text_model.double()
        self.model.double()

    def forward(self, x):
        # Encode the text
        x0 = torch.zeros(x.shape[0], x.shape[1], self.config["d_model"]).to(self.config["device"])
        for j in range(x.shape[1]):
            output = self.text_model.encoder(
                x[:, j, :], padding_mask=(x[:, j, :] != 0).float()
            )  # (batch_size, max_len, d_model)
            if self.config["text_model"] == "custom":
                x0[:, j] = output[:, 0]  # (batch_size, d_model)
            else:
                x0[:, j] = output.last_hidden_state[:, 0]  # (batch_size, d_model)

        # Perturb the input
        t = np.random.randint(0, 1000)
        xt, noise = q_sample(x0, t, self.betas)  # (batch_size, num_of_properties, d_model)
        xt = xt.to(device)
        x0_pred = self.model(xt, t)  # (batch_size, num_of_properties, d_model)

        # Decode the x0_pred back to text
        prediction = []
        for j in range(x0_pred.shape[1]):
            target = self.text_model._shift_right(x[:, j, :])  # (batch_size, max_len)
            output = self.text_model.decoder(target, x0[:, j].unsqueeze(1))  # (batch_size, max_len, vocab_size)
            output = output.unsqueeze(1)
            prediction.append(output)
        prediction = torch.cat(prediction, dim=1)  # (batch_size, num_of_properties, max_len, vocab_size)
        return prediction
