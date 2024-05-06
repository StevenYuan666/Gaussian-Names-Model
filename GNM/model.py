import torch.nn as nn
from data.dataloader import *
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the Denoising Model
class DenoisingModel(nn.Module):
    def __init__(self, hidden_size, num_encoder_layers, num_heads, dim_feedforward):
        super(DenoisingModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_encoder_layers
        )

    def forward(self, x, time_step):
        time_embedding = self.time_embedding(time_step, x.size(-1))
        # repeat time embedding for each row in the batch and each column in the table
        time_embedding = time_embedding.repeat(x.size(0), x.size(1), 1).to(device)
        x = x + time_embedding
        x = self.transformer_encoder(x)
        return x

    def time_embedding(self, time_step, hidden_size):
        half_dim = hidden_size // 2
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / half_dim)
        )
        emb = time_step * emb[None, :]
        return torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)


def calculate_alpha_betas(T):
    betas = torch.linspace(0.0001, 0.02, T)
    alphas = 1 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_cumprod


def smooth_l1_loss(epsilon, epsilon_theta, beta):
    difference = epsilon - epsilon_theta
    abs_difference = torch.abs(difference)
    smooth_l1 = torch.where(
        abs_difference < beta,
        0.5 * (difference**2) / beta,
        abs_difference - 0.5 * beta,
    )
    return smooth_l1.mean()


def train_diffusion_model(dataloader, val_loader, model, num_epochs=10, T=1000, beta_L1=1.0, learning_rate=1e-4):
    betas, alphas, alpha_cumprod = calculate_alpha_betas(T)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    model = model.to(device)

    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        for x_0 in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x_0 = x_0.to(device)
            batch_size, num_columns, hidden_size = x_0.size()
            t = torch.randint(1, T + 1, (1, ))  # Uniformly sample time steps
            alpha_t = alpha_cumprod[t - 1].unsqueeze(-1).unsqueeze(-1)
            alpha_t = alpha_t.to(device)
            epsilon = torch.randn(batch_size, num_columns, hidden_size)  # Random noise
            epsilon = epsilon.to(device)
            x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * epsilon  # Diffuse x_0

            epsilon_theta = model(x_t, t.squeeze())  # Predict noise
            loss = smooth_l1_loss(epsilon, epsilon_theta, beta_L1)  # Compute loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} Training Loss: {epoch_loss/len(dataloader)}")

        # Evaluate loss on validation set
        val_loss = evaluate_model(model, val_loader, T=T)
        print(f"Epoch {epoch+1} Validation Loss: {val_loss}")

        # Update best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

    return best_model


def evaluate_model(model, dataloader, T=1000):
    model.eval()
    total_loss = 0
    count = 0
    for x_0 in dataloader:
        batch_size, num_columns, hidden_size = x_0.size()
        mask = torch.randint(0, 2, (batch_size, num_columns)).bool()  # Random mask
        masked_x = x_0.clone()
        # Masked values are set to random values between -1 and 1
        masked_x[mask] = 2 * torch.rand_like(masked_x[mask]) - 1
        masked_x = masked_x.to(device)
        mask = mask.to(device)
        reconstructed_x = infer_diffusion_model(model, masked_x, mask, T=T)
        reconstructed_x = reconstructed_x.to("cpu")
        mask = mask.to("cpu")
        loss = torch.nn.functional.mse_loss(reconstructed_x[mask], x_0[mask])
        total_loss += loss.item()
        count += 1
    return total_loss / count if count != 0 else float("inf")


def infer_diffusion_model(model, x, mask, T=1000, J=5):
    betas, alphas, alpha_cumprod = calculate_alpha_betas(T)
    batch_size, num_columns, hidden_size = x.size()

    for t in range(T, 0, -1):
        for j in range(1, J + 1):
            epsilon = torch.randn(batch_size, num_columns, hidden_size) if t > 1 else torch.zeros(batch_size, num_columns, hidden_size)
            epsilon = epsilon.to(device)
            alpha = alpha_cumprod[t - 1]
            alpha = alpha.to(device)
            # print(x)
            # Calculate x_known_{t-1}
            x_known_t_1 = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * epsilon
            # print(x_known_t_1)
            # Calculate x_unknown_{t-1}
            x_unknown_t_1 = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha) * model(x, torch.tensor([t])))
            # print(x_unknown_t_1)
            # exit()
            # Combine known and unknown regions based on the mask, for known regions use x_known_{t-1} and for unknown regions use x_unknown_{t-1}
            x = x_known_t_1 * mask.unsqueeze(-1) + x_unknown_t_1 * (~mask).unsqueeze(-1)

            if j < J and t > 1:
                x = torch.sqrt(alpha_cumprod[t - 2]) * x + torch.sqrt(1 - alpha_cumprod[t - 2]) * epsilon

    return x


# Load JSON data
with open("../data/date_dataset.json", "r") as file:
    data = json.load(file)

# Convert JSON data to DataFrame
df = pd.DataFrame.from_dict(data, orient="index")

# Initialize BERT components
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

#  Create dataset and dataloader
dataset = DateDataset(df, tokenizer, bert_model, size=10)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize and train the model
hidden_size = bert_model.config.hidden_size
model = DenoisingModel(hidden_size=hidden_size, num_encoder_layers=2, num_heads=4, dim_feedforward=256)

model = train_diffusion_model(train_loader, val_loader, model, num_epochs=1, T=1000, beta_L1=1.0, learning_rate=1e-4)
