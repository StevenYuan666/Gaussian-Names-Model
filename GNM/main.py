from model import *
import wandb
from utils import parse_args
# from config import defaults_hf as config
from config import defaults_customLM as config
import json
import pandas as pd
from data.dataloader import DateDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from sinkhorn import SinkhornSolver

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
config = parse_args(config)
set_seed(config["seed"])
print("Seed:", config["seed"])
print("Device:", config["device"])

# Load JSON data
with open("../data/date_dataset_no_order.json", "r") as file:
    data = json.load(file)

# Convert JSON data to DataFrame
df = pd.DataFrame.from_dict(data, orient="index")
config = parse_args(config)
# Create dataset and dataloader
dataset = DateDataset(df, config)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

if config["wandb"]:
        run_name = (
            f"lr{config['lr']}_wd{config['weight_decay']}"
        )
        wandb.login()
        wandb.init(
            # project=f"(Pretrained-T5) Unified Continuous Diffusion Models with Text Encoder-Decoder",
            # project=f"Unified Continuous Diffusion Models with Text Encoder-Decoder",
            project=f"(No Order) Unified Continuous Diffusion Models with Text Encoder-Decoder",
            config=config,
            name=run_name,
        )

model = GaussianNamesModel(config, dataset.tokenizer)
model.to(config["device"])
# loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = SinkhornSolver(epsilon=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
# Train the model
step = 0
for epoch in range(config["epochs"]):
    loop = tqdm(dataloader, leave=True)
    train_loss = 0
    for x in loop:
        x = x.to(config["device"])  # (batch_size, num_of_properties, max_len)
        x = x.long()
        prediction = model(x)
        loss, _ = loss_fn(prediction.view(-1, prediction.size(-1)), x.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_description(f"Epoch [{epoch + 1}/{config['epochs']}]")
        loop.set_postfix(loss=loss.item())
        train_loss += loss.item()
        if config["wandb"] and step % 100 == 0:
            wandb.log(
                {
                    "train_loss_step": loss.item(),
                }
            )
        if step % 100 == 0:
            print()
            print("Prediction:", dataset.tokenizer.batch_decode(torch.argmax(prediction[0], dim=-1), skip_special_tokens=True))
            print("Ground truth:", dataset.tokenizer.batch_decode(x[0], skip_special_tokens=True))
        step += 1
    print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {train_loss / len(dataloader)}")
    if config["wandb"]:
        wandb.log(
            {
                "train_loss_epoch": train_loss / len(dataloader),
            }
        )
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"Gaussian_Names_Model_epoch{epoch + 1}.pt")
wandb.finish()
print("Training complete!")
