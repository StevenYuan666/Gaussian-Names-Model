import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data.dataloader import ARDateDataset
from GNM.utils import parse_args
from GNM.model import set_seed
from GNM.config import defaults_hf as config
from transformers import T5ForConditionalGeneration, T5Tokenizer
import wandb
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)
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
dataset = ARDateDataset(df, config)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
if config["wandb"]:
        run_name = (
            f"lr{config['lr']}_wd{config['weight_decay']}"
        )
        wandb.login()
        wandb.init(
            # project=f"(Pretrained-T5) Unified Continuous Diffusion Models with Text Encoder-Decoder",
            # project=f"Unified Continuous Diffusion Models with Text Encoder-Decoder",
            project=f"(No Order) Autoregressive Baseline",
            config=config,
            name=run_name,
        )
# Train the model
step = 0
for epoch in range(config["epochs"]):
    loop = tqdm(dataloader, leave=True)
    train_loss = 0
    for batch in loop:
        outputs = model(
            input_ids=batch["input"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["target"].to(device),
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
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
            print("Prediction:", dataset.tokenizer.batch_decode(model.generate((batch["input"]).to(device), max_length=128), skip_special_tokens=True))
            print("Ground truth:", dataset.tokenizer.batch_decode((batch["target"]).to(device), skip_special_tokens=True))
        step += 1
    print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {train_loss / len(dataloader)}")
    if config["wandb"]:
        wandb.log(
            {
                "train_loss_epoch": train_loss / len(dataloader),
            }
        )
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"AR_Model_epoch{epoch + 1}.pt")
wandb.finish()
print("Training complete!")
