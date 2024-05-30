from modules import TextModule
from config import defaults_customLM as config
from utils import parse_args
import json
import pandas as pd
from data.dataloader import DateDataset
from torch.utils.data import DataLoader
import torch
from model import *
import wandb


config = parse_args(config)
set_seed(config["seed"])
print("Seed:", config["seed"])
print("Device:", config["device"])

# Load JSON data
with open("../data/date_dataset.json", "r") as file:
    data = json.load(file)

# Convert JSON data to DataFrame
df = pd.DataFrame.from_dict(data, orient="index")
config = parse_args(config)
# Create dataset and dataloader
dataset = DateDataset(df, config, max_len=15)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

if config["wandb"]:
        run_name = (
            f"lr{config['lr']}_wd{config['weight_decay']}"
        )
        wandb.login()
        wandb.init(
            project=f"Unified Continuous Diffusion Models with Text Encoder-Decoder",
            config=config,
            name=run_name,
        )

text_model = TextModule(config)
model = TransformerDenoiseModel(feature_size=config["d_model"])
betas = linear_beta_schedule(T=1000)  # Define this function as provided in earlier steps
text_model.to(config["device"])
model.to(config["device"])
model.double()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(nn.ModuleList([text_model, model]).parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
# Train the model
step = 0
for epoch in range(config["epochs"]):
    loop = tqdm(dataloader, leave=True)
    train_loss = 0
    for x in loop:
        # Encode the text
        x = x.to(config["device"])  # (batch_size, num_of_properties, max_len)
        x0 = torch.zeros(x.shape[0], x.shape[1], config["d_model"]).to(config["device"])
        for j in range(x.shape[1]):
            output = text_model.encoder(
                x[:, j, :], padding_mask=(x[:, j, :] != 0).float()
            )  # (batch_size, max_len, d_model)
            if config["text_model"] == "custom":
                x0[:, j] = output[:, 0]  # (batch_size, d_model)
            else:
                x0[:, j] = output.last_hidden_state[:, 0]  # (batch_size, d_model)


        # Perturb the input
        t = np.random.randint(0, 1000)
        xt, noise = q_sample(x0, t, betas)  # (batch_size, num_of_properties, d_model)
        xt = xt.to(device)
        x0_pred = model(xt, t)  # (batch_size, num_of_properties, d_model)

        # Decode the x0_pred back to text
        prediction = []
        loss = 0
        for j in range(x0_pred.shape[1]):
            target = text_model._shift_right(x[:, j, :])  # (batch_size, max_len)
            output = text_model.decoder(target, x0[:, j].unsqueeze(1))  # (batch_size, max_len, vocab_size)
            output = output.unsqueeze(1)
            prediction.append(output)
        prediction = torch.cat(prediction, dim=1)  # (batch_size, num_of_properties, max_len, vocab_size)

        loss = loss_fn(prediction.view(-1, prediction.size(-1)), x.view(-1))
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
            print("Prediction:", dataset.tokenizer.batch_decode(torch.argmax(prediction[0], dim=-1)))
            print("Ground truth:", dataset.tokenizer.batch_decode(x[0]))
        step += 1
    print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {train_loss / len(dataloader)}")
    if config["wandb"]:
        wandb.log(
            {
                "train_loss_epoch": train_loss / len(dataloader),
            }
        )
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"denoise_model_epoch{epoch + 1}.pt")
        torch.save(text_model.state_dict(), f"text_model_epoch{epoch + 1}.pt")
wandb.finish()
print("Training complete!")
