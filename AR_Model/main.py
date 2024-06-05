import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data.dataloader import ARDateDataset, ARTestDataset
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
test_dataset = ARTestDataset(df, config, size=2000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
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
    model.train()
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
        if (step + 1) % 100 == 0:
            print()
            prediction = dataset.tokenizer.batch_decode(model.generate((batch["input"]).to(device), max_length=128))
            gt = dataset.tokenizer.batch_decode((batch["target"]).to(device))
            prediction = [p.replace("<pad>", "").replace("</s>", "").replace("<unk>", "") for p in prediction]
            gt = [g.replace("<pad>", "").replace("</s>", "").replace("<unk>", "") for g in gt]
            prediction = [p.split("<sep>") for p in prediction]
            gt = [g.split("<sep>") for g in gt]
            # flatten the list
            prediction = [item for sublist in prediction for item in sublist]
            gt = [item for sublist in gt for item in sublist]
            # calculate recall
            gt_in_prediction = sum([1 for g in gt if g in prediction])
            recall = gt_in_prediction / len(gt)
            print("Recall:", recall)
            # calculate precision
            prediction_in_gt = sum([1 for p in prediction if p in gt])
            precision = prediction_in_gt / len(prediction)
            print("Precision:", precision)
            print("F1 Score:", 2 * (precision * recall) / (precision + recall))
            print("Prediction:", prediction)
            print("Ground truth:", gt)
        step += 1
        if config["wandb"] and step % 100 == 0:
            wandb.log(
                {
                    "train_loss_step": loss.item(),
                    "Train Recall step": recall,
                    "Train Precision step": precision,
                    "Train F1 Score step": 2 * (precision * recall) / (precision + recall)
                }
            )
    print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {train_loss / len(dataloader)}")
    if config["wandb"]:
        wandb.log(
            {
                "train_loss_epoch": train_loss / len(dataloader),
            }
        )
    with torch.no_grad():
        model.eval()
        test_loss = 0
        total_gt_in_prediction = 0
        total_prediction_in_gt = 0
        gt_num = 0
        pred_num = 0
        for batch in test_dataloader:
            outputs = model(
                input_ids=batch["input"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["target"].to(device),
            )
            loss = outputs.loss
            test_loss += loss.item()
            prediction = dataset.tokenizer.batch_decode(model.generate((batch["input"]).to(device), max_length=128))
            gt = dataset.tokenizer.batch_decode((batch["target"]).to(device))
            prediction = [p.replace("<pad>", "").replace("</s>", "").replace("<unk>", "") for p in prediction]
            gt = [g.replace("<pad>", "").replace("</s>", "").replace("<unk>", "") for g in gt]
            prediction = [p.split("<sep>") for p in prediction]
            gt = [g.split("<sep>") for g in gt]
            # flatten the list
            prediction = [item for sublist in prediction for item in sublist]
            gt = [item for sublist in gt for item in sublist]
            # calculate recall
            gt_in_prediction = sum([1 for g in gt if g in prediction])
            # calculate precision
            prediction_in_gt = sum([1 for p in prediction if p in gt])
            total_gt_in_prediction += gt_in_prediction
            total_prediction_in_gt += prediction_in_gt
            gt_num += len(gt)
            pred_num += len(prediction)
        recall = total_gt_in_prediction / gt_num
        precision = total_prediction_in_gt / pred_num
        print(f"Test Recall: {recall}")
        print(f"Test Precision: {precision}")
        print(f"Test F1 Score: {2 * (precision * recall) / (precision + recall)}")
        print(f"Test Loss: {test_loss / len(test_dataloader)}")
        if config["wandb"]:
            wandb.log(
                {
                    "test_loss_epoch": test_loss / len(test_dataloader),
                    "Test Recall epoch": recall,
                    "Test Precision epoch": precision,
                    "Test F1 Score epoch": 2 * (precision * recall) / (precision + recall)
                }
            )
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"AR_Model_epoch{epoch + 1}.pt")
wandb.finish()
print("Training complete!")
