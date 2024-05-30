import tqdm
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

model = GaussianNamesModel(config)
model.to(config["device"])
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(GaussianNamesModel.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
# Train the model
step = 0
for epoch in range(config["epochs"]):
    loop = tqdm(dataloader, leave=True)
    train_loss = 0
    for x in loop:
        x = x.to(config["device"])  # (batch_size, num_of_properties, max_len)
        prediction = model(x)
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
        torch.save(model.state_dict(), f"Gaussian_Names_Model_epoch{epoch + 1}.pt")
wandb.finish()
print("Training complete!")
