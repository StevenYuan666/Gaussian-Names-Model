from model import *
import wandb
from utils import parse_args
# from config import defaults_hf as config
from config import defaults_customLM as config
import json
import pandas as pd
from data.dataloader import DateDataset, TestDateDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from sinkhorn import SinkhornSolver
from torch.optim.lr_scheduler import CosineAnnealingLR

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
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
dataset = DateDataset(df, config)
test_dataset = TestDateDataset(df, config, size=100)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

if config["wandb"]:
        run_name = (
            f"lr{config['lr']}_wd{config['weight_decay']}"
        )
        wandb.login()
        wandb.init(
            # project=f"(Pretrained-T5) Unified Continuous Diffusion Models with Text Encoder-Decoder",
            project=f"Unified Continuous Diffusion Models with Text Encoder-Decoder",
            # project=f"(No Order) Unified Continuous Diffusion Models with Text Encoder-Decoder",
            config=config,
            name=run_name,
        )

# model = GaussianNamesModel(config, dataset.tokenizer)
text_model = TextModule(config)
model = TransformerDenoiseModel(feature_size=config["d_model"])
betas = linear_beta_schedule(T=1000)  # Define this function as provided in earlier steps
text_model.to(config["device"])
model.to(config["device"])
model.double()
# model.load_state_dict(torch.load("./saved/Gaussian_Names_Model_epoch50.pt"))
loss_fn = torch.nn.CrossEntropyLoss()
# loss_fn = SinkhornSolver(epsilon=0.1)
optimizer = torch.optim.Adam(nn.ModuleList([text_model, model]).parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
# scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
# Train the model
step = 0
best_test_loss = float("inf")
for epoch in range(config["epochs"]):
    model.train()
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
        noise_pred = model(xt, t)  # (batch_size, num_of_properties, d_model)
        # Convert xt back to x0_pred using the noise_pred
        x0_pred = revert_xt_to_x0(xt, noise_pred, betas, t)

        # Decode the x0_pred back to text
        prediction = []
        loss = 0
        for j in range(x0_pred.shape[1]):
            target = text_model._shift_right(x[:, j, :])  # (batch_size, max_len)
            output = text_model.decoder(target, x0[:, j].unsqueeze(1))  # (batch_size, max_len, vocab_size)
            output = output.unsqueeze(1)
            prediction.append(output)
        prediction = torch.cat(prediction, dim=1)  # (batch_size, num_of_properties, max_len, vocab_size)
        # print("predcition shape:", prediction.shape)
        # print("x shape:", x.shape)
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
            print(f"Training Epoch: {epoch + 1}, Step: {step}")
            print("Prediction:", dataset.tokenizer.batch_decode(torch.argmax(prediction[0], dim=-1), skip_special_tokens=True))
            print("Ground truth:", dataset.tokenizer.batch_decode(x[0], skip_special_tokens=True))
        step += 1
    print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {train_loss / len(dataloader)}")
    # scheduler.step()
    # test_loss = 0
    # model.eval()
    # for d in test_dataloader:
    #     x = d["target"]
    #     mask = d["mask"]
    #     mask = mask.squeeze(0)
    #     x = x.to(config["device"])  # (batch_size, num_of_properties, max_len)
    #     # Encode the text
    #     x_label = torch.zeros(x.shape[0], x.shape[1], config["d_model"]).to(config["device"])
    #     for j in range(x.shape[1]):
    #         output = text_model.encoder(
    #             x[:, j, :], padding_mask=(x[:, j, :] != 0).float()
    #         )  # (batch_size, max_len, d_model)
    #         if config["text_model"] == "custom":
    #             x_label[:, j] = output[:, 0]  # (batch_size, d_model)
    #         else:
    #             x_label[:, j] = output.last_hidden_state[:, 0]  # (batch_size, d_model)
    #
    #     # Initialize the random prior
    #     generated = torch.randn_like(x_label)
    #     generated = generated.to(device)
    #
    #     # Denoise
    #     for t in range(999, -1, -1):
    #         noise_pred = model(generated, t)  # (batch_size, num_of_properties, d_model)
    #         # denoise the generated to x_t-1
    #         x_t_1 = denoise_one_step(generated, noise_pred, betas, t)
    #         # Replace the masked tokens with the ground truth noisy observation
    #         ground_truth, _ = q_sample(x_label, t-1, betas)
    #
    #
    #     # Decode generated back to text
    #     generated = generated.float()
    #     prediction = []
    #     for j in range(generated.shape[1]):
    #         target = text_model._shift_right(x[:, j, :])  # (batch_size, max_len)
    #         if config["text_model"] == "custom":
    #             output = text_model.decoder(
    #                 input_ids=target, encoder_hidden_states=generated[:, j, :].unsqueeze(1)
    #             )
    #         else:
    #             output = text_model.decoder(input_ids=target)  # (batch_size, max_len, vocab_size)
    #         output = output.unsqueeze(1)
    #         prediction.append(output)
    #     prediction = torch.cat(prediction, dim=1)  # (batch_size, num_of_properties, max_len, vocab_size)
    #     # only calculate loss on the masked tokens
    #     mask = mask.unsqueeze(0)
    #     mask_prediction = prediction[mask == 0].view(-1, prediction.size(-1))
    #     mask_x = x[mask == 0].view(-1)
    #     loss = loss_fn(mask_prediction, mask_x)
    #     test_loss += loss.item()
    # print(f"Test Loss: {test_loss / len(test_dataloader)}")
    if config["wandb"]:
        wandb.log(
            {
                "train_loss_epoch": train_loss / len(dataloader),
                # "test_loss_epoch": test_loss / len(test_dataloader),
            }
        )
    # if test_loss < best_test_loss:
    #     best_test_loss = test_loss
    #     print("Saving model... at epoch", epoch + 1)
    #     torch.save(model.state_dict(), f"./saved/denoise_best_model.pt")
    #     torch.save(text_model.state_dict(), f"./saved/text_best_model.pt")
wandb.finish()
print("Training complete!")
