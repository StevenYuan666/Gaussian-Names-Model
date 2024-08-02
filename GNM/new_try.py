# Set Seed
import numpy as np
import torch.backends.cudnn as cudnn
import random
import torch


def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


set_seed(123)

# Define Text Encoder Decoder
import torch
import torch.nn as nn
from mup import MuReadout, MuSharedReadout
from positional_encodings import PositionalEncoding
from transformer import TransformerDecoder_custom, TransformerEncoder_custom
from functools import cache


def shift_right(input_ids, inplace=True):
    decoder_start_token_id = 0

    if not inplace:
        shifted_input_ids = input_ids.clone()
    else:
        shifted_input_ids = input_ids

    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id
    return shifted_input_ids


class TextModule(nn.Module):
    def __init__(
            self,
            config: dict,
    ):
        """Module for text fields, includes both an encoder and a decoder.

        Args:
            config (dict): Configuration dictionary. Must contain the following keys:
                - text_model: "custom" or "t5-small"
                - vocab_size: Size of the vocabulary
                - d_model: Dimension of the model
                - dropout: Dropout rate
                - nhead: Number of attention heads
                - num_layers: Number of layers
                - d_ff_mult: Multiplier for the feedforward dimension (d_ff = d_ff_mult * d_model)
                - text_encoder_layers: Number of layers in the encoder
                - text_decoder_layers: Number of layers in the decoder
                - freeze: Whether to freeze the parameters of the T5 model
                - sparse_embedding: Whether to use sparse embeddings

        Raises:
            NotImplementedError: If text_model is not "custom" or "t5-small".

        Returns:
            TextModule
        """
        super().__init__()
        if config["text_model"] == "custom":
            self.input_embedding = nn.Embedding(
                config["vocab_size"],
                config["d_model"],
            )
            self.pe = PositionalEncoding(
                config["d_model"], config["dropout"], max_len=2048
            )
            self.encoder = TextEncoder(
                config,
                config["text_encoder_layers"],
                self.input_embedding,
                self.pe,
            )
            self.decoder = TextDecoder(
                config,
                num_layers=config["text_decoder_layers"],
                embedding=self.input_embedding,
                pe=self.pe,
            )
        elif config["text_model"] == "t5-small":
            from transformers import T5ForConditionalGeneration  # type: ignore

            self.model = T5ForConditionalGeneration.from_pretrained(  # type: ignore
                "t5-small",
            )

            self.encoder = self.model.encoder  # type: ignore

            class LMHeadDecoder(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.decoder = model.decoder
                    self.lm_head = model.lm_head
                    self.scale = model.model_dim ** -0.5

                def forward(self, **kwargs):
                    x = self.decoder(**kwargs).last_hidden_state
                    x = self.lm_head(x * self.scale)
                    return x

            self.decoder = LMHeadDecoder(self.model)  # type: ignore
            self.input_embedding = self.model.get_input_embeddings()  # type: ignore
            if config["freeze"]:
                print("Freezing T5 parameters")
                for param in self.parameters():  # type: ignore
                    param.requires_grad = False
        else:
            raise NotImplementedError

    def _shift_right(self, input_ids, inplace=False):
        return shift_right(input_ids, inplace=inplace)

    def zero_pad(self, pad_token_id):
        self.encoder.embedding.weight.data[pad_token_id] = 0  # type: ignore


class TextEncoder(nn.Module):
    def __init__(self, config, num_layers=None, embedding=None, pe=None, pe_len=None):
        super().__init__()
        if num_layers is None:
            num_layers = config["num_layers"]
        self.encoder = TransformerEncoder_custom(
            d_model=config["d_model"],
            dropout=config["dropout"],
            nhead=config["nhead"],
            dim_feedforward=config["d_model"] * ["d_ff_mult"],
            num_layers=num_layers,
        )
        self.positional_encoding = (
            pe
            if pe is not None
            else PositionalEncoding(
                config["d_model"],
                config["dropout"],
                max_len=pe_len,
            )
        )
        self.embedding = (
            embedding
            if embedding is not None
            else nn.Embedding(
                config["vocab_size"],
                config["d_model"],
            )
        )

        if config["encoder_readout"] == "tied":
            self.readout = LMHead(config, self.embedding)
        elif config["encoder_readout"] == "separate":
            self.readout = LMHead(config)
        elif config["encoder_readout"] == "none":
            self.readout = lambda x: x
        else:
            raise NotImplementedError

        self._zero_pad(config["categorical_pad_token_id"])

    def _zero_pad(self, pad_token_id):
        self.embedding.weight.data[pad_token_id] = 0

    def _shift_right(self, input_ids, inplace=False):
        return shift_right(input_ids, inplace=inplace)

    def _causal_mask_like(self, x):
        @cache
        def cached_call(sz, device):
            return torch.nn.Transformer.generate_square_subsequent_mask(sz, device)

        return cached_call(x.shape[1], x.device)

    def encode(
            self, x, attention_mask=None, padding_mask=None, is_causal=False, shift_right=False
    ):
        """Encode a sequence of tokens.
        Args:
            x (Tensor): Input tokens of shape (batch_size, seq_len).
            attention_mask (Tensor, optional): Square mask of shape (seq_len, seq_len). Defaults to None.
            padding_mask (Tensor, optional): Padding mask of shape (batch_size, seq_len).
                Used for key masking. Defaults to None.
            is_causal (bool, optional): Whether to use a causal mask. This would override the mask
                argument when True. Defaults to False.
            shift_right (bool, optional): Whether to shift the input sequence to the right by one.
                The first token is set to 0. Defaults to False.

        Returns:
            Tensor: Encoded sequence of shape (batch_size, seq_len, d_output). d_output is either
                d_model or vocab_size depending on the readout layer.
        """
        if is_causal:
            if attention_mask is not None:
                raise ValueError("Cannot use both attention_mask and is_causal")
            attention_mask = self._causal_mask_like(x)
        if shift_right:
            x = self._shift_right(x)
            if padding_mask is not None:
                padding_mask = self._shift_right(padding_mask)
            if not is_causal and attention_mask is not None:
                raise NotImplementedError()

        x = self(x, attention_mask, padding_mask, is_causal=is_causal)
        x = self.readout(x)
        # Standardize the output shape
        x = nn.Sigmoid()(x)
        return x

    def forward(
            self,
            src,
            attention_mask=None,
            padding_mask=None,
            is_causal=False,
    ):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        src = self.encoder(
            src,
            attention_mask,
            padding_mask,
            is_causal=is_causal,
        )
        return src


class TextDecoder(nn.Module):
    def __init__(self, config, num_layers=None, embedding=None, pe=None, causal=True):
        super().__init__()
        if num_layers is None:
            num_layers = config["num_layers"]
        self.decoder = TransformerDecoder_custom(
            d_model=config["d_model"],
            dropout=config["dropout"],
            nhead=config["nhead"],
            dim_feedforward=config["d_model"] * 4,
            num_layers=num_layers,
        )
        self.emmbedding = embedding
        self.readout = LMHead(config, embedding)

        if pe is None:
            self.positional_encoding = PositionalEncoding(
                config["d_model"],
                config["dropout"],
            )
        else:
            self.positional_encoding = pe
        self.causal = causal

    def forward(
            self,
            input_ids,
            encoder_hidden_states,
            memory_mask=None,
            attention_mask=None,
            memory_key_padding_mask=None,
    ):
        if self.emmbedding is not None:
            x = self.emmbedding(input_ids)
        else:
            raise NotImplementedError(
                "Need to implement passing embeddings directly to decoder"
            )
        tgt_mask = self._causal_mask(x.shape[1], x.device) if self.causal else None
        x = self.positional_encoding(x)
        x = self.decoder(
            x,
            encoder_hidden_states,
            tgt_mask,
            memory_mask,
            attention_mask,
            memory_key_padding_mask,
        )
        x = self.readout(x)
        return x

    def _causal_mask(self, size, device=None):
        # TODO does caching help here?
        mask = torch.full((size, size), float("-inf"), device=device)
        mask.triu_(diagonal=1)
        return mask


class LMHead(nn.Module):
    def __init__(self, config, embedding=None):
        super().__init__()
        use_mup = config.get("use_mup", False)
        if embedding is not None:
            if use_mup:
                self.linear = MuSharedReadout(embedding.weight, bias=False)
            else:
                self.linear = nn.Linear(config["d_model"], config["vocab_size"])
                self.linear.weight = embedding.weight
        else:
            if use_mup:
                self.linear = MuReadout(config["d_model"], config["vocab_size"])
            else:
                self.linear = nn.Linear(config["d_model"], config["vocab_size"])

    def forward(self, x):
        return self.linear(x)

# Define the Diffusion Model
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TransformerDenoiseModel(nn.Module):
    def __init__(self, feature_size, num_layers=6, nhead=8, dim_feedforward=2048):
        super().__init__()
        self.feature_size = feature_size
        self.positional_encoding = PositionalE(feature_size)
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


class PositionalE(nn.Module):
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
        pos_emb = self.pe[t, :].unsqueeze(1)
        pos_emb = pos_emb.expand(batch_size, feature_size, -1)
        return nn.Sigmoid()(pos_emb)

class Diffusion(nn.Module):
    def __init__(self, model, num_columns, emb_dim, n_times=1000, beta_minmax=[1e-4, 2e-2], device='cuda'):

        super(Diffusion, self).__init__()

        self.n_times = n_times
        self.num_columns = num_columns
        self.emb_dim = emb_dim

        self.model = model

        # define linear variance schedule(betas)
        beta_1, beta_T = beta_minmax
        betas = torch.linspace(start=beta_1, end=beta_T, steps=n_times).to(device)  # follows DDPM paper
        self.sqrt_betas = torch.sqrt(betas)

        # define alpha for forward diffusion kernel
        self.alphas = 1 - betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)

        self.device = device

    def extract(self, a, t, x_shape):
        """
            from lucidrains' implementation
                https://github.com/lucidrains/denoising-diffusion-pytorch/blob/beb2f2d8dd9b4f2bd5be4719f37082fe061ee450/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L376
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def scale_to_minus_one_to_one(self, x):
        # according to the DDPMs paper, normalization seems to be crucial to train reverse process network
        return x * 2 - 1

    def reverse_scale_to_zero_to_one(self, x):
        return (x + 1) * 0.5

    def make_noisy(self, x_zeros, t):
        # perturb x_0 into x_t (i.e., take x_0 samples into forward diffusion kernels)
        epsilon = torch.randn_like(x_zeros).to(self.device)

        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)

        # Let's make noisy sample!: i.e., Forward process with fixed variance schedule
        #      i.e., sqrt(alpha_bar_t) * x_zero + sqrt(1-alpha_bar_t) * epsilon
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar

        return noisy_sample.detach(), epsilon

    def denoise_back_to_x0(self, noisy_sample, predicted_epsilon, t):
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, noisy_sample.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, noisy_sample.shape)

        # denoise at time t, utilizing predicted noise
        x_0 = 1 / sqrt_alpha_bar * (noisy_sample - sqrt_one_minus_alpha_bar * predicted_epsilon)
        return x_0

    def forward(self, x_zeros):
        x_zeros = self.scale_to_minus_one_to_one(x_zeros)

        B, _, _ = x_zeros.shape

        # (1) randomly choose diffusion time-step
        t = torch.randint(low=0, high=self.n_times, size=(B,)).long().to(self.device)

        # (2) forward diffusion process: perturb x_zeros with fixed variance schedule
        perturbed_images, epsilon = self.make_noisy(x_zeros, t)

        # (3) predict epsilon(noise) given perturbed data at diffusion-timestep t.
        pred_epsilon = self.model(perturbed_images, t)

        return perturbed_images, epsilon, pred_epsilon, t

    def denoise_at_t(self, x_t, timestep, t):
        B, _, _ = x_t.shape
        if t > 1:
            z = torch.randn_like(x_t).to(self.device)
        else:
            z = torch.zeros_like(x_t).to(self.device)

        # at inference, we use predicted noise(epsilon) to restore perturbed data sample.
        epsilon_pred = self.model(x_t, timestep)

        alpha = self.extract(self.alphas, timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        sqrt_beta = self.extract(self.sqrt_betas, timestep, x_t.shape)

        # denoise at time t, utilizing predicted noise
        x_t_minus_1 = 1 / sqrt_alpha * (x_t - (1 - alpha) / sqrt_one_minus_alpha_bar * epsilon_pred) + sqrt_beta * z

        return x_t_minus_1.clamp(-1., 1)

    def predict(self, x, mask):
        # start from random noise vector, x_0 (for simplicity, x_T declared as x_t instead of x_T)
        x_t = torch.randn((x.size(0), self.num_columns, self.emb_dim)).to(self.device)
        # Denoise
        for t in range(self.n_times - 1, -1, -1):
            for j in range(5): ## Harmonization
                timestep = torch.tensor([t]).repeat_interleave(x.size(0), dim=0).long().to(self.device)
                x_t_1_unknown = self.denoise_at_t(x_t, timestep, t)
                if t > 0:
                    x_t_1_known, epsilon = self.make_noisy(x, timestep - 1)
                else:
                    x_t_1_known = x
                    epsilon = torch.zeros_like(x).to(self.device)
                x_t_1 = torch.zeros_like(x).to(self.device)
                for i, m in enumerate(mask):
                    if m.item():
                        x_t_1[:, i] = x_t_1_known[:, i]
                    else:
                        x_t_1[:, i] = x_t_1_unknown[:, i]
                if j < 4 and t > 0:
                    # Add noise for one step
                    x_t = self.sqrt_alphas[t] * x_t_1 + self.sqrt_betas[t] * epsilon.to(self.device)
                else:
                    x_t = x_t_1

        x_0 = self.reverse_scale_to_zero_to_one(x_t) # reverse normalization
        # x_0 = x_t
        return x_0

# Import Dataloader
import json
import pandas as pd
from data.dataloader import DateDataset, TestDateDataset
from torch.utils.data import DataLoader
from config import defaults_customLM as config
from utils import parse_args
# Load JSON data
with open("/home/admin1/Documents/Gaussian-Names-Model/data/date_dataset.json", "r") as file:
    data = json.load(file)

# Convert JSON data to DataFrame
df = pd.DataFrame.from_dict(data, orient="index")
config = parse_args(config)
# Create dataset and dataloader
dataset = DateDataset(df, config)
test_dataset = TestDateDataset(df, config, size=5)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Train the model
import wandb
if config["wandb"]:
        run_name = (
            f"lr{config['lr']}_wd{config['weight_decay']}"
        )
        wandb.login()
        wandb.init(
            project=f"New Try",
            config=config,
            name=run_name,
        )
from tqdm import tqdm
model = TransformerDenoiseModel(feature_size=config["d_model"]).to(device)
diffusion = Diffusion(model, num_columns=9, emb_dim=config["d_model"], device=device).to(device)
text_model = TextModule(config).to(device)
print(f"Model has {(sum(p.numel() for p in model.parameters() if p.requires_grad)) + (sum(p.numel() for p in text_model.parameters() if p.requires_grad))} trainable parameters")
optimizer = torch.optim.Adam(nn.ModuleList([text_model, diffusion]).parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
loss_fn = nn.CrossEntropyLoss()
epochs = 50

print("Start training DDPMs...")
model.train()
text_model.train()
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
            x0[:, j] = output[:, 0]  # (batch_size, d_model)
        x0 = x0.to(device)
        # Perturb the data
        x_t, epsilon, pred_epsilon, t = diffusion(x0)
        # Use the predicted noise to predict the original data
        x0_pred = diffusion.denoise_back_to_x0(x_t, pred_epsilon, t)
        # Generate the text based on the reconstructed data
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
        if step % 625 == 0:
            print()
            print(f"Training Epoch: {epoch + 1}, Step: {step}")
            print("Prediction:", dataset.tokenizer.batch_decode(torch.argmax(prediction[0], dim=-1), skip_special_tokens=True))
            print("Ground truth:", dataset.tokenizer.batch_decode(x[0], skip_special_tokens=True))
        step += 1
    print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {train_loss / len(dataloader)}")
    # scheduler.step()
    test_loss = 0
    model.eval()
    test_count = 0
    for d in test_dataloader:
        x = d["target"]
        mask = d["mask"]
        mask = mask.squeeze(0)
        x = x.to(config["device"])  # (batch_size, num_of_properties, max_len)
        # Encode the text
        x_label = torch.zeros(x.shape[0], x.shape[1], config["d_model"]).to(config["device"])
        for j in range(x.shape[1]):
            output = text_model.encoder(
                x[:, j, :], padding_mask=(x[:, j, :] != 0).float()
            )  # (batch_size, max_len, d_model)
            x_label[:, j] = output[:, 0]  # (batch_size, d_model)

        # Denoise
        # Random prior
        sample = diffusion.predict(x_label, mask)

        # Decode generated back to text
        max_len = x.shape[-1]
        temp = 0.0
        sample = sample.float()
        prediction = []
        for j in range(sample.shape[1]):
            # make initial input
            batch_size = sample.shape[0]
            current_input = torch.full(
                (batch_size, 1),
                config["categorical_pad_token_id"],
                device=sample.device,
            )
            for i in range(max_len):
                pred = text_model.decoder(
                    input_ids=current_input,
                    encoder_hidden_states=sample[:, j, :].unsqueeze(1),
                )
                pred = pred[:, -1, :]
                if temp == 0:
                    current_input = torch.cat(
                        (current_input, pred.argmax(-1, keepdim=True)), -1
                    )
                else:
                    probs = torch.softmax(pred / temp, dim=-1)
                    current_input = torch.cat(
                        (current_input, torch.multinomial(probs, 1)), -1
                    )
            output = current_input[:, 1:]
            output = output.unsqueeze(1)
            prediction.append(output)
        prediction = torch.cat(prediction, dim=1)  # (batch_size, num_of_properties, max_len, vocab_size)
        # only calculate loss on the masked tokens
        mask = mask.unsqueeze(0)
        mask_prediction = prediction[mask == 0].view(-1, prediction.size(-1))
        mask_x = x[mask == 0].view(-1)
        # loss = loss_fn(mask_prediction, mask_x)
        # test_loss += loss.item()
        test_count += 1
        if test_count % 1 == 0:
            print()
            print(f"Test Epoch: {epoch + 1}")
            print("Prediction:", dataset.tokenizer.batch_decode(prediction[0], skip_special_tokens=True))
            print("Mask:", mask[0])
            print("Ground truth:", dataset.tokenizer.batch_decode(x[0], skip_special_tokens=True))
    if config["wandb"]:
        wandb.log(
            {
                "train_loss_epoch": train_loss / len(dataloader),
            }
        )
    # if test_loss < best_test_loss:
    #     best_test_loss = test_loss
    #     print("Saving model... at epoch", epoch + 1)
    #     torch.save(model.state_dict(), f"./saved/denoise_best_model.pt")
    #     torch.save(text_model.state_dict(), f"./saved/text_best_model.pt")
wandb.finish()
print("Saving model at the end:")
torch.save(model.state_dict(), f"./saved/denoise_model_new_try.pt")
torch.save(text_model.state_dict(), f"./saved/text_model_new_try.pt")
torch.save(diffusion.state_dict(), f"./saved/diffusion_model_new_try.pt")
print("Successfully saved the model!")
print("Training complete!")
