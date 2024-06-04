import torch
import torch.nn as nn
from mup import MuReadout, MuSharedReadout
from .positional_encodings import PositionalEncoding
from .transformer import TransformerDecoder, TransformerEncoder
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
        self.encoder = TransformerEncoder(
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
        self.decoder = TransformerDecoder(
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
