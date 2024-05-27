import torch
import tiktoken


# GPT2 Tokenizer
class GPT2Tokenizer:
    def __init__(self) -> None:
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<pad>"
        self.cls_token = "<cls>"
        base = tiktoken.get_encoding("gpt2")
        base._mergeable_ranks[b"!"] = base.n_vocab
        self.enc = tiktoken.Encoding(
            # If you're changing the set of special tokens, make sure to use a different name
            # It should be clear from the name what behaviour to expect.
            name="cl100k_im",
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens={
                **base._special_tokens,
                self.pad_token: 0,
            },
        )

        self.pad_token_id = self.enc.encode(
            self.pad_token, allowed_special={self.pad_token}
        )[0]
        self.eos_token_id = self.enc.encode(
            self.eos_token, allowed_special={self.eos_token}
        )[0]
        self.cls_token_id = self.enc.encode(
            self.cls_token, allowed_special={self.cls_token}
        )[0]

    def encode(self, text: str, allowed_special=set(), disallowed_special="all"):
        return self.enc.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )

    def decode(self, tokens) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.enc.decode(tokens)

    def batch_decode(self, sequence, skip_special_tokens=True) -> list:
        if isinstance(sequence, int):
            sequence = [sequence]
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist()
        if isinstance(sequence[0], int):
            sequence = [sequence]
        if skip_special_tokens:
            stripped = []
            for s in sequence:
                stripped.append(
                    [t for t in s if t not in [self.pad_token_id, self.eos_token_id]]
                )
            sequence = stripped
        return self.enc.decode_batch(sequence)  # type: ignore

    def __call__(self, sequences, return_tensors="pt", padding=True):
        if isinstance(sequences, str):
            sequences = [sequences]
        sequences = self.enc.encode_batch(sequences, allowed_special={self.pad_token})
        sequences = [seq + [self.eos_token_id] for seq in sequences]
        attention_mask = [
            [token != self.pad_token_id for token in seq] for seq in sequences
        ]
        if return_tensors == "pt":
            sequences = [torch.tensor(t) for t in sequences]
            attention_mask = [torch.tensor(t) for t in attention_mask]
            if padding:
                sequences = torch.nn.utils.rnn.pad_sequence(
                    sequences,
                    batch_first=True,
                    padding_value=self.pad_token_id,
                )
                attention_mask = torch.nn.utils.rnn.pad_sequence(
                    attention_mask,
                    batch_first=True,
                    padding_value=0,
                )
        return_dict = {
            "input_ids": sequences,
            "attention_mask": attention_mask,
        }
        return return_dict

    def __len__(self):
        return self.enc.n_vocab
