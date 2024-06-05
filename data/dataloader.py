import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from GNM.tokenizer import GPT2Tokenizer
from transformers import T5Tokenizer
import random


# Custom Dataset
class DateDataset(Dataset):
    def __init__(self, df, config, max_len=15, size=None):
        if size:
            df = df.sample(size)
        self.df = df
        self.config = config
        if config["tokenizer"] == "gpt2":
            tokenizer = GPT2Tokenizer()
        elif config["tokenizer"] == "t5-small":
            tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
            tokenizer.add_tokens(["<cls>"])
            tokenizer.cls_token = "<cls>"
        self.tokenizer = tokenizer
        self.max_len = max_len
        config.update(
            {
                "num_fields": len(df.columns),
                "vocab_size": len(tokenizer),
                "categorical_pad_token_id": 0,
            }
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        target = []
        for column_value in row:
            if self.config["tokenizer"] == "gpt2":
                encoded = self.tokenizer.encode(
                    self.tokenizer.cls_token + " " + column_value,
                )
            elif self.config["tokenizer"] == "t5-small":
                encoded = self.tokenizer.encode(
                    self.tokenizer.cls_token + " " + column_value,
                )
            if len(encoded) < self.max_len:
                encoded += [self.tokenizer.pad_token_id] * (self.max_len - len(encoded))
            else:
                encoded = encoded[: self.max_len]
            target.append(encoded)
        return torch.tensor(target)


class ARDateDataset(Dataset):
    def __init__(self, df, config, max_len=128, size=None):
        if size:
            df = df.sample(size)
        self.df = df
        self.config = config
        if config["tokenizer"] == "gpt2":
            tokenizer = GPT2Tokenizer()
        elif config["tokenizer"] == "t5-small":
            tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
            tokenizer.add_tokens(["<cls>", "<sep>"])
            tokenizer.cls_token = "<cls>"
            tokenizer.sep_token = "<sep>"
        self.tokenizer = tokenizer
        self.max_len = max_len
        config.update(
            {
                "num_fields": len(df.columns),
                "vocab_size": len(tokenizer),
                "categorical_pad_token_id": 0,
            }
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # randomly choose a subset from the row
        row = row.dropna().tolist()
        the_one = random.sample(row, random.randint(1, len(row)-1))
        others = [x for x in row if x not in the_one]
        input = self.tokenizer(self.tokenizer.sep_token.join(the_one), max_length=15, truncation=True, padding="max_length", return_tensors="pt")
        target = self.tokenizer(self.tokenizer.sep_token.join(others), max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")

        return {"input": input.input_ids.squeeze(), "target": target.input_ids.squeeze(), "attention_mask": input.attention_mask.squeeze(), "target_attention_mask": target.attention_mask.squeeze()}


class ARTestDataset(ARDateDataset):
    def __init__(self, df, config, max_len=128, size=None):
        super().__init__(df, config, max_len, size)
        self.df = df[:size]
    def __getitem__(self, index):
        row = self.df.iloc[index]
        row = row.dropna().tolist()
        random_list = []
        for _ in range(len(row)):
            random_list.append(random.randint(0, 1))
        the_one = [x for i, x in enumerate(row) if random_list[i]]
        others = [x for i, x in enumerate(row) if not random_list[i]]
        input = self.tokenizer(self.tokenizer.sep_token.join(the_one), max_length=15, truncation=True, padding="max_length", return_tensors="pt")
        target = self.tokenizer(self.tokenizer.sep_token.join(others), max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")

        return {"input": input.input_ids.squeeze(), "target": target.input_ids.squeeze(), "attention_mask": input.attention_mask.squeeze(), "target_attention_mask": target.attention_mask.squeeze()}


if __name__ == "__main__":
    # Load JSON data
    with open("date_dataset_no_order.json", "r") as file:
        data = json.load(file)

    # Convert JSON data to DataFrame
    df = pd.DataFrame.from_dict(data, orient="index")

    from GNM.config import defaults_hf as config
    from GNM.utils import parse_args

    config = parse_args(config)

    # Create dataset and dataloader
    dataset = ARDateDataset(df, config)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Test the DataLoader
    for x in dataloader:
        print(x["input"].shape, x["target"].shape, x["attention_mask"].shape, x["target_attention_mask"].shape)
        break
