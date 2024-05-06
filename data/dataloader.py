import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# Custom Dataset
class DateDataset(Dataset):
    def __init__(self, df, tokenizer, bert_model, max_len=10, size=None):
        if size:
            df = df.sample(size)
        self.df = df
        self.tokenizer = tokenizer
        self.bert_model = bert_model.to(device)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        row_embeddings = []
        for column_value in row:
            encoded = self.tokenizer.encode_plus(
                self.tokenizer.cls_token + column_value,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
                add_special_tokens=True,
            ).to(device)
            with torch.no_grad():
                bert_output = self.bert_model(**encoded)
            cls_embedding = bert_output.last_hidden_state[:, 0, :]
            row_embeddings.append(cls_embedding.squeeze())
        return torch.stack(row_embeddings).to("cpu")


if __name__ == "__main__":
    # Load JSON data
    with open("date_dataset.json", "r") as file:
        data = json.load(file)

    # Convert JSON data to DataFrame
    df = pd.DataFrame.from_dict(data, orient="index")

    # Use BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create dataset and dataloader
    model = BertModel.from_pretrained("bert-base-uncased")
    dataset = DateDataset(df, tokenizer, model, max_len=10)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Test the DataLoader
    for batch in dataloader:
        print("Batch Shape:", batch.shape)
        break
