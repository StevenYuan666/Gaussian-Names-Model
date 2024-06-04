import pandas as pd
import json


df = pd.read_csv("date_dataset_set.csv")

data = df.to_dict(orient="index")
with open("date_dataset_no_order.json", "w") as file:
    json.dump(data, file, indent=4)
