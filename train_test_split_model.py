import json
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


file_path1 = os.path.join("artifact", "inputs.json")
file_path2 = os.path.join("artifact", "outputs.json")


with open(file_path1, "r") as f:
    X = json.load(f)


with open(file_path2, "r") as f:
    Y = json.load(f)


model = AutoModelForSequenceClassification.from_pretrained("sbcBI/sentiment_analysis_model", num_labels = 9, ignore_mismatched_sizes=True, problem_type="multi_label_classification")
tokenizer_path = os.path.join("artifact", "tokenizer")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train_tokenized = tokenizer(X_train, padding="max_length", truncation=True, return_tensors="pt")
X_test_tokenized = tokenizer(X_test, padding="max_length", truncation=True, return_tensors="pt")
Y_train_tensor = torch.tensor(Y_train)
Y_test_tensor = torch.tensor(Y_test)


torch.save(X_test_tokenized, os.path.join('artifact', 'X_test_tokenized.pt'))
torch.save(Y_test_tensor, os.path.join('artifact', 'Y_test_tensor.pt'))
torch.save(X_train_tokenized, os.path.join('artifact', 'X_train_tokenized.pt'))
torch.save(Y_train_tensor, os.path.join('artifact', 'Y_train_tensor.pt'))


model.save_pretrained(os.path.join('artifact', "downloaded_model"))


print("Done splitting the data and downloaded the model")