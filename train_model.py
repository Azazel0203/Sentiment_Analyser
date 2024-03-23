from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer_path = os.path.join("artifact", "tokenizer")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

model = AutoModelForSequenceClassification.from_pretrained(os.path.join("artifact", "downloaded_model"))

X_train_tokenized = torch.load(os.path.join("artifact", 'X_train_tokenized.pt')).to(device)
Y_train_tensor = torch.load(os.path.join("artifact", 'Y_train_tensor.pt')).to(device)
model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
dataset_train = TensorDataset(X_train_tokenized.input_ids, X_train_tokenized.attention_mask, Y_train_tensor.float())
train_dataloader = DataLoader(dataset_train, batch_size=8, shuffle=True)

print("Done loading model")
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    print("Running first epoch")
    for input_ids, attention_mask, labels in train_dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}')

model.save_pretrained(os.path.join("artifact", "trained_model"))



