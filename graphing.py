import os

# Create a folder named "graph_stuff" if it does not exist
if not os.path.exists('graph_stuff'):
    os.makedirs('graph_stuff')

# Change the current working directory to "graph_stuff"
os.chdir('graph_stuff')

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer_path = os.path.join("artifact", "tokenizer")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained("sbcBI/sentiment_analysis_model", num_labels = 9, ignore_mismatched_sizes=True, problem_type="multi_label_classification")
model.to(device)

# Load training data
X_train_tokenized = torch.load('X_train_tokenized.pt').to(device)
Y_train_tensor = torch.load('Y_train_tensor.pt').to(device)
dataset_train = TensorDataset(X_train_tokenized.input_ids, X_train_tokenized.attention_mask, Y_train_tensor.float())
train_dataloader = DataLoader(dataset_train, batch_size=8, shuffle=True)

# Load test data
X_test_tokenized = torch.load('X_test_tokenized.pt').to(device)
Y_test_tensor = torch.load('Y_test_tensor.pt').to(device)
dataset_val = TensorDataset(X_test_tokenized.input_ids, X_test_tokenized.attention_mask, torch.tensor(Y_test_tensor).float())
val_dataloader = DataLoader(dataset_val, batch_size=4, shuffle=False)

# Define learning rates to try
learning_rates = [5e-5, 1e-4, 2e-4]

for lr in learning_rates:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Training loop
        for input_ids, attention_mask, labels in train_dataloader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Learning Rate: {lr}, Average Training Loss: {avg_train_loss:.4f}')

        # Validation loop
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_dataloader:
                input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = output.logits
                probs = torch.sigmoid(logits)
                threshold = 0.5
                binary_preds = (probs >= threshold).float()
                predictions = binary_preds
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        auc_roc = roc_auc_score(y_true, y_pred)

        print(f'Epoch {epoch + 1}/{num_epochs}, Learning Rate: {lr}, Validation Metrics:')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'AUC-ROC Score: {auc_roc:.4f}')

        # Save metrics
        metrics_file_path = f"graph_stuff/metrics_lr_{lr}_epoch_{epoch+1}.txt"
        with open(metrics_file_path, 'w') as metrics_file:
            metrics_file.write(f'Epoch {epoch + 1}/{num_epochs}, Learning Rate: {lr}, Validation Metrics:\n')
            metrics_file.write(f'Accuracy: {accuracy:.4f}\n')
            metrics_file.write(f'Precision: {precision:.4f}\n')
            metrics_file.write(f'Recall: {recall:.4f}\n')
            metrics_file.write(f'F1 Score: {f1:.4f}\n')
            metrics_file.write(f'AUC-ROC Score: {auc_roc:.4f}\n')

        # Save model
        model_save_dir = f"graph_stuff/trained_model_lr_{lr}_epoch_{epoch+1}"
        os.makedirs(model_save_dir, exist_ok=True)
        model.save_pretrained(model_save_dir)
