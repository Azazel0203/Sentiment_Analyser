

from transformers import AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np


warnings.filterwarnings("ignore")
model_untrained = AutoModelForSequenceClassification.from_pretrained(os.path.join("artifact", "downloaded_model"))
model_unquantized = AutoModelForSequenceClassification.from_pretrained(os.path.join("artifact", "trained_model"))

model_quantized = torch.jit.load(os.path.join("artifact", "quantized_model.pt")).to(device)

models = {
    "Untrained_Model": model_untrained,
    "UnQuantized_Model": model_unquantized,
    "Quantized_Model": model_quantized
}

# model = AutoModelForSequenceClassification.from_pretrained("trained_model").to(device)
# model_type = "trained"
# if model_type == "quantized_model.pt":
#     print(model_type)
#     device = "cpu"
#     model = torch.jit.load("quantized_model.pt").to(device)
# else:
#     print(model_type)
#     device = "cuda"
#     model = AutoModelForSequenceClassification.from_pretrained("trained_model").to(device)

# model_size_bytes = os.path.getsize("trained_model/model.safetensors")
# model_size_mb = model_size_bytes / (1024 * 1024)
# print(f"Model size: {model_size_mb:.2f} MB")
X_test_tokenized = torch.load(os.path.join('artifact', 'X_test_tokenized.pt')).to(device)
Y_test_tensor = torch.load(os.path.join('artifact', 'Y_test_tensor.pt')).to(device)
dataset_val = TensorDataset(X_test_tokenized.input_ids, X_test_tokenized.attention_mask, Y_test_tensor.float())
val_dataloader = DataLoader(dataset_val, batch_size=4, shuffle=False)  



for model_name, model in models.items():    
    print(model_name)
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            try:
                logits = output.logits
            except Exception as e:
                logits = output['logits']
                # print(e)
            probs = torch.sigmoid(logits)
            threshold = 0.5
            binary_preds = (probs >= threshold).float()
            predictions = binary_preds
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())


    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc_roc = roc_auc_score(y_true, y_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC-ROC Score: {auc_roc:.4f}')
    
    num_params = sum(p.numel() for p in model.parameters)
    size = num_params * 4 / (1024**2)
    print(f"Size of the {model_name} -> {size}")

    warnings.resetwarnings()
    print("================================================")