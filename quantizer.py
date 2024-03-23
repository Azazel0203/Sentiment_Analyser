import torch
import os
from torch.quantization import quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = AutoModelForSequenceClassification.from_pretrained(os.path.join("artifact", "trained_model"))

quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

input_ids = torch.randint(0, 1000, size=(1, 512), dtype=torch.long)
attention_mask = torch.randint(0, 2, size=(1, 512), dtype=torch.long) 



dummy_input = (input_ids, attention_mask)
traced_model = torch.jit.trace(quantized_model, dummy_input, strict=False)
torch.jit.save(traced_model, os.path.join("artifact", "quantized_model.pt"))


