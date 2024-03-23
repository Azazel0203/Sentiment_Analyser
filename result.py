import time
start_time = time.time()
print("Getting the required nltk libraries")
from transformers import AutoTokenizer
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import nltk
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

import re
execution_time = time.time() - start_time
print(f"Execution time: {execution_time} seconds")
start_time = time.time()


print ("Loading the Model and tokenizer")
device = "cpu"
model = torch.jit.load("model\quantized_model.pt").to(device)
tokenizer = AutoTokenizer.from_pretrained("sbcBI/sentiment_analysis_model")

num_params = sum(p.numel() for p in model.parameters())
model_file_size = os.path.getsize("model\quantized_model.pt")

model_file_size_mb = model_file_size / (1024 * 1024)  


print(f"Model file size: {model_file_size_mb: .4f} MB")
print(f"Number of parameters: {num_params}")

execution_time = time.time() - start_time
print(f"Execution time: {execution_time} seconds")
start_time = time.time()

print("Reading the input 'text.txt' file")

inputs = []
with open('text.txt', 'r') as file:
    content = file.read()
paragraphs = re.findall(r'\{(.*?)\}', content, re.DOTALL)
for paragraph in paragraphs:
    inputs.append(paragraph.strip())
    
def clean_text(data):
  result = []
  for text in data:
      text = text.lower()
      text_p = "".join([char for char in text if char not in string.punctuation])
      words = word_tokenize(text_p)
      stop_words = stopwords.words('english')
      filtered_words = [word for word in words if word not in stop_words]
      porter = PorterStemmer()
      final = [porter.stem(word) for word in filtered_words]
      final = " ".join([porter.stem(word) for word in filtered_words])
      result.append(final)
  return result

cleaned_input = clean_text(inputs)
tokenized_texts = tokenizer(cleaned_input, padding="max_length", truncation=True, truncation_strategy='only_last', return_tensors="pt")

execution_time = time.time() - start_time
print(f"Execution time: {execution_time} seconds")
start_time = time.time()

print("Inferencing the model")
result = []
with torch.no_grad():
    for input_ids, attention_mask in zip(tokenized_texts.input_ids, tokenized_texts.attention_mask):
        # print(input_ids.shape, attention_mask.shape)
        input_ids=input_ids.view(1, -1)
        attention_mask = attention_mask.view(1, -1)
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output['logits']
        probs = torch.sigmoid(logits)
        threshold = 0.5
        binary_preds = (probs >= threshold).float()
        predictions = binary_preds
        result.append(predictions.tolist()[0])
execution_time = time.time() - start_time
print(f"Execution time: {execution_time} seconds")

lookup_table = ["Feeling-bad-about-yourself-or-that-you-are-a-failure-or-have-let-yourself-or-your-family-down",
                "Feeling-down-depressed-or-hopeless",
                "Feeling-tired-or-having-little-energy",
                "Little-interest-or-pleasure-in-doing",
                "Moving-or-speaking-so-slowly-that-other-people-could-have-noticed-Or-the-opposite-being-so-fidgety-or-restless-that-you-have-been-moving-around-a-lot-more-than-usual",
                "Poor-appetite-or-overeating",
                "Thoughts-that-you-would-be-better-off-dead-or-of-hurting-yourself-in-some-way",
                "Trouble-concentrating-on-things-such-as-reading-the-newspaper-or-watching-television",
                "Trouble-falling-or-staying-asleep-or-sleeping-too-much",
                ]


result_text = []
for r in result:
    result_0 = []
    for i, j in enumerate(r):
        result_0_0 = []
        result_0_0.append(lookup_table[i])
        result_0_0.append("yes" if j==0.0 else "no")
        result_0.append(result_0_0)
    result_text.append(result_0)



with open('result.txt', 'w') as file:
    for inner_list in result_text:
        file.write(f'{{{[inner_list]}}}\n')


print("Results Stored in result.txt")


        
        




