import pandas as pd
from transformers import AutoTokenizer
import torch.nn.functional as F
import nltk
import string
import os
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
import json


tokenizer = AutoTokenizer.from_pretrained("sbcBI/sentiment_analysis_model")
tokenizer_path = os.path.join("artifact", "tokenizer")
tokenizer.save_pretrained(tokenizer_path)
# model = AutoModelForSequenceClassification.from_pretrained("sbcBI/sentiment_analysis_model", num_labels = 9, ignore_mismatched_sizes=True, problem_type="multi_label_classification")


json_data = pd.read_json("primate_dataset.json")

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

def clean_annotations(data):
    Y = []
    for row in data:
        # print (row)
        individual = []
        for r in row:
            if r[1]=='yes':
                individual.append(1)
            else:
                individual.append(0)
        Y.append(individual)
    return Y


Y = clean_annotations(json_data['annotations'])
X = clean_text(json_data['post_text'])

print (len(X))
print (len(Y))


def split_string(text):
    words = text.split(" ")
    total_words = len(words)
    midpoint_index = total_words // 2
    first_half = ' '.join(words[:midpoint_index])
    second_half = ' '.join(words[midpoint_index:])
    return first_half, second_half


while (True):
    print ("================================")
    tokenized_texts = tokenizer(X, padding="max_length", truncation=True, truncation_strategy='only_last')
    print("tokenization done")
    indexes = [] # indexes to remove
    for i, t in enumerate(tokenized_texts['input_ids']):
      if t[-1] != 0:
        indexes.append(i)
    
    print(f"Indexes to remove -> {len(indexes)}")
    if len(indexes) == 0:
        break
    
    X_to_break = []
    y_to_break = []
    for ind in indexes:
        X_to_break.append(X[ind])
        y_to_break.append(Y[ind])
    
    print (f"X_to_break -> {len(X_to_break)}")
    new_x = []
    new_y = []
    for i, x in enumerate(X_to_break):
        a, b = split_string(x)
        new_x.append(a)
        new_x.append(b)
        new_y.append(y_to_break[i])
        new_y.append(y_to_break[i])
    print (f"new_x ->{len(new_x)}")
    
    # deletion from original data
    index_to_remove = indexes.copy()
    index_to_remove.sort(reverse=True)
    for ind in index_to_remove:
        del X[ind]
        del Y[ind]
    
    X = X + new_x
    Y = Y + new_y
    print(len(X))
    print(len(Y))

print("Final")
print(len(X))
print(len(Y))
tokenized_texts = tokenizer(X, padding="max_length", truncation=True, truncation_strategy='only_last')

overflowed = []
for i, t in enumerate(tokenized_texts['input_ids']):
    if (t[-1]!=0):
        overflowed.append(i)

print (len(overflowed))





# Define file paths
file_path1 = os.path.join("artifact", "inputs.json")
file_path2 = os.path.join("artifact", "outputs.json")

# Save list1 to JSON
with open(file_path1, "w") as f:
    json.dump(X, f)

# Save list2 to JSON
with open(file_path2, "w") as f:
    json.dump(Y, f)

print("Lists saved as JSON files.")


    
    