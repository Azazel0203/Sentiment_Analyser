import streamlit as st
import torch
import os
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Function for text cleaning
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

# Function for model inference
def inference(tokenizer, model, cleaned_input):
    tokenized_texts = tokenizer(cleaned_input, padding="max_length", truncation=True, truncation_strategy='only_last', return_tensors="pt")
    result = []
    with torch.no_grad():
        for input_ids, attention_mask in zip(tokenized_texts.input_ids, tokenized_texts.attention_mask):
            input_ids = input_ids.view(1, -1)
            attention_mask = attention_mask.view(1, -1)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output['logits']
            probs = torch.sigmoid(logits)
            threshold = 0.5
            binary_preds = (probs >= threshold).float()
            predictions = binary_preds
            result.append(predictions.tolist()[0])
    return result[0]

# Main function for Streamlit app
def main():
    st.title('Sentiment Analyser')

    # Load tokenizer and model
    # with st.spinner("Loading the Model and tokenizer..."):
    tokenizer = AutoTokenizer.from_pretrained("Aad456334/Sentiment_Analyser")
    device = "cpu"  # Change to "cuda" if you have GPU support
    model = AutoModelForSequenceClassification.from_pretrained("Aad456334/Sentiment_Analyser")

    

    st.write("Done loading the model.")

    # Input text area
    input_text = st.text_area("Enter your text here:", "")

    if st.button("Analyze"):
        # Text cleaning
        st.write("Cleaning the text...")
        inputs = [input_text]
        cleaned_input = clean_text(inputs)

        # Model inference
        st.write("Performing sentiment analysis...")
        result = inference(tokenizer, model, cleaned_input)

        # Define classes
        lookup_table = [
            "Feeling-bad-about-yourself-or-that-you-are-a-failure-or-have-let-yourself-or-your-family-down",
            "Feeling-down-depressed-or-hopeless",
            "Feeling-tired-or-having-little-energy",
            "Little-interest-or-pleasure-in-doing",
            "Moving-or-speaking-so-slowly-that-other-people-could-have-noticed-Or-the-opposite-being-so-fidgety-or-restless-that-you-have-been-moving-around-a-lot-more-than-usual",
            "Poor-appetite-or-overeating",
            "Thoughts-that-you-would-be-better-off-dead-or-of-hurting-yourself-in-some-way",
            "Trouble-concentrating-on-things-such-as-reading-the-newspaper-or-watching-television",
            "Trouble-falling-or-staying-asleep-or-sleeping-too-much"
        ]

        # Display results
        st.write("Results:")
        for i, prediction in enumerate(result):
            st.write(f"- {lookup_table[i]}: {'yes' if prediction == 0 else 'no'}")

# Run the app
if __name__ == '__main__':
    main()