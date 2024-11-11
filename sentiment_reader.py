import pandas as pd
import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def sentiment_reader(baslik):
    file_name = baslik.replace(" ","_")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
    model = AutoModelForSequenceClassification.from_pretrained("savasy/bert-base-turkish-sentiment-cased")

    def sentiment_analysis(text):
        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities from logits
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities

    # Read the CSV file
    input_file_path = file_name + '.csv'
    output_file_path = file_name + '_Sentiment.csv'

    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file_path)

    # Initialize lists to store the results
    probabilities_negative = []
    probabilities_positive = []

    # Perform sentiment analysis on the first column
    for text in df.iloc[:, 0]:
        probabilities = sentiment_analysis(text)
        probabilities_negative.append(probabilities[0, 0].item())  # Negative probability
        probabilities_positive.append(probabilities[0, 1].item())  # Positive probability

    # Add the results to the DataFrame
    df['Negative'] = probabilities_negative
    df['Positive'] = probabilities_positive

    # Save the updated DataFrame to a CSV file
    df.to_csv(output_file_path, index=False,quoting=csv.QUOTE_ALL)

    print(f"Sentiment analysis completed. Output saved to {output_file_path}")
