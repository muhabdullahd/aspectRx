import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

dataset_folder = os.path.join(os.path.dirname(__file__), "..", "Dataset")
data_path = os.path.join(dataset_folder, "cadec_absa_train.tsv")  # Using CADEC dataset instead

# Read the TSV file into a DataFrame (Note: This will be modified to use CADEC properly)
# This is just a placeholder since Restaurant_Reviews.tsv was removed
try:
    data = pd.read_csv(data_path, delimiter='\t')
except FileNotFoundError:
    print("Warning: Using empty DataFrame as placeholder since dataset file was not found")
    data = pd.DataFrame(columns=['tokens', 'absa1'])

# Preprocess the data
def preprocess_text(text):
    # Uncapitalize
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load preprocessed data
def load_data():
    data['Review'] = data['Review'].apply(preprocess_text)
    # Tokenize the text
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['Review']).toarray()
    y = data['Liked'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = {'text': X_train, 'label': y_train}
    test_data = {'text': X_test, 'label': y_test}
    # Print shapes of the datasets
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    return train_data, test_data