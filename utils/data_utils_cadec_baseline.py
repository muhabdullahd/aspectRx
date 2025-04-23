import pandas as pd
import re
import os
from sklearn.feature_extraction.text import CountVectorizer

def load_cadec_data(binary_sentiment=True):
    """
    Load and preprocess CADEC dataset for baseline models.
    
    Args:
        binary_sentiment (bool): If True, converts sentiment to binary (positive/negative)
                                 by treating neutral as negative.
    
    Returns:
        tuple: (train_data, test_data) dictionaries containing 'text' and 'label'
    """
    # Paths to dataset files
    dataset_folder = os.path.join(os.path.dirname(__file__), "..", "Dataset")
    train_path = os.path.join(dataset_folder, "cadec_absa_train.tsv")
    test_path = os.path.join(dataset_folder, "cadec_absa_test.tsv")
    
    # Load datasets
    train_df = pd.read_csv(train_path, delimiter='\t')
    test_df = pd.read_csv(test_path, delimiter='\t')
    
    # Extract review text and sentiment (absa1 column contains sentiment values)
    X_train = train_df['tokens'].values
    X_test = test_df['tokens'].values
    
    # Extract sentiment scores (0=negative, 1=neutral, 2=positive)
    # For training data
    y_train = train_df['absa1'].apply(lambda x: int(x.split(';')[-1]) if isinstance(x, str) else x).values

    # For test data
    y_test = test_df['absa1'].apply(lambda x: int(x.split(';')[-1]) if isinstance(x, str) else x).values
    
    # Convert to binary sentiment if needed
    if binary_sentiment:
        # Convert: (0,1) → 0 (negative/neutral), 2 → 1 (positive)
        y_train = (y_train == 2).astype(int)
        y_test = (y_test == 2).astype(int)
    
    # Preprocess text
    X_train = [preprocess_text(text) for text in X_train]
    X_test = [preprocess_text(text) for text in X_test]
    
    # Vectorize text
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()
    
    train_data = {'text': X_train_vec, 'label': y_train}
    test_data = {'text': X_test_vec, 'label': y_test}
    
    # Print shapes of the datasets
    print(f"Training data shape: {X_train_vec.shape}")
    print(f"Testing data shape: {X_test_vec.shape}")
    
    return train_data, test_data

def preprocess_text(text):
    """Preprocess text for baseline models."""
    # Uncapitalize
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text