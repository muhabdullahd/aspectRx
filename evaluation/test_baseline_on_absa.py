import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys
sys.path.append('..')  # Add parent directory to path
from utils.data_utils_baseline import load_data

def load_absa_test_dataset(file_path):
    """Load and preprocess ABSA test dataset"""
    absa_test_df = pd.read_csv(file_path, delimiter="\t")
    
    # Convert tokens from string representation to actual list and join them
    absa_test_df["text"] = absa_test_df["tokens"].apply(lambda x: " ".join(eval(x)))
    
    # Convert ABSA labels (extract sentiment from absa1)
    absa_test_df["label"] = absa_test_df["absa1"].apply(lambda x: 
        1 if "2" in str(x) else 0  # 2 indicates positive sentiment
    )
    
    return absa_test_df

def main():
    try:
        # First load training data to get the same vectorizer
        train_data = pd.read_csv("Dataset/Restaurant_Reviews.tsv", delimiter="\t")
        
        # Initialize and fit vectorizer on training reviews
        vectorizer = CountVectorizer(max_features=2000)
        X_train = vectorizer.fit_transform(train_data['Review']).toarray()
        y_train = train_data['Liked'].values
        
        # Train model
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB()
        model.fit(X_train, y_train)
        
        # Load and preprocess ABSA test data
        absa_test_path = "Dataset/rest16_quad_test_cleaned.tsv"
        absa_test_df = load_absa_test_dataset(absa_test_path)
        
        # Transform ABSA test text using the same vectorizer
        X_absa_test = vectorizer.transform(absa_test_df["text"]).toarray()
        y_absa_test = absa_test_df["label"].values
        
        # Make predictions
        y_pred = model.predict(X_absa_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_absa_test, y_pred)
        f1 = f1_score(y_absa_test, y_pred, average="macro")
        
        print("\nResults on ABSA test dataset:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_absa_test, y_pred))
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()