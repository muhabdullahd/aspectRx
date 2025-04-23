"""
Train an SVM classifier on CADEC dataset for sentiment analysis.
"""
import os
import sys
import pandas as pd
import pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.data_utils_cadec_baseline import load_cadec_data

def train_svm():
    """Train and evaluate an SVM model on the CADEC dataset."""
    # Load preprocessed data
    train_data, test_data = load_cadec_data(binary_sentiment=True)
    
    # Separate features and labels
    X_train, y_train = train_data['text'], train_data['label']
    X_test, y_test = test_data['text'], test_data['label']
    
    print(f"Training SVM classifier on CADEC dataset...")
    
    # Train an SVM classifier
    model = LinearSVC(random_state=42, max_iter=10000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    # Print classification report
    print(classification_report(y_test, y_pred))
    
    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), "cadec_svm_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    train_svm()