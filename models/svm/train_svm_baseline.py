import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import sys
sys.path.append('../..')  # Add parent directory to path
from utils.data_utils_baseline import load_data

def train_svm_baseline():
    # Load preprocessed data
    train_data, test_data = load_data()

    # Separate features and labels
    X_train, y_train = train_data['text'], train_data['label']
    X_test, y_test = test_data['text'], test_data['label']

    # Train an SVM classifier with RBF kernel
    print("Training SVM classifier...")
    model = SVC(kernel='rbf', C=1.0, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print("\nSVM Baseline Results:")
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_svm_baseline()