import unittest
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
import sys
sys.path.append('../..')  # Add parent directory to path
from utils.data_utils_baseline import load_data

class TestSVMBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load preprocessed data
        cls.train_data, cls.test_data = load_data()
        # Separate features and labels
        cls.X_train, cls.y_train = cls.train_data['text'], cls.train_data['label']
        cls.X_test, cls.y_test = cls.test_data['text'], cls.test_data['label']
        # Train SVM classifier
        cls.model = SVC(kernel='rbf', C=1.0, random_state=42)
        cls.model.fit(cls.X_train, cls.y_train)
        # Make predictions
        cls.y_pred = cls.model.predict(cls.X_test)

    def test_accuracy(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_f1_score(self):
        f1 = f1_score(self.y_test, self.y_pred, average="macro")
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)

if __name__ == '__main__':
    unittest.main()