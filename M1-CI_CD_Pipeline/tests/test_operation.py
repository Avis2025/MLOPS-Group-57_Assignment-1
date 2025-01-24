from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer
import joblib
import sys
import os

# Add the path to scr directory to import model.py
#sys.path.append(os.path.abspath("C:\Users\DS\Desktop\SEM 3\MLOPS\MLOPS Assignment\MLOPS-Group-57_Assignment-1\M1-CI_CD_Pipeline\src\model.py"))
from src.model import train_and_save_model

# Import the training function
from src.model import train_and_save_model

def test_model(model_path="../models/logistic_regression_model.pkl", scaler_path="../models/scaler.pkl"):
    
    """
    Test the saved logistic regression model on the Breast Cancer dataset.

    Args:
        model_path (str): Path to the saved trained model.
        scaler_path (str): Path to the saved scaler.
    """
    # Load the saved model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Load the dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Scale the data using the saved scaler
    X = scaler.transform(X)

    # Split the data into testing sets only
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Make predictions using the loaded model
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    # First, train the model (this can be skipped if the model is already saved)
    train_and_save_model()

    # Then, test the model
    test_model()
