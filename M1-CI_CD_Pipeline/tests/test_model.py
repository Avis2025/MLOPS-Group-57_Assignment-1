import pytest
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

MODEL_PATH = "model.pkl"
DATA_PATH = "data/breast_cancer.csv"

@pytest.fixture
def load_data():
    """
    Load the Breast Cancer dataset for testing.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    return data

@pytest.fixture
def load_model():
    """
    Load the trained model for testing.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model

def test_model_type(load_model):
    """
    Test if the loaded model is of the expected type.
    """
    model = load_model
    assert isinstance(model, RandomForestClassifier), "Loaded model is not a RandomForestClassifier."

def test_model_prediction_shape(load_model, load_data):
    """
    Test if the model predicts the correct number of outputs.
    """
    model = load_model
    data = load_data

    # Prepare input data (drop target column if it exists)
    if "target" in data.columns:
        X = data.drop(columns=["target"])
    else:
        X = data

    predictions = model.predict(X)
    assert len(predictions) == len(X), "Number of predictions does not match the number of input samples."

def test_model_accuracy(load_model, load_data):
    """
    Test if the model achieves at least 80% accuracy on the test dataset.
    """
    model = load_model
    data = load_data

    # Prepare features and target
    X = data.drop(columns=["target"])
    y = data["target"]

    # Predict and calculate accuracy
    predictions = model.predict(X)
    accuracy = (predictions == y).mean()
    assert accuracy >= 0.8, f"Model accuracy is below expected threshold: {accuracy:.2f}"

def test_model_file_exists():
    """
    Test if the model file exists.
    """
    assert os.path.exists(MODEL_PATH), "Model file does not exist."

def test_data_file_exists():
    """
    Test if the data file exists.
    """
    assert os.path.exists(DATA_PATH), "Data file does not exist."
