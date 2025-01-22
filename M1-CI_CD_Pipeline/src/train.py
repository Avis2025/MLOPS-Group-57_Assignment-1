import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def load_data(file_path):
    """
    Load dataset from a CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    data = pd.read_csv(file_path)
    return data

def train_model(data):
    """
    Train a RandomForest model on the dataset.
    """
    # Split dataset into features and target
    X = data.drop(columns=["target"])
    y = data["target"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Training Complete. Model Accuracy: {accuracy}")

    return model

def save_model(model, output_path="model.pkl"):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, output_path)
    print(f"Model saved at: {output_path}")

if __name__ == "__main__":
    # File path to the dataset
    data_file_path = "data/breast_cancer.csv"

    # Load the data
    print("Loading data...")
    dataset = load_data(data_file_path)

    # Train the model
    print("Training model...")
    trained_model = train_model(dataset)

    # Save the model
    print("Saving model...")
    save_model(trained_model)
