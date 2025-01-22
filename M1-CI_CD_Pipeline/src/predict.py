import pandas as pd
import joblib
import argparse
import os

def load_model(model_path):
    """
    Load the trained model from a file.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    return model

def make_predictions(model, input_data):
    """
    Make predictions using the trained model.
    """
    predictions = model.predict(input_data)
    return predictions

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict outcomes using a trained ML model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file (e.g., model.pkl).")
    parser.add_argument("--input", type=str, required=True, help="Path to the input data file (CSV format).")
    parser.add_argument("--output", type=str, required=False, default="predictions.csv", help="Path to save predictions (CSV format).")
    args = parser.parse_args()

    # Load the model
    print("Loading model...")
    model = load_model(args.model)

    # Load input data
    print("Loading input data...")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input data file not found at: {args.input}")
    input_data = pd.read_csv(args.input)

    # Ensure input data does not contain the target column
    if "target" in input_data.columns:
        input_data = input_data.drop(columns=["target"])

    # Make predictions
    print("Making predictions...")
    predictions = make_predictions(model, input_data)

    # Save predictions
    output_path = args.output
    print(f"Saving predictions to: {output_path}")
    output_df = pd.DataFrame({"Prediction": predictions})
    output_df.to_csv(output_path, index=False)

    print("Predictions saved successfully.")
