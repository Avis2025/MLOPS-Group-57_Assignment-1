import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Scale the features for better model performance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model with increased iterations
model = LogisticRegression(max_iter=2000, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Ensure the 'models' directory exists
os.makedirs('../models', exist_ok=True)

# Save the trained model and the scaler
joblib.dump(model, '../models/logistic_regression_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')

print("Model and scaler saved successfully!")