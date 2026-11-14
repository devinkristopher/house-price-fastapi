import joblib
import numpy as np
import json
from typing import Dict, Any
import os

# Global variable to store the loaded model
model = None
metadata = None

def load_model_and_metadata():
    """
    Load the trained model and metadata from disk.
    This function should:
    1. Load model.pkl using joblib.load()
    2. Load model_metadata.json using json.load()
    3. Store them in the global variables
    4. Return True if successful, False if there's an error
    """
    global model, metadata

    try:
        # Load the model
        model = joblib.load("model.pkl")

        # Load the metadata
        with open("model_metadata.json", "r") as f:
            metadata = json.load(f)

        print("Model and metadata loaded successfully!")
        return True

    except Exception as e:
        print(f"Error loading model or metadata: {e}")
        return False

def make_prediction(house_features: Dict[str, Any]) -> float:
    """
    Make a price prediction for a single house.

    Args:
        house_features: Dictionary containing all 13 features

    Returns:
        Predicted price as a float

    This function should:
    1. Extract the features in the correct order (same order as training)
    2. Convert to numpy array with shape (1, 13)
    3. Use model.predict() to get prediction
    4. Return the prediction as a float
    """
    global model, metadata

    if model is None:
        raise ValueError("Model not loaded")

    if metadata is None:
        raise ValueError("Model metadata not loaded")

    # Extract features in the correct order
    feature_values = [house_features[feature] for feature in metadata["features"]]

    # Convert to numpy array with correct shape
    X = np.array([feature_values])

    # Make prediction and extract single value
    prediction = model.predict(X)[0]

    # Round to 2 decimal places for currency
    return round(float(prediction), 2)

def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.

    Returns:
        Dictionary containing model metadata

    This function should simply return the loaded metadata dictionary.
    If metadata is not loaded, it should raise an error.
    """
    global metadata

    if metadata is None:
        raise ValueError("Model metadata not loaded")

    return metadata

def check_health() -> Dict[str, Any]:
    """
    Check the health status of the service.

    Returns:
        Dictionary with health status information

    This function should:
    1. Check if model is loaded (model is not None)
    2. Check if metadata is loaded (metadata is not None)
    3. Return a dictionary with status information
    """
    global model, metadata

    model_loaded = model is not None
    metadata_loaded = metadata is not None

    if model_loaded and metadata_loaded:
        health_status = {
            "status": "healthy",
            "model_loaded": True,
            "message": "Model and metadata are loaded successfully"
        }
    else:
        health_status = {
            "status": "unhealthy",
            "model_loaded": model_loaded,
            "message": "Model or metadata is not loaded"
        }

    return health_status