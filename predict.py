import sys
import joblib
import numpy as np

from feature_extractor import FeatureExtractor


# ======================================================
# Configuration
# ======================================================

MODEL_PATH = "xgb_model.pkl"
# Change to rf_model.pkl if you want Random Forest


# ======================================================
# Load Model
# ======================================================

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

extractor = FeatureExtractor()


# ======================================================
# Prediction Function
# ======================================================

def predict(image_path):

    try:

        features = extractor.extract(image_path)

        features = features.reshape(1, -1)

        probability = model.predict_proba(features)[0][1]

        probability = float(probability)

        probability = max(0.0, min(1.0, probability))

        return probability

    except Exception as e:

        print(f"Prediction Error: {e}")

        return None


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("Usage:")

        print("python predict.py image.jpg")

        sys.exit(1)

    image_path = sys.argv[1]

    score = predict(image_path)

    if score is not None:

        print(f"{score:.4f}")