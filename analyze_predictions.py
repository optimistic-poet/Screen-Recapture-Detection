import os
import glob
import shutil
import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from feature_extractor import FeatureExtractor


# ======================================================
# Configuration
# ======================================================

MODEL_PATH = "rf_model.pkl"
# MODEL_PATH = "xgb_model.pkl"

REAL_FOLDER = "dataset/real"
SCREEN_FOLDER = "dataset/screen"

ANALYSIS_FOLDER = "analysis"

REAL_AS_SCREEN = os.path.join(
    ANALYSIS_FOLDER,
    "real_as_screen"
)

SCREEN_AS_REAL = os.path.join(
    ANALYSIS_FOLDER,
    "screen_as_real"
)

os.makedirs(REAL_AS_SCREEN, exist_ok=True)
os.makedirs(SCREEN_AS_REAL, exist_ok=True)


# ======================================================
# Load Model
# ======================================================

model = joblib.load(MODEL_PATH)

extractor = FeatureExtractor()


# ======================================================
# Dataset
# ======================================================

extensions = [
    "*.jpg",
    "*.JPG",
    "*.jpeg",
    "*.JPEG",
    "*.png",
    "*.PNG",
    "*.bmp",
    "*.BMP"
]

image_paths = []
labels = []


def load_folder(folder, label):

    files = []

    for ext in extensions:
        files.extend(
            glob.glob(
                os.path.join(folder, ext)
            )
        )

    print(f"Loading {folder}")

    for f in files:

        image_paths.append(f)

        labels.append(label)


load_folder(REAL_FOLDER, 0)
load_folder(SCREEN_FOLDER, 1)


print()

print("Total Images :", len(image_paths))


# ======================================================
# Prediction
# ======================================================

rows = []

predictions = []

probabilities = []

y_true = []


for img_path, label in tqdm(
    zip(image_paths, labels),
    total=len(image_paths)
):

    try:

        feat = extractor.extract(img_path)

        feat = feat.reshape(1, -1)

        pred = int(
            model.predict(feat)[0]
        )

        prob = float(
            model.predict_proba(feat)[0][1]
        )

        predictions.append(pred)

        probabilities.append(prob)

        y_true.append(label)

        rows.append({

            "image": os.path.basename(img_path),

            "path": img_path,

            "true_label": "Real" if label == 0 else "Screen",

            "predicted": "Real" if pred == 0 else "Screen",

            "probability_screen": prob

        })

        # ------------------------------------------
        # Save Wrong Predictions
        # ------------------------------------------

        if pred != label:

            if label == 0:

                shutil.copy(

                    img_path,

                    os.path.join(

                        REAL_AS_SCREEN,

                        os.path.basename(img_path)

                    )

                )

            else:

                shutil.copy(

                    img_path,

                    os.path.join(

                        SCREEN_AS_REAL,

                        os.path.basename(img_path)

                    )

                )

    except Exception as e:

        print("Skipping", img_path)

        print(e)
        # ======================================================
# Convert to DataFrame
# ======================================================

df = pd.DataFrame(rows)

csv_path = os.path.join(
    ANALYSIS_FOLDER,
    "predictions.csv"
)

df.to_csv(
    csv_path,
    index=False
)

print("\nPredictions saved to:")
print(csv_path)


# ======================================================
# Metrics
# ======================================================

accuracy = accuracy_score(
    y_true,
    predictions
)

print("\n========================================")
print(f"Accuracy : {accuracy*100:.2f}%")
print("========================================")

print("\nClassification Report\n")

print(
    classification_report(
        y_true,
        predictions,
        target_names=[
            "Real",
            "Screen"
        ]
    )
)

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_true,
        predictions
    )
)


# ======================================================
# Feature Importance
# ======================================================

if hasattr(model, "feature_importances_"):

    importance = model.feature_importances_

    feature_names = extractor.get_feature_names()

    feature_df = pd.DataFrame({

        "Feature_Name": feature_names,

        "Importance": importance

    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )
    feature_df["Rank"] = range(
    1,
    len(feature_df) + 1
)   
    feature_df = feature_df[
    [
        "Rank",
        "Feature_Name",
        "Importance"
    ]
]

    feature_csv = os.path.join(
        ANALYSIS_FOLDER,
        "feature_importance.csv"
    )

    feature_df.to_csv(
        feature_csv,
        index=False
    )

    print("\nFeature importance saved to:")

    print(feature_csv)


# ======================================================
# Misclassification Summary
# ======================================================

real_to_screen = len(os.listdir(REAL_AS_SCREEN))

screen_to_real = len(os.listdir(SCREEN_AS_REAL))

print("\n========================================")

print("Misclassified Images")

print("========================================")

print(f"Real   -> Screen : {real_to_screen}")

print(f"Screen -> Real   : {screen_to_real}")

print("\nAnalysis folder created successfully.")