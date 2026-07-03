import os
import glob
import time
import joblib
import numpy as np

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from feature_extractor import FeatureExtractor


# =====================================================
# Configuration
# =====================================================

MODEL_PATH = "rf_model.pkl"
# Change to xgb_model.pkl if needed

REAL_FOLDER = "dataset/real"
SCREEN_FOLDER = "dataset/screen"


# =====================================================
# Load Model
# =====================================================

model = joblib.load(MODEL_PATH)

extractor = FeatureExtractor()


# =====================================================
# Dataset
# =====================================================

X = []
y = []

extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]


def process_folder(folder, label):

    files = []

    for ext in extensions:
        files.extend(
            glob.glob(
                os.path.join(folder, ext)
            )
        )

    print(f"Processing {folder}")

    for img in tqdm(files):

        feat = extractor.extract(img)

        X.append(feat)

        y.append(label)


process_folder(REAL_FOLDER, 0)
process_folder(SCREEN_FOLDER, 1)

X = np.array(X)
y = np.array(y)

print("\nDataset Shape")

print(X.shape)


# =====================================================
# Prediction + Latency
# =====================================================

pred = []
prob = []

times = []

for sample in X:

    sample = sample.reshape(1, -1)

    start = time.perf_counter()

    p = model.predict(sample)[0]

    pr = model.predict_proba(sample)[0][1]

    end = time.perf_counter()

    pred.append(p)

    prob.append(pr)

    times.append(end - start)


pred = np.array(pred)
prob = np.array(prob)


# =====================================================
# Metrics
# =====================================================

accuracy = accuracy_score(y, pred)

auc = roc_auc_score(y, prob)


print("\n============================")

print("Accuracy :", round(accuracy * 100, 2), "%")

print("ROC AUC :", round(auc, 4))

print("============================")


print("\nClassification Report\n")

print(

    classification_report(

        y,

        pred,

        target_names=[

            "Real",

            "Screen"

        ]

    )

)


print("\nConfusion Matrix\n")

print(

    confusion_matrix(

        y,

        pred

    )

)


# =====================================================
# Latency
# =====================================================

latency = np.mean(times) * 1000

print("\nAverage Latency")

print(f"{latency:.2f} ms / image")

print("\nMinimum")

print(f"{min(times)*1000:.2f} ms")

print("\nMaximum")

print(f"{max(times)*1000:.2f} ms")