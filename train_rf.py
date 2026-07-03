import os
import glob
import joblib
import numpy as np

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from feature_extractor import FeatureExtractor


# ===========================================
# Configuration
# ===========================================

REAL_FOLDER = "dataset/real"
SCREEN_FOLDER = "dataset/screen"

MODEL_PATH = "rf_model.pkl"

RANDOM_STATE = 42


# ===========================================
# Load Dataset
# ===========================================

extractor = FeatureExtractor()

X = []
y = []


def load_images(folder, label):

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    files = []

    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))

    print(f"\nLoading {len(files)} images from {folder}")

    for img_path in tqdm(files):

        try:

            features = extractor.extract(img_path)

            X.append(features)

            y.append(label)

        except Exception as e:

            print(f"Skipping {img_path}")

            print(e)


load_images(REAL_FOLDER, 0)

load_images(SCREEN_FOLDER, 1)

X = np.array(X)

y = np.array(y)

print("\nDataset Shape")

print(X.shape)

print(y.shape)


# ===========================================
# Train Test Split
# ===========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y

)


print("\nTraining Samples :", len(X_train))

print("Testing Samples :", len(X_test))


# ===========================================
# Random Forest
# ===========================================

rf = RandomForestClassifier(

    n_estimators=300,

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    random_state=RANDOM_STATE,

    n_jobs=-1

)

print("\nTraining Random Forest...")

rf.fit(

    X_train,

    y_train

)


# ===========================================
# Prediction
# ===========================================

pred = rf.predict(X_test)

prob = rf.predict_proba(X_test)[:, 1]


# ===========================================
# Evaluation
# ===========================================

acc = accuracy_score(

    y_test,

    pred

)

print("\n==============================")

print("Accuracy :", round(acc * 100, 2), "%")

print("==============================")

print("\nClassification Report\n")

print(

    classification_report(

        y_test,

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

        y_test,

        pred

    )

)


# ===========================================
# Feature Importance
# ===========================================

importance = rf.feature_importances_

indices = np.argsort(importance)[::-1]

print("\nTop 20 Important Features\n")

for i in range(min(20, len(indices))):

    idx = indices[i]

    print(

        f"{i+1:2d}. "

        f"Feature {idx:3d} "

        f"Importance = {importance[idx]:.5f}"

    )


# ===========================================
# Save Model
# ===========================================

joblib.dump(

    rf,

    MODEL_PATH

)

print("\nModel Saved ->", MODEL_PATH)