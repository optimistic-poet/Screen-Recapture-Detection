import os
import glob
import joblib
import numpy as np

from tqdm import tqdm


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

from feature_extractor import FeatureExtractor


# ======================================================
# Configuration
# ======================================================

REAL_FOLDER = "dataset/real"
SCREEN_FOLDER = "dataset/screen"

MODEL_PATH = "xgb_model.pkl"

RANDOM_STATE = 42


# ======================================================
# Feature Extraction
# ======================================================

extractor = FeatureExtractor()

X = []
y = []


def load_images(folder, label):

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    files = []

    for ext in extensions:
        files.extend(
            glob.glob(
                os.path.join(folder, ext)
            )
        )

    print(f"\nLoading {len(files)} images from {folder}")

    for img in tqdm(files):

        try:

            feat = extractor.extract(img)

            X.append(feat)

            y.append(label)

        except Exception as e:

            print("Skipping :", img)
            print(e)


load_images(REAL_FOLDER, 0)

load_images(SCREEN_FOLDER, 1)

X = np.array(X)
y = np.array(y)

print("\nDataset Shape")

print(X.shape)
print(y.shape)


# ======================================================
# Train Test Split
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    stratify=y,

    random_state=RANDOM_STATE

)


# ======================================================
# XGBoost Model
# ======================================================

model = XGBClassifier(

    n_estimators=700,

    learning_rate=0.02,

    max_depth=5,

    subsample=0.8,

    colsample_bytree=1.0,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=RANDOM_STATE,

    n_jobs=-1

)

print("\nTraining XGBoost...\n")

model.fit(

    X_train,

    y_train

)


# ======================================================
# Prediction
# ======================================================

pred = model.predict(X_test)

prob = model.predict_proba(X_test)[:,1]


# ======================================================
# Accuracy
# ======================================================

acc = accuracy_score(

    y_test,

    pred

)

print("\n================================")

print("Accuracy :", round(acc*100,2), "%")

print("================================")


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


# ======================================================
# Feature Importance
# ======================================================

importance = model.feature_importances_

indices = np.argsort(

    importance

)[::-1]


print("\nTop 20 Features\n")

for i in range(20):

    idx = indices[i]

    print(

        f"{i+1:2d}. "

        f"Feature {idx:3d}"

        f" -> "

        f"{importance[idx]:.5f}"

    )


# ======================================================
# Save Model
# ======================================================

joblib.dump(

    model,

    MODEL_PATH

)

print("\nModel Saved Successfully")