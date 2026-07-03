"""
=========================================================
Feature Selection Experiment
---------------------------------------------------------
Real Photo vs Screen Recapture Detection

Part 1

✔ Imports
✔ Load Dataset
✔ Feature Extraction
✔ Train/Test Split
✔ Train Baseline Random Forest
✔ Rank Features

Author : Yashi
=========================================================
"""

import os
import glob
import numpy as np
import pandas as pd

from tqdm import tqdm
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from feature_extractor import FeatureExtractor


# ==========================================================
# Configuration
# ==========================================================

REAL_FOLDER = "dataset/real"
SCREEN_FOLDER = "dataset/screen"

MODEL_SAVE_PATH = "feature_selector_rf.pkl"

RANDOM_STATE = 42


# ==========================================================
# Load Dataset
# ==========================================================

extractor = FeatureExtractor()

X = []
y = []

extensions = [
   

    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp"

]



def load_folder(folder, label):

    files = []

    for ext in extensions:
        files.extend(
            glob.glob(
                os.path.join(folder, ext)
            )
        )

    print(f"\nLoading {len(files)} images from {folder}")

    for img_path in tqdm(files):

        try:

            feat = extractor.extract(img_path)

            X.append(feat)

            y.append(label)

        except Exception as e:

            print(f"Skipping : {img_path}")

            print(e)


load_folder(REAL_FOLDER, 0)

load_folder(SCREEN_FOLDER, 1)


X = np.array(X)

y = np.array(y)


print("\n=====================================")
print("Dataset Loaded")
print("=====================================")

print("Feature Matrix :", X.shape)

print("Labels         :", y.shape)


# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y

)

print("\nTraining Samples :", len(X_train))

print("Testing Samples  :", len(X_test))


# ==========================================================
# Baseline Random Forest
# ==========================================================

print("\nTraining Baseline Random Forest...\n")

rf = RandomForestClassifier(

    n_estimators=500,

    random_state=RANDOM_STATE,

    n_jobs=-1

)

rf.fit(

    X_train,

    y_train

)

pred = rf.predict(X_test)

accuracy = accuracy_score(

    y_test,

    pred

)

print("=" * 40)

print(f"Baseline Accuracy : {accuracy*100:.2f}%")

print("=" * 40)


# ==========================================================
# Rank Features
# ==========================================================

feature_names = extractor.get_feature_names()

importance = rf.feature_importances_

feature_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance

})

feature_df = feature_df.sort_values(

    by="Importance",

    ascending=False

).reset_index(drop=True)

feature_df["Rank"] = feature_df.index + 1

feature_df = feature_df[

    [

        "Rank",

        "Feature",

        "Importance"

    ]

]

feature_df.to_csv(

    "feature_ranking.csv",

    index=False

)

print("\nFeature Ranking Saved -> feature_ranking.csv")


# ==========================================================
# Save RF Model
# ==========================================================

joblib.dump(

    rf,

    MODEL_SAVE_PATH

)

print("\nBaseline Model Saved ->", MODEL_SAVE_PATH)

print("\nReady for Feature Selection (Part 2)")
# ==========================================================
# Feature Selection Experiment
# ==========================================================

print("\n")
print("="*60)
print("Feature Selection Experiment")
print("="*60)

feature_counts = [

    20,
    40,
    60,
    80,
    100,
    133

]

results = []

for n_features in feature_counts:

    print(f"\nTraining using Top {n_features} Features")

    # ------------------------------------
    # Select Top Features
    # ------------------------------------

    selected = feature_df.iloc[
        :n_features
    ]["Feature"].tolist()

    indices = [

        feature_names.index(name)

        for name in selected

    ]

    X_train_sel = X_train[:, indices]

    X_test_sel = X_test[:, indices]

    # ------------------------------------
    # Train RF
    # ------------------------------------

    rf = RandomForestClassifier(

        n_estimators=500,

        random_state=RANDOM_STATE,

        n_jobs=-1

    )

    rf.fit(

        X_train_sel,

        y_train

    )

    pred = rf.predict(

        X_test_sel

    )

    acc = accuracy_score(

        y_test,

        pred

    )

    print(

        f"Accuracy : {acc*100:.2f}%"

    )

    results.append({

        "Features": n_features,

        "RF Accuracy": round(

            acc*100,

            2

        )

    })


# ==========================================================
# Save Results
# ==========================================================

results_df = pd.DataFrame(

    results

)

results_df.to_csv(

    "feature_selection_results.csv",

    index=False

)

print("\n")
print("="*60)
print("Final Results")
print("="*60)

print(results_df)

print("\nResults Saved -> feature_selection_results.csv")