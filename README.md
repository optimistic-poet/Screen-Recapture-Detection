# Screen Recapture Detection using Machine Learning

## Overview

This project detects whether an image is:

- 📷 Real Camera Photo
- 🖥️ Screen Recaptured Photo

using handcrafted forensic image features and machine learning algorithms.

The project extracts multiple image characteristics such as texture, frequency-domain information, color statistics, edge information, JPEG artifacts, glare, and noise patterns to distinguish between genuine photographs and images captured from digital screens.

---

## Features Extracted

The feature extraction pipeline includes:

- FFT Features
- Advanced FFT Features
- Local Binary Pattern (LBP)
- Gray Level Co-occurrence Matrix (GLCM)
- Edge Features
- Sobel Features
- JPEG Artifact Features
- Glare Detection
- Noise Statistics
- Color Statistics
- Brightness Features
- Dynamic Range Features
- Entropy Features

**Total Features:** 140

---

## Machine Learning Models

The following models were implemented and compared:

- Random Forest
- XGBoost

---

## Results

| Model | Accuracy |
|-------|----------|
| Random Forest | **84.62%** |
| XGBoost | **86.81%** |

XGBoost achieved the best overall performance on the dataset.

---

## Project Structure

```
SALESCODE/
│
├── feature_extractor.py
├── train_rf.py
├── train_xg.py
├── predict.py
├── feature_selection.py
├── analyze_predictions.py
├── evaluate.py
├── test_feature.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Screen-Recapture-Detection.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Train Random Forest

```bash
python train_rf.py
```

---

## Train XGBoost

```bash
python train_xg.py
```

---

## Predict

```bash
python predict.py image.jpg
```

---

## Future Improvements

- Hyperparameter tuning of XGBoost
- Larger and more diverse dataset
- MobileNetV3 Transfer Learning
- Real-time deployment using Streamlit or Flask

---

## Author

**Yashi Singh**