# Health Risk Prediction App 🩺

A machine learning project that predicts the likelihood of **Diabetes**, **Heart Disease**, and **Stroke** using structured healthcare data. Built with Python and scikit-learn, the app processes input features and trains separate models for each disease using classification algorithms like Random Forest and SVM.

---

## 📁 Project Structure
health-risk-prediction/
│
├── data/
│ ├── raw/ # Original datasets (CSV files)
│ └── processed/ # Cleaned and preprocessed data
│
├── models/ # Saved trained models (.pkl files)
│
├── src/ # Source code
│ ├── preprocess/ # Preprocessing scripts for each dataset
│ ├── train/ # Model training scripts
│ ├── config.py # Centralized path config
│
├── utils/ # Utilities (e.g., logging)
│ └── logger.py
│
├── requirements.txt # Project dependencies
├── README.md # Project overview

## ⚙️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Hitarth203/health-risk-prediction.git
cd health-risk-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing

Each dataset has its own preprocessing script:

```bash
python src/preprocess/preprocess_diabetes.py
python src/preprocess/preprocess_heart.py
python src/preprocess/preprocess_stroke.py
```

### 4. Train Models

```bash
python src/train/train_diabetes.py
python src/train/train_heart.py
python src/train/train_stroke.py
```

> ✅ Trained models will be saved in the `models/` directory.

---

## 🔍 Example Metrics Logged

```yaml
Accuracy: 0.87
Precision: 0.90
Recall: 0.84
F1 Score: 0.87
ROC AUC: 0.91
```

> 📌 Detailed classification reports are also printed in the console.

---

## 📦 Dependencies

- Python 3.8+
- scikit-learn
- pandas
- numpy
- seaborn
- matplotlib
- joblib

> See `requirements.txt` for the complete list.
**Author**
Hitarth Wadhwani
BCA (AI & ML), Data Science Enthusiast
GitHub: @Hitarth203
