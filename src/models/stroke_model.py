import os
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from utils.logger import setup_logger
from src.config import Config


logger = setup_logger()

def train_stroke_model():
    logger.info("Training Stroke Prediction Model using SVM...")

    # Load processed stroke data
    processed_dir = Config.DATA_PROCESSED_DIR
    X_train = pd.read_csv(os.path.join(processed_dir, "stroke_X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "stroke_X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "stroke_y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_dir, "stroke_y_test.csv")).values.ravel()

    # Train the model
    model = SVC(kernel='linear', class_weight='balanced',C=0.5, probability=True, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    logger.info("Classification Report:\n" + classification_report(y_test, y_pred))
    logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

    # Save the model
    model_path = os.path.join(Config.MODELS_DIR, "stroke_model.pkl")
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Stroke model saved at: {model_path}")

