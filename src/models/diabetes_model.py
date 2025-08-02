import pandas as pd 
import os 
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from src.config import Config
from utils.logger import setup_logger

logger = setup_logger()

def train_diabetes_model():
    logger.info("Training Logistic Regression model on Diabetes dataset...")

    # Load preprocessed data
    processed_dir = Config.DATA_PROCESSED_DIR
    X_train = pd.read_csv(os.path.join(processed_dir, "diabetes_X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "diabetes_X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "diabetes_y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_dir, "diabetes_y_test.csv")).values.ravel()

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)

    logger.info("Model Evaluation Metrics:")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"Precision: {precision_score(y_test, y_pred):.4f}")
    logger.info(f"Recall: {recall_score(y_test, y_pred):.4f}")
    logger.info(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    logger.info(f"ROC AUC: {roc_auc_score(y_test, y_pred):.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    # Save model
    model_path = os.path.join(Config.MODELS_DIR, "diabetes_model.pkl")
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")
    