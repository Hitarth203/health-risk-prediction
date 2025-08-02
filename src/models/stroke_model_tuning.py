import os
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from utils.logger import setup_logger
from src.config import Config

logger = setup_logger()

def tune_stroke_model():
    logger.info("Starting hyperparameter tuning for Stroke Prediction Model...")

    # Load preprocessed data
    processed_dir = Config.DATA_PROCESSED_DIR
    X_train = pd.read_csv(os.path.join(processed_dir, "stroke_X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "stroke_X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "stroke_y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_dir, "stroke_y_test.csv")).values.ravel()

    # Define parameter grid
    param_grid = {
        "C": [0.10, 1, 10],
        "kernel": ["linear", "rbf"],
        "class_weight": [None, "balanced"],
        "gamma": ["scale", "auto"]
    }

    # Initialize base model
    svc = SVC(random_state=42)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # GridSearchCV
    grid_search = GridSearchCV(
        svc,
        param_grid,
        cv=skf,
        scoring="f1",
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    logger.info(f"Best Parameters: {grid_search.best_params_}")

    # Evaluate
    y_pred = best_model.predict(X_test)
    logger.info("Classification Report:\n" + classification_report(y_test, y_pred))
    logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

    # Save best model
    model_path = os.path.join(Config.MODELS_DIR, "stroke_model_tuned.pkl")
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, model_path)
    logger.info(f"Tuned Stroke model saved at: {model_path}")

if __name__ == "__main__":
    tune_stroke_model()