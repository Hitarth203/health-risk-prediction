import os

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

    # File paths
    DIABETES_PATH = os.path.join(DATA_RAW_DIR, "diabetes.csv")
    HEART_PATH = os.path.join(DATA_RAW_DIR, "heart.csv")
    STROKE_PATH = os.path.join(DATA_RAW_DIR, "stroke_data.csv")
    
    MODELS_DIR = os.path.join("artifacts", "models")
