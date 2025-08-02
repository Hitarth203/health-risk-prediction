import os
from src.config import Config
from utils.logger import setup_logger
from src.data_preprocessing import preprocess_diabetes_data, preprocess_heart_data, preprocess_stroke_data
from src.eda import perform_eda_on_diabetes
from src.models.diabetes_model import train_diabetes_model
from src.models.heart_model import train_heart_model
from src.models.stroke_model import train_stroke_model
from src.models.stroke_model_tuning import tune_stroke_model        

# Setup logger
logger = setup_logger()

def main():
    logger.info("Starting Health Risk Prediction App")

    # Define paths to the raw datasets
    diabetes_path = Config.DIABETES_PATH
    heart_path = Config.HEART_PATH
    stroke_path = Config.STROKE_PATH
    
    logger.info(f"Diabetes dataset: {diabetes_path}")
    logger.info(f"Heart disease dataset: {heart_path}")
    logger.info(f"Stroke dataset: {stroke_path}")

    # Diabetes Dataset Preprocessing
    logger.info("Preprocessing Diabetes Dataset...")
    X_train , X_test, y_train, y_test = preprocess_diabetes_data()
    logger.info("Diabetes Data Preprocessing Complete.")

    # Preprocess Heart Disease Data
    logger.info("Preprocessing Heart Disease dataset...")
    preprocess_heart_data()

    # Stroke preprocessing will be added later
    logger.info("Preprocessing Stroke dataset...")
    preprocess_stroke_data()

    # Perform EDA on Diabetes data
    perform_eda_on_diabetes()

    #Train Diabetes Model
    train_diabetes_model()

    #Train Heart Model
    train_heart_model()

    #Train Stroke Model
    train_stroke_model()

    #Tune Stroke Model
    tune_stroke_model()

if __name__ == "__main__":
    main()



     

