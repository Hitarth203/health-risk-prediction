import pandas as pd
from sklearn.model_selection import train_test_split
from utils.logger import setup_logger
from src.config import Config
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

logger = setup_logger()

def preprocess_diabetes_data():
    logger.info("Loading diabetes dataset...")
    df = pd.read_csv(Config.DIABETES_PATH)
    logger.info(f"Initial Shape:{df.shape}")

    # Split features and target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Train-test split
    logger.info("Splitting dataset into train and test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        stratify=y, random_state=42)
    # Save processed data (optional)
    processed_dir = Config.DATA_PROCESSED_DIR
    os.makedirs(processed_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(processed_dir, "diabetes_X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "diabetes_X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "diabetes_y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "diabetes_y_test.csv"), index=False)

    logger.info("Preprocessing complete for diabetes dataset.")
    return X_train, X_test, y_train, y_test

def preprocess_heart_data():
    df = pd.read_csv(Config.HEART_PATH)
    logger.info("Loaded Heart Disease dataset successfully.")
    # Check for nulls
    if df.isnull().sum().sum() > 0:
        logger.warning("Missing values found. Dropping rows with missing values.")
        df.dropna(inplace=True)

    # Split features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    logger.info("Splitting dataset into train and test...")
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.2, stratify=y, random_state=42)
    # Apply SMOTE to handle class imbalance
    logger.info("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Save processed data
    processed_dir = Config.DATA_PROCESSED_DIR
    os.makedirs(processed_dir, exist_ok=True)

    pd.DataFrame(X_train_res).to_csv(os.path.join(processed_dir, "heart_X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(processed_dir, "heart_X_test.csv"), index=False)
    pd.DataFrame(y_train_res).to_csv(os.path.join(processed_dir, "heart_y_train.csv"), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(processed_dir, "heart_y_test.csv"), index=False)

    logger.info("Preprocessing complete for heart disease dataset.")
    return X_train_res, X_test, y_train_res, y_test

def preprocess_stroke_data():
    try:
        df = pd.read_csv(Config.STROKE_PATH)
        logger.info("Loaded Stroke dataset successfully.")
        logger.info(f"Initial shape: {df.shape}")
        # Drop ID column if present
        if "id" in df.columns:
            df.drop("id",axis=1, inplace=True)

        # Handle missing values
        if df.isnull().sum().sum() > 0:
            df.dropna(inplace=True)

        # Encode categorical variables
        cat_cols = df.select_dtypes(include="object").columns
        for col in cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col])
        
        # Separate features and target
        X = df.drop("stroke", axis=1)
        y = df["stroke"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,
                                                            stratify=y, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Save processed data
        processed_dir = Config.DATA_PROCESSED_DIR
        os.makedirs(processed_dir, exist_ok=True)
        pd.DataFrame(X_train).to_csv(os.path.join(processed_dir, "stroke_X_train.csv"), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(processed_dir, "stroke_X_test.csv"), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(processed_dir, "stroke_y_train.csv"), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(processed_dir, "stroke_y_test.csv"), index=False)


        logger.info("Preprocessing complete for Stroke dataset.")
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Error while preprocessing Stroke dataset: {e}")
        raise

