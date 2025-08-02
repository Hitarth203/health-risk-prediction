import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from src.config import Config
from utils.logger import setup_logger
import os 

logger = setup_logger()

def perform_eda_on_diabetes():
    logger.info("Loading diabetes dataset for EDA...")
    df = pd.read_csv(Config.DIABETES_PATH)

    logger.info(f"Dataset Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info("First 5 rows:")
    logger.info(df.head())
    logger.info(df.describe())

    # Target variable distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Outcome')
    plt.title("Distribution of Outcome (0 = No Diabetes, 1 = Diabetes)")
    plt.savefig("outputs/diabetes_outcome_distribution.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("outputs/diabetes_correlation_heatmap.png")
    plt.close()

    # Missing values check
    missing = df.isnull().sum()
    logger.info("Missing values per column:")
    logger.info(missing[missing > 0])

    # Duplicates
    dup_count = df.duplicated().sum()
    logger.info(f"Number of duplicate rows: {dup_count}")
    logger.info("EDA completed. Visuals saved to outputs folder.")
