import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "ml_engine" / "models"
CLEANED_DATA_PATH = DATA_DIR / "Loan_Default_Cleaned.csv"

logger.info(f"Data path: {CLEANED_DATA_PATH}")
logger.info(f"Model path: {MODEL_DIR}") 
def load_and_prepare(path):
    """Load data and prepare for anomaly detection"""
    logger.info("Loading data...")
    df = pd.read_csv(path)

    # Key numerical features that define normal financial behavior
    feature_cols = [
        'loan_amount', 'rate_of_interest', 'property_value',
        'income', 'Credit_Score', 'dtir1', 'term'
    ]

    df = df[feature_cols].dropna()

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    # Save scaler
    joblib.dump(scaler, MODEL_DIR / "anomaly_scaler.pkl")
    logger.info(f"Scaler saved")

    logger.info(f"Data shape: {X.shape}")
    return X, df, scaler, feature_cols

def train_anomaly_detector(X):
    """Train Isolation Forest for anomaly detection"""
    logger.info("Training Isolation Forest...")

    model = IsolationForest(
        n_estimators=100,      # 100 isolation trees
        contamination=0.05,    # expect 5% anomalies
        random_state=42,
        n_jobs=-1              # use all CPU cores
    )

    model.fit(X)

    # Predict — Isolation Forest returns:
    # 1  = normal
    # -1 = anomaly
    predictions = model.predict(X)
    scores = model.decision_function(X)

    normal_count = (predictions == 1).sum()
    anomaly_count = (predictions == -1).sum()

    logger.info(f"Normal transactions: {normal_count}")
    logger.info(f"Anomalies detected: {anomaly_count}")
    logger.info(f"Anomaly rate: {anomaly_count/len(predictions)*100:.2f}%")

    # Save model
    model_path = MODEL_DIR / "isolation_forest.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to: {model_path}")

    return model, predictions, scores
def analyze_anomalies(df, predictions, scores, feature_cols):
    """Analyze and explain detected anomalies"""

    results = df.copy()
    results['anomaly_score'] = scores
    results['is_anomaly'] = predictions == -1

    # Get top 10 most suspicious cases
    anomalies = results[results['is_anomaly'] == True]
    top_anomalies = anomalies.nsmallest(10, 'anomaly_score')

    logger.info(f"\nTop 10 Most Suspicious Applications:")
    print(top_anomalies.to_string())

    # Explain what makes each anomaly suspicious
    normal_stats = results[results['is_anomaly'] == False][feature_cols].describe()

    logger.info("\n--- Anomaly Profile vs Normal Profile ---")
    anomaly_stats = anomalies[feature_cols].mean()
    normal_means = results[results['is_anomaly'] == False][feature_cols].mean()

    comparison = pd.DataFrame({
        'Normal_Mean': normal_means,
        'Anomaly_Mean': anomaly_stats,
        'Difference_%': ((anomaly_stats - normal_means) / normal_means * 100).round(2)
    })

    print(comparison)

    # Save anomaly report
    report_path = MODEL_DIR / "anomaly_report.csv"
    results[results['is_anomaly'] == True].to_csv(report_path, index=False)
    logger.info(f"\nAnomaly report saved to: {report_path}")

    return results, anomalies

if __name__ == "__main__":
    X, df, scaler, feature_cols = load_and_prepare(CLEANED_DATA_PATH)
    model, predictions, scores = train_anomaly_detector(X)
    results, anomalies = analyze_anomalies(df, predictions, scores, feature_cols)