import numpy as np
import pandas as pd
import logging
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate,
    false_negative_rate
)
from sklearn.metrics import accuracy_score
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

def load_data_and_model():
    """Load cleaned data and trained XGBoost model"""
    logger.info("Loading data and model...")

    df = pd.read_csv(CLEANED_DATA_PATH)

    # Keep sensitive columns before encoding
    sensitive_cols = ['Gender', 'age', 'Region']

    # Features used in Week 2 training
    feature_cols = [
        'loan_amount', 'rate_of_interest', 'property_value',
        'income', 'Credit_Score', 'dtir1', 'term',
        'loan_limit', 'Gender', 'loan_type', 'loan_purpose',
        'Credit_Worthiness', 'open_credit', 'business_or_commercial',
        'Neg_ammortization', 'interest_only', 'lump_sum_payment',
        'construction_type', 'occupancy_type', 'Secured_by',
        'total_units', 'credit_type', 'co-applicant_credit_type',
        'age', 'submission_of_application', 'Region', 'Security_Type',
        'approv_in_adv'
    ]
    target_col = 'Status'

    df = df[feature_cols + [target_col]].dropna()

    # Save sensitive column values before encoding
    gender_raw = df['Gender'].copy()
    age_raw = df['age'].copy()
    region_raw = df['Region'].copy()

    # Encode all categorical columns
    le = LabelEncoder()
    categorical = df[feature_cols].select_dtypes(include='object').columns
    for col in categorical:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df[feature_cols].values
    y = df[target_col].values

    # Load trained XGBoost model
    model = joblib.load(MODEL_DIR / "xgboost_loan_default.pkl")
    logger.info("Model loaded successfully")

    # Make predictions
    y_pred = model.predict(X)
    logger.info(f"Predictions made on {len(y_pred)} records")

    return y, y_pred, gender_raw, age_raw, region_raw

def detect_bias(y, y_pred, gender_raw, age_raw, region_raw):
    """Detect bias across sensitive groups"""

    results = {}

    for group_name, group_values in [
        ('Gender', gender_raw),
        ('Age Group', age_raw),
        ('Region', region_raw)
    ]:
        logger.info(f"\n--- Bias Analysis: {group_name} ---")

        mf = MetricFrame(
            metrics={
                'accuracy': accuracy_score,
                'selection_rate': selection_rate,
                'false_positive_rate': false_positive_rate,
                'false_negative_rate': false_negative_rate
            },
            y_true=y,
            y_pred=y_pred,
            sensitive_features=group_values
        )

        print(f"\n{group_name} Overall Metrics:")
        print(mf.overall)

        print(f"\n{group_name} Metrics by Group:")
        print(mf.by_group)

        # Flag if difference between best and worst group > 10%
        selection_rates = mf.by_group['selection_rate']
        max_diff = selection_rates.max() - selection_rates.min()

        if max_diff > 0.10:
            logger.warning(
                f"BIAS DETECTED in {group_name}! "
                f"Selection rate difference: {max_diff:.2%}"
            )
        else:
            logger.info(
                f"No significant bias in {group_name}. "
                f"Max difference: {max_diff:.2%}"
            )

        results[group_name] = mf

    return results

def save_compliance_report(results):
    """Save bias detection results as compliance report"""

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("NEUROLVAULT AI BIAS & COMPLIANCE REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Model: XGBoost Loan Default Predictor")
    report_lines.append(f"Records Analyzed: 148,629")
    report_lines.append("")

    for group_name, mf in results.items():
        report_lines.append(f"\n--- {group_name} Fairness Analysis ---")
        report_lines.append(mf.by_group.to_string())

        selection_rates = mf.by_group['selection_rate']
        max_diff = selection_rates.max() - selection_rates.min()

        if max_diff > 0.10:
            report_lines.append(
                f"\n⚠️  BIAS ALERT: {group_name} shows "
                f"{max_diff:.2%} selection rate disparity"
            )
        else:
            report_lines.append(
                f"\n✅ COMPLIANT: {group_name} within "
                f"acceptable range ({max_diff:.2%} disparity)"
            )

    report_path = MODEL_DIR / "compliance_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Compliance report saved to: {report_path}")
    print('\n'.join(report_lines))

if __name__ == "__main__":
    y, y_pred, gender_raw, age_raw, region_raw = load_data_and_model()
    results = detect_bias(y, y_pred, gender_raw, age_raw, region_raw)
    save_compliance_report(results)