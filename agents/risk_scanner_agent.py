import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "ml_engine" / "models"
REPORTS_DIR = BASE_DIR / "agents"
CLEANED_DATA_PATH = DATA_DIR / "Loan_Default_Cleaned.csv"


def scan_high_risk_applications(threshold=0.7):
    """
    Scans all loan applications and flags high risk ones.
    threshold=0.7 means flag anyone with 70%+ default probability
    """
    logger.info("Risk Scanner Agent starting...")
    logger.info(f"Flagging applications with default probability > {threshold*100}%")

    # Load data
    df = pd.read_csv(CLEANED_DATA_PATH)

    feature_cols = [
        'loan_amount', 'rate_of_interest', 'property_value',
        'income', 'Credit_Score', 'dtir1', 'term',
        'loan_limit', 'Gender', 'loan_type', 'loan_purpose',
        'Credit_Worthiness', 'open_credit', 'business_or_commercial',
        'Neg_ammortization', 'interest_only', 'lump_sum_payment',
        'construction_type', 'occupancy_type', 'Secured_by',
        'total_units', 'credit_type', 'co-applicant_credit_type',
        'age', 'submission_of_application', 'Region',
        'Security_Type', 'approv_in_adv'
    ]

    df_features = df[feature_cols].copy()

    # Encode categorical columns
    le = LabelEncoder()
    categorical = df_features.select_dtypes(include='object').columns
    for col in categorical:
        df_features[col] = le.fit_transform(
            df_features[col].astype(str)
        )

    df_features = df_features.fillna(0)

    # Load XGBoost model
    model = joblib.load(MODEL_DIR / "xgboost_loan_default.pkl")

    # Get default probabilities
    probabilities = model.predict_proba(df_features.values)[:, 1]

    # Flag high risk
    df['default_probability'] = probabilities
    df['risk_level'] = pd.cut(
        probabilities,
        bins=[0, 0.3, 0.7, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH']
    )

    high_risk = df[df['default_probability'] >= threshold].copy()

    logger.info(f"Total applications scanned: {len(df)}")
    logger.info(f"High risk applications found: {len(high_risk)}")
    logger.info(f"Risk distribution:\n{df['risk_level'].value_counts()}")

    # Save high risk report
    report_path = REPORTS_DIR / "high_risk_report.csv"
    high_risk.to_csv(report_path, index=False)
    logger.info(f"High risk report saved to: {report_path}")

    return high_risk, df


def generate_risk_summary(high_risk_df):
    """
    Uses Groq LLM to generate a human-readable
    summary of all high risk applications found
    """
    logger.info("Generating risk summary with Groq...")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Prepare summary stats
    summary = f"""
Total High Risk Applications: {len(high_risk_df)}
Average Default Probability: {high_risk_df['default_probability'].mean():.1%}
Regions Affected: {high_risk_df['Region'].value_counts().to_dict()}
Average Loan Amount: ${high_risk_df['loan_amount'].mean():,.0f}
Average Income: ${high_risk_df['income'].mean():,.0f}
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are NeuralVault AI risk analyst. Generate a concise executive summary."
            },
            {
                "role": "user",
                "content": f"Generate an executive risk summary for bank management based on this data:\n{summary}"
            }
        ],
        temperature=0.3,
        max_tokens=300
    )

    summary_text = response.choices[0].message.content
    logger.info("Risk summary generated")
    return summary_text


if __name__ == "__main__":
    high_risk, full_df = scan_high_risk_applications(threshold=0.7)
    summary = generate_risk_summary(high_risk)
    print("\n" + "="*60)
    print("NEUROLVAULT RISK SCANNER AGENT REPORT")
    print("="*60)
    print(summary)
    print("="*60) 
