import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate
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


def run_compliance_check():
    """
    Runs full bias and compliance check on model predictions.
    Generates regulatory-ready compliance report.
    """
    logger.info("Compliance Agent starting...")

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

    # Save sensitive columns before encoding
    gender_raw = df['Gender'].copy()
    age_raw = df['age'].copy()
    region_raw = df['Region'].copy()

    df_features = df[feature_cols].copy()
    le = LabelEncoder()
    categorical = df_features.select_dtypes(include='object').columns
    for col in categorical:
        df_features[col] = le.fit_transform(df_features[col].astype(str))
    df_features = df_features.fillna(0)

    model = joblib.load(MODEL_DIR / "xgboost_loan_default.pkl")
    y_true = df['Status'].values
    y_pred = model.predict(df_features.values)

    # Run bias checks
    compliance_results = {}
    for group_name, group_values in [
        ('Gender', gender_raw),
        ('Age', age_raw),
        ('Region', region_raw)
    ]:
        mf = MetricFrame(
            metrics={
                'accuracy': accuracy_score,
                'selection_rate': selection_rate,
                'false_positive_rate': false_positive_rate
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=group_values
        )

        selection_rates = mf.by_group['selection_rate']
        max_diff = selection_rates.max() - selection_rates.min()
        compliant = max_diff <= 0.10

        compliance_results[group_name] = {
            'compliant': compliant,
            'max_disparity': max_diff,
            'by_group': mf.by_group
        }

        status = "COMPLIANT" if compliant else "NON-COMPLIANT"
        logger.info(f"{group_name}: {status} (disparity: {max_diff:.2%})")

    return compliance_results, y_true, y_pred


def generate_compliance_report(compliance_results):
    """Generate AI-written compliance report using Groq"""

    summary = ""
    for group, result in compliance_results.items():
        status = "COMPLIANT" if result['compliant'] else "NON-COMPLIANT"
        summary += f"{group}: {status} - Max disparity: {result['max_disparity']:.2%}\n"

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are NeuralVault AI compliance officer. Write regulatory compliance reports."
            },
            {
                "role": "user",
                "content": f"Write a regulatory compliance report based on these AI fairness metrics:\n{summary}"
            }
        ],
        temperature=0.3,
        max_tokens=400
    )

    report = response.choices[0].message.content

    report_path = REPORTS_DIR / "compliance_agent_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("NEUROLVAULT AI COMPLIANCE REPORT\n")
        f.write("="*60 + "\n")
        f.write(summary + "\n")
        f.write("="*60 + "\n")
        f.write(report)

    logger.info(f"Compliance report saved to: {report_path}")

    print("\n" + "="*60)
    print("NEUROLVAULT AI COMPLIANCE REPORT")
    print("="*60)
    print(summary)
    print("="*60)
    print(report)
    print("="*60)

    return report


if __name__ == "__main__":
    compliance_results, y_true, y_pred = run_compliance_check()
    generate_compliance_report(compliance_results) 
