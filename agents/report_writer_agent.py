import logging
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from groq import Groq
import os
import faiss
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
VECTOR_STORE_DIR = BASE_DIR / "rag_system" / "vector_store"
REPORTS_DIR = BASE_DIR / "agents"
CLEANED_DATA_PATH = DATA_DIR / "Loan_Default_Cleaned.csv"


def get_customer_prediction(customer_data):
    """Get XGBoost prediction for a specific customer"""
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

    le = LabelEncoder()
    df = customer_data[feature_cols].copy()
    categorical = df.select_dtypes(include='object').columns
    for col in categorical:
        df[col] = le.fit_transform(df[col].astype(str))
    df = df.fillna(0)

    model = joblib.load(MODEL_DIR / "xgboost_loan_default.pkl")
    probability = model.predict_proba(df.values)[0][1]
    prediction = "HIGH RISK" if probability >= 0.7 else \
                 "MEDIUM RISK" if probability >= 0.3 else "LOW RISK"

    return probability, prediction


def generate_customer_report(loan_id):
    """
    Generate full risk report for a specific customer.
    Uses RAG to find customer data + Groq to write report.
    """
    logger.info(f"Generating report for Loan ID: {loan_id}")

    # Load data and find customer
    df = pd.read_csv(CLEANED_DATA_PATH)
    customer = df[df['ID'] == loan_id]

    if customer.empty:
        logger.warning(f"Loan ID {loan_id} not found")
        return None

    customer_info = customer.iloc[0]

    # Get ML prediction
    probability, risk_level = get_customer_prediction(customer)

    # Build customer profile
    profile = f"""
Loan ID: {loan_id}
Gender: {customer_info.get('Gender', 'N/A')}
Age Group: {customer_info.get('age', 'N/A')}
Region: {customer_info.get('Region', 'N/A')}
Loan Amount: ${customer_info.get('loan_amount', 0):,.0f}
Rate of Interest: {customer_info.get('rate_of_interest', 0)}%
Property Value: ${customer_info.get('property_value', 0):,.0f}
Annual Income: ${customer_info.get('income', 0):,.0f}
Credit Score: {customer_info.get('Credit_Score', 0)}
Debt-to-Income Ratio: {customer_info.get('dtir1', 0)}%
Loan Term: {customer_info.get('term', 0)} months
ML Risk Assessment: {risk_level}
Default Probability: {probability:.1%}
Actual Status: {'DEFAULT' if customer_info.get('Status', 0) == 1 else 'NO DEFAULT'}
    """

    # Generate report with Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are NeuralVault AI. Write professional bank risk reports."
            },
            {
                "role": "user",
                "content": f"""Write a professional loan risk assessment report 
for the following customer. Include risk factors, 
recommendations, and next steps.\n\n{profile}"""
            }
        ],
        temperature=0.3,
        max_tokens=500
    )

    report = response.choices[0].message.content

    # Save report
    report_path = REPORTS_DIR / f"report_loan_{loan_id}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"NEUROLVAULT RISK REPORT - LOAN ID: {loan_id}\n")
        f.write("="*60 + "\n")
        f.write(profile + "\n")
        f.write("="*60 + "\n")
        f.write(report)

    logger.info(f"Report saved to: {report_path}")
    print("\n" + "="*60)
    print(f"NEUROLVAULT RISK REPORT - LOAN ID: {loan_id}")
    print("="*60)
    print(profile)
    print("="*60)
    print(report)
    print("="*60)

    return report


if __name__ == "__main__":
    # Generate report for first 3 loan IDs
    df = pd.read_csv(CLEANED_DATA_PATH)
    sample_ids = df['ID'].head(3).tolist()

    for loan_id in sample_ids:
        generate_customer_report(loan_id)
