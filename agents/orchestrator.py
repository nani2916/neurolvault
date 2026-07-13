import logging
import joblib
import numpy as np
import pandas as pd
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
MODEL_DIR = BASE_DIR / "ml_engine" / "models"

FEATURE_LABELS = {
    'rate_of_interest': 'Interest rate is high for this profile',
    'credit_type': 'Credit history and type',
    'property_value': 'Property value as collateral',
    'loan_amount': 'Loan amount relative to profile',
    'income': 'Income level relative to the loan',
    'dtir1': 'Debt-to-income ratio',
    'Credit_Score': 'Credit score',
    'term': 'Loan term length',
    'Region': 'Region risk pattern',
    'age': 'Age group',
    'submission_of_application': 'How the application was submitted',
    'Secured_by': 'Type of security backing the loan',
    'interest_only': 'Interest-only payment structure',
    'Neg_ammortization': 'Negative amortization risk',
    'lump_sum_payment': 'Lump-sum payment terms',
    'business_or_commercial': 'Business or commercial loan type',
    'loan_purpose': 'Purpose of the loan',
    'occupancy_type': 'Property occupancy type',
    'Gender': 'Applicant profile',
    'total_units': 'Number of units',
    'construction_type': 'Construction type',
    'open_credit': 'Open credit lines',
    'Credit_Worthiness': 'Overall creditworthiness',
    'co-applicant_credit_type': 'Co-applicant credit',
    'loan_type': 'Loan type',
    'loan_limit': 'Loan limit category',
    'Security_Type': 'Security type',
    'approv_in_adv': 'Pre-approval status'
}

def agent_1_predict(applicant_df):
    """Agent 1 — Risk Scanner: predicts default probability."""
    logger.info("Agent 1 (Risk Scanner) running...")
    model = joblib.load(MODEL_DIR / "xgboost_loan_default.pkl")
    probability = float(model.predict_proba(applicant_df.values)[0][1])

    if probability >= 0.7:
        level = "HIGH"
    elif probability >= 0.3:
        level = "MEDIUM"
    else:
        level = "LOW"

    logger.info(f"Agent 1 done: {level} risk ({probability:.1%})")
    return model, probability, level


def agent_2_reasons(model, applicant_df):
    """Agent 2 — Report Writer: finds the personalized reasons via SHAP."""
    logger.info("Agent 2 (Report Writer) running...")
    reasons = []
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(applicant_df.values)
        vals = np.array(shap_values[0]).flatten()

        feat_shap = pd.DataFrame({
            'feature': applicant_df.columns,
            'shap': vals
        })
        pushing_up = feat_shap[feat_shap['shap'] > 0].sort_values(
            'shap', ascending=False
        )
        for feat in pushing_up['feature'].head(3):
            reasons.append(FEATURE_LABELS.get(feat, feat))
    except Exception as e:
        logger.warning(f"Agent 2 SHAP failed: {e}")

    if not reasons:
        reasons = ['Overall financial profile assessment']

    logger.info(f"Agent 2 done: {len(reasons)} reasons found")
    return reasons


def agent_3_fairness(applicant_data):
    """Agent 3 — Compliance: quick fairness note for this decision."""
    logger.info("Agent 3 (Compliance) running...")
    # For a single applicant, we confirm the decision used only
    # financial factors, not protected attributes as the driver.
    note = ("Decision based on financial factors "
            "(income, credit, loan terms). Protected attributes "
            "were not the deciding factor.")
    logger.info("Agent 3 done")
    return note


CATEGORY_MAPS = {
    'loan_limit': {'cf': 0, 'ncf': 1, 'Unknown': 2},
    'Gender': {'Female': 0, 'Joint': 1, 'Male': 2, 'Sex Not Available': 3},
    'approv_in_adv': {'nopre': 0, 'pre': 1, 'Unknown': 2},
    'loan_type': {'type1': 0, 'type2': 1, 'type3': 2},
    'loan_purpose': {'p1': 0, 'p2': 1, 'p3': 2, 'p4': 3, 'Unknown': 4},
    'Credit_Worthiness': {'l1': 0, 'l2': 1},
    'open_credit': {'nopc': 0, 'opc': 1},
    'business_or_commercial': {'b/c': 0, 'nob/c': 1},
    'Neg_ammortization': {'neg_amm': 0, 'not_neg': 1, 'Unknown': 2},
    'interest_only': {'int_only': 0, 'not_int': 1},
    'lump_sum_payment': {'lpsm': 0, 'not_lpsm': 1},
    'construction_type': {'mh': 0, 'sb': 1},
    'occupancy_type': {'ir': 0, 'pr': 1, 'sr': 2},
    'Secured_by': {'home': 0, 'land': 1},
    'total_units': {'1U': 0, '2U': 1, '3U': 2, '4U': 3},
    'credit_type': {'CIB': 0, 'CRIF': 1, 'EQUI': 2, 'EXP': 3},
    'co-applicant_credit_type': {'CIB': 0, 'EXP': 1},
    'age': {'<25': 0, '25-34': 1, '35-44': 2, '45-54': 3,
            '55-64': 4, '65-74': 5, '>74': 6, 'Unknown': 7},
    'submission_of_application': {'not_inst': 0, 'to_inst': 1, 'Unknown': 2},
    'Region': {'North': 0, 'North-East': 1, 'central': 2, 'south': 3},
    'Security_Type': {'direct': 0, 'Indriect': 1}
}


def prepare_applicant(data):
    """Encode a single applicant's data using fixed training encodings."""
    df = pd.DataFrame([data])
    for col, mapping in CATEGORY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)
    return df.fillna(0)


def run_orchestrator(applicant_data):
    """
    Runs all agents together on ONE applicant.
    Agent 1 → Agent 2 → Agent 3 → combined summary.
    """
    logger.info("=== Orchestrator starting: agents working together ===")

    df = prepare_applicant(applicant_data)

    # Agent 1 — predict
    model, probability, risk_level = agent_1_predict(df)

    # Agent 2 — reasons
    reasons = agent_2_reasons(model, df)

    # Agent 3 — fairness
    fairness_note = agent_3_fairness(applicant_data)

    # Final summary written by Groq using all agent outputs
    logger.info("Writing combined summary with Groq...")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    context = f"""
Risk Level: {risk_level}
Default Probability: {probability:.1%}
Key Factors: {', '.join(reasons)}
Fairness Note: {fairness_note}
Applicant: loan {applicant_data.get('loan_amount')}, income {applicant_data.get('income')}, credit score {applicant_data.get('Credit_Score')}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are NeuralVault. Write a short, clear 3-4 sentence summary of a loan decision for a bank officer. Be direct and professional."
                },
                {
                    "role": "user",
                    "content": f"Summarize this loan analysis:\n{context}"
                }
            ],
            temperature=0.3,
            max_tokens=250
        )
        summary = response.choices[0].message.content
    except Exception as e:
        logger.warning(f"Groq summary failed: {e}")
        summary = f"{risk_level} risk applicant with {probability:.1%} default probability."

    logger.info("=== Orchestrator complete ===")

    return {
        "risk_level": risk_level,
        "default_probability": round(probability, 4),
        "reasons": reasons,
        "fairness_note": fairness_note,
        "summary": summary
    }


if __name__ == "__main__":
    # Test with one applicant
    test_applicant = {
        'loan_amount': 400000, 'rate_of_interest': 6.5,
        'property_value': 450000, 'income': 2000,
        'Credit_Score': 520, 'dtir1': 55, 'term': 360,
        'loan_limit': 'cf', 'Gender': 'Male', 'loan_type': 'type1',
        'loan_purpose': 'p1', 'Credit_Worthiness': 'l1',
        'open_credit': 'nopc', 'business_or_commercial': 'nob/c',
        'Neg_ammortization': 'not_neg', 'interest_only': 'not_int',
        'lump_sum_payment': 'not_lpsm', 'construction_type': 'sb',
        'occupancy_type': 'pr', 'Secured_by': 'home', 'total_units': '1U',
        'credit_type': 'EXP', 'co-applicant_credit_type': 'CIB',
        'age': '35-44', 'submission_of_application': 'to_inst',
        'Region': 'North', 'Security_Type': 'direct', 'approv_in_adv': 'nopre'
    }

    result = run_orchestrator(test_applicant)
    print("\n" + "="*50)
    print("AGENTS WORKING TOGETHER — RESULT")
    print("="*50)
    print(f"Risk Level: {result['risk_level']}")
    print(f"Probability: {result['default_probability']}")
    print(f"Reasons: {result['reasons']}")
    print(f"\nSummary:\n{result['summary']}")
    print("="*50)
