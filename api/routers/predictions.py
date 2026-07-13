from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "ml_engine" / "models"


class LoanApplication(BaseModel):
    loan_amount: float
    rate_of_interest: float
    property_value: float
    income: float
    Credit_Score: int
    dtir1: float
    term: float
    loan_limit: Optional[str] = "cf"
    Gender: Optional[str] = "Male"
    loan_type: Optional[str] = "type1"
    loan_purpose: Optional[str] = "p1"
    Credit_Worthiness: Optional[str] = "l1"
    open_credit: Optional[str] = "nopc"
    business_or_commercial: Optional[str] = "nob/c"
    Neg_ammortization: Optional[str] = "not_neg"
    interest_only: Optional[str] = "not_int"
    lump_sum_payment: Optional[str] = "not_lpsm"
    construction_type: Optional[str] = "sb"
    occupancy_type: Optional[str] = "pr"
    Secured_by: Optional[str] = "home"
    total_units: Optional[str] = "1U"
    credit_type: Optional[str] = "EXP"
    co_applicant_credit_type: Optional[str] = "CIB"
    age: Optional[str] = "35-44"
    submission_of_application: Optional[str] = "to_inst"
    Region: Optional[str] = "North"
    Security_Type: Optional[str] = "direct"
    approv_in_adv: Optional[str] = "nopre"


class PredictionResponse(BaseModel):
    loan_id: Optional[str]
    default_probability: float
    risk_level: str
    recommendation: str
    top_risk_factors: list


# Readable names for features
FEATURE_LABELS = {
    'rate_of_interest': 'Interest rate is high for this profile',
    'credit_type': 'Credit history and type',
    'property_value': 'Property value as collateral',
    'loan_amount': 'Loan amount is large relative to profile',
    'income': 'Income level relative to the loan',
    'dtir1': 'Debt-to-income ratio',
    'Credit_Score': 'Credit score',
    'term': 'Loan term length',
    'Gender': 'Applicant profile',
    'Region': 'Region risk pattern',
    'loan_purpose': 'Purpose of the loan',
    'age': 'Age group'
}


@router.post("/predict", response_model=PredictionResponse)
async def predict_loan_default(application: LoanApplication):
    """Predict loan default and return personalized SHAP reasons."""
    try:
        data = {
            'loan_amount': application.loan_amount,
            'rate_of_interest': application.rate_of_interest,
            'property_value': application.property_value,
            'income': application.income,
            'Credit_Score': application.Credit_Score,
            'dtir1': application.dtir1,
            'term': application.term,
            'loan_limit': application.loan_limit,
            'Gender': application.Gender,
            'loan_type': application.loan_type,
            'loan_purpose': application.loan_purpose,
            'Credit_Worthiness': application.Credit_Worthiness,
            'open_credit': application.open_credit,
            'business_or_commercial': application.business_or_commercial,
            'Neg_ammortization': application.Neg_ammortization,
            'interest_only': application.interest_only,
            'lump_sum_payment': application.lump_sum_payment,
            'construction_type': application.construction_type,
            'occupancy_type': application.occupancy_type,
            'Secured_by': application.Secured_by,
            'total_units': application.total_units,
            'credit_type': application.credit_type,
            'co-applicant_credit_type': application.co_applicant_credit_type,
            'age': application.age,
            'submission_of_application': application.submission_of_application,
            'Region': application.Region,
            'Security_Type': application.Security_Type,
            'approv_in_adv': application.approv_in_adv
        }

        df = pd.DataFrame([data])

       # Fixed encodings matching training data
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

        for col, mapping in CATEGORY_MAPS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0)

        df = df.fillna(0)

        model = joblib.load(MODEL_DIR / "xgboost_loan_default.pkl")
        probability = float(model.predict_proba(df.values)[0][1])

        if probability >= 0.7:
            risk_level = "HIGH"
            recommendation = "REJECT — High probability of default"
        elif probability >= 0.3:
            risk_level = "MEDIUM"
            recommendation = "REVIEW — Manual assessment required"
        else:
            risk_level = "LOW"
            recommendation = "APPROVE — Low default risk"

        # Personalized reasons using SHAP for THIS applicant
        top_factors = []
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df.values)

            # Handle both array and list return formats
            vals = shap_values[0] if isinstance(shap_values, np.ndarray) else shap_values[0]

            feature_shap = pd.DataFrame({
                'feature': df.columns,
                'shap': np.array(vals).flatten()
            })

            # Factors pushing toward default = positive SHAP values
            pushing_up = feature_shap[feature_shap['shap'] > 0].sort_values(
                'shap', ascending=False
            )

            for feat in pushing_up['feature'].head(3):
                top_factors.append(FEATURE_LABELS.get(feat, feat))

        except Exception as shap_err:
            logger.warning(f"SHAP failed, using fallback: {shap_err}")

        if not top_factors:
            top_factors = ['Overall financial profile assessment']

        return PredictionResponse(
            loan_id=str(application.loan_amount),
            default_probability=round(probability, 4),
            risk_level=risk_level,
            recommendation=recommendation,
            top_risk_factors=top_factors
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    return {"status": "NeuralVault API is running", "version": "1.0"}