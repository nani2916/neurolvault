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


# Input model — what Salesforce sends to our API
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


# Output model — what our API sends back to Salesforce
class PredictionResponse(BaseModel):
    loan_id: Optional[str]
    default_probability: float
    risk_level: str
    recommendation: str
    top_risk_factors: list

@router.post("/predict", response_model=PredictionResponse)
async def predict_loan_default(application: LoanApplication):
    """
    Main prediction endpoint.
    Salesforce sends loan data → returns risk assessment.
    
    URL: POST /api/predictions/predict
    """
    try:
        # Convert to DataFrame
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

        # Encode categorical columns
        le = LabelEncoder()
        categorical = df.select_dtypes(include='object').columns
        for col in categorical:
            df[col] = le.fit_transform(df[col].astype(str))

        df = df.fillna(0)

        # Load model and predict
        model = joblib.load(MODEL_DIR / "xgboost_loan_default.pkl")
        probability = model.predict_proba(df.values)[0][1]

        # Determine risk level
        if probability >= 0.7:
            risk_level = "HIGH"
            recommendation = "REJECT — High probability of default"
        elif probability >= 0.3:
            risk_level = "MEDIUM"
            recommendation = "REVIEW — Manual assessment required"
        else:
            risk_level = "LOW"
            recommendation = "APPROVE — Low default risk"

        # Top risk factors based on feature importance
        feature_importance = pd.read_csv(
            MODEL_DIR / "feature_importance.csv"
        )
        top_factors = feature_importance.head(3)['feature'].tolist()

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
    """Health check endpoint — confirms API is running"""
    return {"status": "NeuralVault API is running", "version": "1.0"}
