from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))


class OrchestratorRequest(BaseModel):
    loan_amount: float
    rate_of_interest: float
    property_value: float
    income: float
    Credit_Score: int
    dtir1: float
    term: float


class OrchestratorResponse(BaseModel):
    risk_level: str
    default_probability: float
    reasons: list
    fairness_note: str
    summary: str


@router.post("/analyze", response_model=OrchestratorResponse)
async def analyze_applicant(req: OrchestratorRequest):
    """Runs all agents together on one applicant."""
    try:
        from agents.orchestrator import run_orchestrator

        applicant = {
            'loan_amount': req.loan_amount,
            'rate_of_interest': req.rate_of_interest,
            'property_value': req.property_value,
            'income': req.income,
            'Credit_Score': req.Credit_Score,
            'dtir1': req.dtir1,
            'term': req.term,
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

        result = run_orchestrator(applicant)
        return OrchestratorResponse(**result)

    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def orchestrator_health():
    return {"status": "Orchestrator running"}