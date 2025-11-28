from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from models.diversion_detection import DiversionDetectionModel
from services.risk_calculator import calculate_diversion_risk
from utils.logging import logger

router = APIRouter()

# Define request/response models
class PatientData(BaseModel):
    patient_id: str
    age: int
    gender: str
    medications: List[Dict[str, Any]]
    prescriptions: List[Dict[str, Any]]
    clinical_indicators: Dict[str, Any]
    history: Dict[str, Any]

class RiskAssessmentRequest(BaseModel):
    patient_data: PatientData
    use_xai: bool = False

class RiskAssessmentResponse(BaseModel):
    patient_id: str
    risk_score: float
    risk_level: str
    red_flags: List[str]
    xai_explanation: Optional[Dict[str, Any]] = None
    timestamp: datetime

class BatchRiskAssessmentRequest(BaseModel):
    patients_data: List[PatientData]

class BatchRiskAssessmentResponse(BaseModel):
    results: List[RiskAssessmentResponse]

@router.post("/risk-assessment", response_model=RiskAssessmentResponse)
async def assess_diversion_risk(request: RiskAssessmentRequest):
    """
    Assess the risk of medication diversion for a single patient
    """
    try:
        logger.info(f"Assessing diversion risk for patient {request.patient_data.patient_id}")
        
        # Calculate risk score
        risk_score, red_flags = calculate_diversion_risk(
            patient_data=request.patient_data
        )
        
        # Determine risk level based on score
        if risk_score >= 0.8:
            risk_level = "HIGH"
        elif risk_score >= 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Prepare response
        response = RiskAssessmentResponse(
            patient_id=request.patient_data.patient_id,
            risk_score=risk_score,
            risk_level=risk_level,
            red_flags=red_flags,
            timestamp=datetime.now()
        )
        
        # Add XAI explanation if requested
        if request.use_xai:
            from services.xai_service import generate_xai_explanation
            response.xai_explanation = generate_xai_explanation(
                patient_data=request.patient_data,
                risk_score=risk_score
            )
        
        logger.info(f"Completed risk assessment for patient {request.patient_data.patient_id}, score: {risk_score}")
        return response
        
    except Exception as e:
        logger.error(f"Error in risk assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-risk-assessment", response_model=BatchRiskAssessmentResponse)
async def batch_assess_diversion_risk(request: BatchRiskAssessmentRequest):
    """
    Assess the risk of medication diversion for multiple patients
    """
    try:
        logger.info(f"Starting batch risk assessment for {len(request.patients_data)} patients")
        
        results = []
        for patient_data in request.patients_data:
            risk_score, red_flags = calculate_diversion_risk(patient_data=patient_data)
            
            # Determine risk level
            if risk_score >= 0.8:
                risk_level = "HIGH"
            elif risk_score >= 0.5:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
                
            result = RiskAssessmentResponse(
                patient_id=patient_data.patient_id,
                risk_score=risk_score,
                risk_level=risk_level,
                red_flags=red_flags,
                timestamp=datetime.now()
            )
            results.append(result)
        
        logger.info(f"Completed batch risk assessment for {len(results)} patients")
        return BatchRiskAssessmentResponse(results=results)
        
    except Exception as e:
        logger.error(f"Error in batch risk assessment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk-assessment/features", response_model=List[str])
async def get_risk_features():
    """
    Get the list of features used for risk assessment
    """
    try:
        # Return the feature names used in the model
        from models.diversion_detection import get_feature_names
        return get_feature_names()
    except Exception as e:
        logger.error(f"Error getting risk features: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))