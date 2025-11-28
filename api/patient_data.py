from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from services.patient_service import PatientService
from utils.logging import logger

router = APIRouter()

class Patient(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    medications: List[Dict[str, Any]]
    prescriptions: List[Dict[str, Any]]
    clinical_indicators: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class PatientCreateRequest(BaseModel):
    name: str
    age: int
    gender: str
    medications: List[Dict[str, Any]]
    prescriptions: List[Dict[str, Any]]
    clinical_indicators: Dict[str, Any]

class PatientUpdateRequest(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    medications: Optional[List[Dict[str, Any]]] = None
    prescriptions: Optional[List[Dict[str, Any]]] = None
    clinical_indicators: Optional[Dict[str, Any]] = None

@router.get("/patients", response_model=List[Patient])
async def get_patients(skip: int = 0, limit: int = 100):
    """
    Get a list of patients
    """
    try:
        logger.info(f"Fetching patients with skip={skip}, limit={limit}")
        patient_service = PatientService()
        return patient_service.get_patients(skip, limit)
    except Exception as e:
        logger.error(f"Error fetching patients: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/patients/{patient_id}", response_model=Patient)
async def get_patient(patient_id: str):
    """
    Get a specific patient by ID
    """
    try:
        logger.info(f"Fetching patient with ID: {patient_id}")
        patient_service = PatientService()
        patient = patient_service.get_patient_by_id(patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        return patient
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching patient {patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/patients", response_model=Patient)
async def create_patient(request: PatientCreateRequest):
    """
    Create a new patient
    """
    try:
        logger.info(f"Creating new patient: {request.name}")
        patient_service = PatientService()
        patient = patient_service.create_patient(request)
        return patient
    except Exception as e:
        logger.error(f"Error creating patient: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/patients/{patient_id}", response_model=Patient)
async def update_patient(patient_id: str, request: PatientUpdateRequest):
    """
    Update a patient's information
    """
    try:
        logger.info(f"Updating patient with ID: {patient_id}")
        patient_service = PatientService()
        updated_patient = patient_service.update_patient(patient_id, request)
        if not updated_patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        return updated_patient
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating patient {patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/patients/{patient_id}")
async def delete_patient(patient_id: str):
    """
    Delete a patient
    """
    try:
        logger.info(f"Deleting patient with ID: {patient_id}")
        patient_service = PatientService()
        success = patient_service.delete_patient(patient_id)
        if not success:
            raise HTTPException(status_code=404, detail="Patient not found")
        return {"message": "Patient deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting patient {patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/patients/{patient_id}/risk-history")
async def get_patient_risk_history(patient_id: str):
    """
    Get the risk assessment history for a specific patient
    """
    try:
        logger.info(f"Fetching risk history for patient: {patient_id}")
        risk_history = PatientService().get_patient_risk_history(patient_id)
        return risk_history
    except Exception as e:
        logger.error(f"Error fetching risk history for patient {patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))