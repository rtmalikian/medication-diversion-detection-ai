import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from datetime import datetime
import json

from models.diversion_detection import diversion_model
from services.evaluation_framework import evaluation_framework
from services.xai_service import xai_service
from services.database import db_manager
from data.processor import data_processor

# Configure logging
logger = logging.getLogger(__name__)

class RiskCalculatorService:
    """
    Service for calculating medication diversion risk
    """
    
    def __init__(self):
        self.model = diversion_model
        self.evaluation = evaluation_framework
        self.xai_service = xai_service
        
        # Initialize model if not already done
        if not self.model.is_trained:
            logger.info("Initializing model...")
            self.model.load_model()
    
    def calculate_diversion_risk(self, patient_data: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Calculate the risk of medication diversion for a patient
        Returns: (risk_score, red_flags)
        """
        try:
            logger.info(f"Calculating diversion risk for patient {patient_data.get('patient_id', 'unknown')}")
            
            # Get evidence-based evaluation
            evaluation_result = self.evaluation.evaluate_patient_risk(patient_data)
            red_flags = evaluation_result.get('red_flags', [])
            
            # Process patient data for ML model
            features_df = data_processor.process_patient_data(patient_data)
            features_dict = features_df.iloc[0].to_dict()
            
            # Get ML prediction
            ml_prediction, ml_probability = self.model.predict_single(features_dict)
            
            # Combine evidence-based evaluation with ML prediction
            risk_score = self._combine_scores(
                ml_probability, 
                evaluation_result['overall_risk_level'], 
                len(red_flags)
            )
            
            # Log the risk calculation details
            logger.info(f"Risk calculation for {patient_data.get('patient_id', 'unknown')}: "
                       f"ML probability: {ml_probability:.4f}, "
                       f"Combined risk score: {risk_score:.4f}, "
                       f"Red flags: {len(red_flags)}")
            
            return risk_score, [flag['description'] for flag in red_flags]
            
        except Exception as e:
            logger.error(f"Error calculating diversion risk: {str(e)}")
            raise
    
    def _combine_scores(self, ml_probability: float, risk_level: str, red_flag_count: int) -> float:
        """
        Combine ML probability with evidence-based risk level and red flag count
        """
        try:
            # Weight the different components
            ml_weight = 0.6
            risk_level_weight = 0.3
            red_flags_weight = 0.1
            
            # Convert risk level to score (LOW=0.1, MEDIUM=0.4, HIGH=0.7, CRITICAL=0.9)
            level_scores = {
                'LOW': 0.1,
                'MEDIUM': 0.4,
                'HIGH': 0.7,
                'CRITICAL': 0.9
            }
            risk_level_score = level_scores.get(risk_level, 0.1)
            
            # Calculate red flags contribution (saturate after 5 flags)
            red_flags_score = min(red_flag_count / 10.0, 0.5)  # Max contribution of 0.5
            
            # Combine scores
            combined_score = (
                ml_probability * ml_weight +
                risk_level_score * risk_level_weight +
                red_flags_score * red_flags_weight
            )
            
            # Ensure score is between 0 and 1
            return min(max(combined_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error combining scores: {str(e)}")
            # Return ML probability as fallback
            return ml_probability
    
    def batch_calculate_risk(self, patients_data: List[Dict[str, Any]]) -> List[Tuple[str, float, List[str]]]:
        """
        Calculate risk for multiple patients
        Returns: List of (patient_id, risk_score, red_flags)
        """
        try:
            logger.info(f"Calculating diversion risk for {len(patients_data)} patients")
            
            results = []
            for patient_data in patients_data:
                risk_score, red_flags = self.calculate_diversion_risk(patient_data)
                patient_id = patient_data.get('patient_id', 'unknown')
                results.append((patient_id, risk_score, red_flags))
            
            logger.info(f"Batch risk calculation completed for {len(results)} patients")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch risk calculation: {str(e)}")
            raise
    
    def get_risk_features(self) -> List[str]:
        """
        Get the list of features used for risk assessment
        """
        try:
            return self.model.get_feature_names()
        except Exception as e:
            logger.error(f"Error getting risk features: {str(e)}")
            return []


class PatientService:
    """
    Service for patient-related operations
    """
    
    def __init__(self):
        self.db = db_manager
        self.risk_calculator = RiskCalculatorService()
    
    def get_patients(self, skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get a list of patients
        """
        try:
            patients_db = self.db.get_all_patients(skip=skip, limit=limit)
            
            patients = []
            for patient_db in patients_db:
                patient = {
                    "patient_id": patient_db.patient_id,
                    "name": patient_db.name,
                    "age": patient_db.age,
                    "gender": patient_db.gender,
                    "medications": json.loads(patient_db.medications) if patient_db.medications else [],
                    "prescriptions": json.loads(patient_db.prescriptions) if patient_db.prescriptions else [],
                    "clinical_indicators": json.loads(patient_db.clinical_indicators) if patient_db.clinical_indicators else {},
                    "history": json.loads(patient_db.history) if patient_db.history else {},
                    "created_at": patient_db.created_at,
                    "updated_at": patient_db.updated_at
                }
                patients.append(patient)
            
            return patients
        except Exception as e:
            logger.error(f"Error getting patients: {str(e)}")
            raise
    
    def get_patient_by_id(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a patient by ID
        """
        try:
            patient_db = self.db.get_patient(patient_id)
            if patient_db:
                return {
                    "patient_id": patient_db.patient_id,
                    "name": patient_db.name,
                    "age": patient_db.age,
                    "gender": patient_db.gender,
                    "medications": json.loads(patient_db.medications) if patient_db.medications else [],
                    "prescriptions": json.loads(patient_db.prescriptions) if patient_db.prescriptions else [],
                    "clinical_indicators": json.loads(patient_db.clinical_indicators) if patient_db.clinical_indicators else {},
                    "history": json.loads(patient_db.history) if patient_db.history else {},
                    "created_at": patient_db.created_at,
                    "updated_at": patient_db.updated_at
                }
            return None
        except Exception as e:
            logger.error(f"Error getting patient {patient_id}: {str(e)}")
            raise
    
    def create_patient(self, patient_request: Any) -> Dict[str, Any]:
        """
        Create a new patient
        """
        try:
            # Convert request object to dict
            patient_data = {
                "patient_id": patient_request.patient_id if hasattr(patient_request, 'patient_id') else f"PT{int(datetime.now().timestamp())}",
                "name": patient_request.name,
                "age": patient_request.age,
                "gender": patient_request.gender,
                "medications": patient_request.medications,
                "prescriptions": patient_request.prescriptions,
                "clinical_indicators": patient_request.clinical_indicators,
                "history": patient_request.history
            }
            
            # Create in DB
            patient_db = self.db.create_patient(patient_data)
            
            # Return the created patient
            return {
                "patient_id": patient_db.patient_id,
                "name": patient_db.name,
                "age": patient_db.age,
                "gender": patient_db.gender,
                "medications": json.loads(patient_db.medications) if patient_db.medications else [],
                "prescriptions": json.loads(patient_db.prescriptions) if patient_db.prescriptions else [],
                "clinical_indicators": json.loads(patient_db.clinical_indicators) if patient_db.clinical_indicators else {},
                "history": json.loads(patient_db.history) if patient_db.history else {},
                "created_at": patient_db.created_at,
                "updated_at": patient_db.updated_at
            }
        except Exception as e:
            logger.error(f"Error creating patient: {str(e)}")
            raise
    
    def update_patient(self, patient_id: str, patient_request: Any) -> Optional[Dict[str, Any]]:
        """
        Update a patient
        """
        try:
            # Prepare update data
            patient_data = {}
            if hasattr(patient_request, 'name') and patient_request.name is not None:
                patient_data['name'] = patient_request.name
            if hasattr(patient_request, 'age') and patient_request.age is not None:
                patient_data['age'] = patient_request.age
            if hasattr(patient_request, 'gender') and patient_request.gender is not None:
                patient_data['gender'] = patient_request.gender
            if hasattr(patient_request, 'medications') and patient_request.medications is not None:
                patient_data['medications'] = patient_request.medications
            if hasattr(patient_request, 'prescriptions') and patient_request.prescriptions is not None:
                patient_data['prescriptions'] = patient_request.prescriptions
            if hasattr(patient_request, 'clinical_indicators') and patient_request.clinical_indicators is not None:
                patient_data['clinical_indicators'] = patient_request.clinical_indicators
            if hasattr(patient_request, 'history') and patient_request.history is not None:
                patient_data['history'] = patient_request.history
            
            # Update in DB
            patient_db = self.db.update_patient(patient_id, patient_data)
            
            if patient_db:
                return {
                    "patient_id": patient_db.patient_id,
                    "name": patient_db.name,
                    "age": patient_db.age,
                    "gender": patient_db.gender,
                    "medications": json.loads(patient_db.medications) if patient_db.medications else [],
                    "prescriptions": json.loads(patient_db.prescriptions) if patient_db.prescriptions else [],
                    "clinical_indicators": json.loads(patient_db.clinical_indicators) if patient_db.clinical_indicators else {},
                    "history": json.loads(patient_db.history) if patient_db.history else {},
                    "created_at": patient_db.created_at,
                    "updated_at": patient_db.updated_at
                }
            
            return None
        except Exception as e:
            logger.error(f"Error updating patient {patient_id}: {str(e)}")
            raise
    
    def delete_patient(self, patient_id: str) -> bool:
        """
        Delete a patient
        """
        try:
            return self.db.delete_patient(patient_id)
        except Exception as e:
            logger.error(f"Error deleting patient {patient_id}: {str(e)}")
            raise
    
    def get_patient_risk_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Get the risk assessment history for a patient
        """
        try:
            assessments_db = self.db.get_patient_risk_history(patient_id)
            
            assessments = []
            for assessment_db in assessments_db:
                assessment = {
                    "patient_id": assessment_db.patient_id,
                    "risk_score": assessment_db.risk_score,
                    "risk_level": assessment_db.risk_level,
                    "red_flags": json.loads(assessment_db.red_flags) if assessment_db.red_flags else [],
                    "assessment_date": assessment_db.assessment_date,
                    "model_version": assessment_db.model_version
                }
                assessments.append(assessment)
            
            return assessments
        except Exception as e:
            logger.error(f"Error getting risk history for patient {patient_id}: {str(e)}")
            raise


# Global instances
risk_calculator = RiskCalculatorService()
patient_service = PatientService()