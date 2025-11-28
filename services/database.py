from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
import logging

from config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Database setup
SQLALCHEMY_DATABASE_URL = settings.database_url
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, unique=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer)
    gender = Column(String)
    medications = Column(Text)  # JSON string
    prescriptions = Column(Text)  # JSON string
    clinical_indicators = Column(Text)  # JSON string
    history = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class RiskAssessment(Base):
    __tablename__ = "risk_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    risk_score = Column(Float)
    risk_level = Column(String)
    red_flags = Column(Text)  # JSON string
    assessment_date = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String)

class ModelPerformance(Base):
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String)
    version = Column(String)
    accuracy = Column(Float)
    auc_score = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    evaluation_date = Column(DateTime, default=datetime.utcnow)
    metrics = Column(Text)  # JSON string

# Create tables
Base.metadata.create_all(bind=engine)

class DatabaseManager:
    """
    Database manager for the clinical decision support tool
    """
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    def get_db(self) -> Session:
        """
        Get database session
        """
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def create_patient(self, patient_data: Dict[str, Any]) -> Patient:
        """
        Create a new patient record
        """
        try:
            db = next(self.get_db())
            
            # Convert complex objects to JSON strings
            patient = Patient(
                patient_id=patient_data['patient_id'],
                name=patient_data['name'],
                age=patient_data['age'],
                gender=patient_data['gender'],
                medications=json.dumps(patient_data.get('medications', [])),
                prescriptions=json.dumps(patient_data.get('prescriptions', [])),
                clinical_indicators=json.dumps(patient_data.get('clinical_indicators', {})),
                history=json.dumps(patient_data.get('history', {}))
            )
            
            db.add(patient)
            db.commit()
            db.refresh(patient)
            
            logger.info(f"Patient {patient_data['patient_id']} created successfully")
            return patient
            
        except Exception as e:
            logger.error(f"Error creating patient: {str(e)}")
            raise
        finally:
            db.close()
    
    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """
        Get a patient by ID
        """
        try:
            db = next(self.get_db())
            patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
            return patient
        except Exception as e:
            logger.error(f"Error getting patient {patient_id}: {str(e)}")
            raise
        finally:
            db.close()
    
    def update_patient(self, patient_id: str, patient_data: Dict[str, Any]) -> Optional[Patient]:
        """
        Update a patient record
        """
        try:
            db = next(self.get_db())
            patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
            
            if patient:
                # Update fields
                patient.name = patient_data.get('name', patient.name)
                patient.age = patient_data.get('age', patient.age)
                patient.gender = patient_data.get('gender', patient.gender)
                patient.medications = json.dumps(patient_data.get('medications', []))
                patient.prescriptions = json.dumps(patient_data.get('prescriptions', []))
                patient.clinical_indicators = json.dumps(patient_data.get('clinical_indicators', {}))
                patient.history = json.dumps(patient_data.get('history', {}))
                patient.updated_at = datetime.utcnow()
                
                db.commit()
                db.refresh(patient)
                
                logger.info(f"Patient {patient_id} updated successfully")
                return patient
            else:
                logger.warning(f"Patient {patient_id} not found for update")
                return None
                
        except Exception as e:
            logger.error(f"Error updating patient {patient_id}: {str(e)}")
            raise
        finally:
            db.close()
    
    def delete_patient(self, patient_id: str) -> bool:
        """
        Delete a patient record
        """
        try:
            db = next(self.get_db())
            patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
            
            if patient:
                db.delete(patient)
                db.commit()
                logger.info(f"Patient {patient_id} deleted successfully")
                return True
            else:
                logger.warning(f"Patient {patient_id} not found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting patient {patient_id}: {str(e)}")
            raise
        finally:
            db.close()
    
    def get_all_patients(self, skip: int = 0, limit: int = 100) -> List[Patient]:
        """
        Get all patients with pagination
        """
        try:
            db = next(self.get_db())
            patients = db.query(Patient).offset(skip).limit(limit).all()
            return patients
        except Exception as e:
            logger.error(f"Error getting patients: {str(e)}")
            raise
        finally:
            db.close()
    
    def create_risk_assessment(self, assessment_data: Dict[str, Any]) -> RiskAssessment:
        """
        Create a new risk assessment record
        """
        try:
            db = next(self.get_db())
            
            assessment = RiskAssessment(
                patient_id=assessment_data['patient_id'],
                risk_score=assessment_data['risk_score'],
                risk_level=assessment_data['risk_level'],
                red_flags=json.dumps(assessment_data.get('red_flags', [])),
                model_version=assessment_data.get('model_version', '1.0.0')
            )
            
            db.add(assessment)
            db.commit()
            db.refresh(assessment)
            
            logger.info(f"Risk assessment created for patient {assessment_data['patient_id']}")
            return assessment
            
        except Exception as e:
            logger.error(f"Error creating risk assessment: {str(e)}")
            raise
        finally:
            db.close()
    
    def get_patient_risk_history(self, patient_id: str, limit: int = 10) -> List[RiskAssessment]:
        """
        Get risk assessment history for a patient
        """
        try:
            db = next(self.get_db())
            assessments = (
                db.query(RiskAssessment)
                .filter(RiskAssessment.patient_id == patient_id)
                .order_by(RiskAssessment.assessment_date.desc())
                .limit(limit)
                .all()
            )
            return assessments
        except Exception as e:
            logger.error(f"Error getting risk history for patient {patient_id}: {str(e)}")
            raise
        finally:
            db.close()
    
    def save_model_performance(self, performance_data: Dict[str, Any]) -> ModelPerformance:
        """
        Save model performance metrics
        """
        try:
            db = next(self.get_db())
            
            performance = ModelPerformance(
                model_name=performance_data['model_name'],
                version=performance_data['version'],
                accuracy=performance_data.get('accuracy'),
                auc_score=performance_data.get('auc_score'),
                precision=performance_data.get('precision'),
                recall=performance_data.get('recall'),
                f1_score=performance_data.get('f1_score'),
                metrics=json.dumps(performance_data.get('metrics', {}))
            )
            
            db.add(performance)
            db.commit()
            db.refresh(performance)
            
            logger.info(f"Model performance saved for {performance_data['model_name']}")
            return performance
            
        except Exception as e:
            logger.error(f"Error saving model performance: {str(e)}")
            raise
        finally:
            db.close()
    
    def get_model_performance(self, model_name: str = None, limit: int = 10) -> List[ModelPerformance]:
        """
        Get model performance records
        """
        try:
            db = next(self.get_db())
            query = db.query(ModelPerformance)
            
            if model_name:
                query = query.filter(ModelPerformance.model_name == model_name)
            
            performances = (
                query
                .order_by(ModelPerformance.evaluation_date.desc())
                .limit(limit)
                .all()
            )
            return performances
        except Exception as e:
            logger.error(f"Error getting model performance: {str(e)}")
            raise
        finally:
            db.close()


# Global instance
db_manager = DatabaseManager()