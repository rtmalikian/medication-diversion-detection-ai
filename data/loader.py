import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
import json
import csv
from datetime import datetime
import sqlite3
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Class to load and manage patient data for the diversion detection system
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.database_url.replace("sqlite:///", "")
    
    def load_patient_data_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load patient data from CSV file
        """
        try:
            logger.info(f"Loading patient data from CSV: {file_path}")
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} patient records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading patient data from CSV: {str(e)}")
            raise
    
    def load_patient_data_from_json(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load patient data from JSON file
        """
        try:
            logger.info(f"Loading patient data from JSON: {file_path}")
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} patient records from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading patient data from JSON: {str(e)}")
            raise
    
    def load_patient_data_from_db(self, patient_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Load patient data from database
        """
        try:
            logger.info("Loading patient data from database")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if patient_ids:
                placeholders = ','.join(['?' for _ in patient_ids])
                query = f"""
                SELECT * FROM patients 
                WHERE patient_id IN ({placeholders})
                """
                cursor.execute(query, patient_ids)
            else:
                cursor.execute("SELECT * FROM patients")
            
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
            # Convert rows to list of dictionaries
            patients = []
            for row in rows:
                patient = {columns[i]: row[i] for i in range(len(columns))}
                patients.append(patient)
            
            conn.close()
            logger.info(f"Loaded {len(patients)} patient records from database")
            return patients
            
        except Exception as e:
            logger.error(f"Error loading patient data from database: {str(e)}")
            raise
    
    def load_synthetic_data(self, size: int = 1000) -> List[Dict[str, Any]]:
        """
        Generate synthetic patient data for testing and development
        """
        try:
            logger.info(f"Generating synthetic patient data of size {size}")
            
            # Create synthetic patient records following evidence-based guidelines
            patients = []
            
            for i in range(size):
                patient = self._generate_synthetic_patient(i)
                patients.append(patient)
            
            logger.info(f"Generated {len(patients)} synthetic patient records")
            return patients
            
        except Exception as e:
            logger.error(f"Error generating synthetic patient data: {str(e)}")
            raise
    
    def _generate_synthetic_patient(self, patient_idx: int) -> Dict[str, Any]:
        """
        Generate a single synthetic patient record
        """
        import random
        
        # Randomly assign demographic information
        gender = random.choice(['Male', 'Female'])
        age = random.randint(18, 85)
        
        # Generate medications based on age and gender
        medications = []
        num_meds = random.randint(1, 15)  # Varying number of medications
        
        high_risk_drugs = [
            {"name": "Oxycodone", "category": "opioid", "strength": random.randint(5, 80)},
            {"name": "Hydrocodone", "category": "opioid", "strength": random.randint(2.5, 10)},
            {"name": "Fentanyl", "category": "opioid", "strength": random.random()},
            {"name": "Alprazolam", "category": "benzodiazepine", "strength": random.uniform(0.25, 2.0)},
            {"name": "Lorazepam", "category": "benzodiazepine", "strength": random.uniform(0.5, 2.0)},
            {"name": "Adderall", "category": "stimulant", "strength": random.randint(5, 30)},
            {"name": "Ritalin", "category": "stimulant", "strength": random.randint(5, 20)}
        ]
        
        selected_meds = random.sample(high_risk_drugs, min(num_meds, len(high_risk_drugs)))
        medications = [
            {
                "id": f"med_{patient_idx}_{i}",
                "name": med["name"],
                "category": med["category"],
                "strength": med["strength"],
                "dosage_form": random.choice(["tablet", "capsule", "liquid", "patch"]),
                "frequency": random.randint(1, 4)
            }
            for i, med in enumerate(selected_meds)
        ]
        
        # Generate prescriptions
        prescriptions = []
        num_prescriptions = random.randint(1, 20)
        
        for j in range(num_prescriptions):
            prescriber = f"Dr. Smith {j+1}" if j < 3 else f"Dr. Johnson {j+1}"
            pharmacy = f"Pharmacy {random.randint(1, 100)}"
            
            prescriptions.append({
                "id": f"rx_{patient_idx}_{j}",
                "drug_name": medications[j % len(medications)]["name"] if medications else "Unknown",
                "prescriber_id": prescriber,
                "pharmacy_id": pharmacy,
                "quantity": random.randint(10, 120),
                "days_supply": random.randint(7, 90),
                "date": datetime.now().strftime('%Y-%m-%d'),
                "cost": random.uniform(10, 500),
                "dose": medications[j % len(medications)]["strength"] if medications else 0,
                "early_refill": random.random() < 0.1,  # 10% chance of early refill
                "concurrent": random.random() < 0.05,   # 5% chance of concurrent prescriptions
                "out_of_region": random.random() < 0.02 # 2% chance of out-of-region
            })
        
        # Generate clinical indicators based on DEA, CDC, and SAMHSA guidelines
        clinical_indicators = {
            "history_of_diversion": random.random() < 0.05,  # 5% chance of history
            "substance_abuse_history": random.random() < 0.15,  # 15% chance
            "mental_health_history": random.random() < 0.25,  # 25% chance
            "chronic_pain_history": random.random() < 0.30,  # 30% chance
            "high_risk_behavior_patterns": random.random() < 0.10  # 10% chance
        }
        
        # Generate medical history
        history = {
            "ed_visits": random.randint(0, 10),  # Emergency department visits
            "hospitalizations": random.randint(0, 5),
            "primary_care_visits": random.randint(0, 12),
            "specialist_visits": random.randint(0, 8)
        }
        
        # Calculate some risk indicators
        opioid_prescriptions = sum(1 for p in prescriptions if 'opioid' in p['drug_name'].lower())
        benzodiazepine_prescriptions = sum(1 for p in prescriptions if 'benzo' in p['drug_name'].lower())
        
        # Create the patient record
        patient = {
            "patient_id": f"PT{patient_idx:05d}",
            "name": f"Patient {patient_idx}",
            "age": age,
            "gender": gender,
            "medications": medications,
            "prescriptions": prescriptions,
            "clinical_indicators": clinical_indicators,
            "history": history,
            # Additional features for ML model
            "opioid_prescriptions": opioid_prescriptions,
            "benzodiazepine_prescriptions": benzodiazepine_prescriptions,
            "num_prescribers": len(set(p['prescriber_id'] for p in prescriptions)),
            "num_pharmacies": len(set(p['pharmacy_id'] for p in prescriptions)),
            "early_refill_events": sum(1 for p in prescriptions if p['early_refill']),
            "concurrent_prescriptions": sum(1 for p in prescriptions if p['concurrent']),
            "out_of_region_prescriptions": sum(1 for p in prescriptions if p['out_of_region'])
        }
        
        return patient
    
    def save_patient_data_to_db(self, patients: List[Dict[str, Any]]) -> None:
        """
        Save patient data to database
        """
        try:
            logger.info(f"Saving {len(patients)} patient records to database")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    name TEXT,
                    age INTEGER,
                    gender TEXT,
                    medications TEXT,
                    prescriptions TEXT,
                    clinical_indicators TEXT,
                    history TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert or update patient records
            for patient in patients:
                # Convert complex objects to JSON strings for storage
                medications_json = json.dumps(patient.get('medications', []))
                prescriptions_json = json.dumps(patient.get('prescriptions', []))
                clinical_indicators_json = json.dumps(patient.get('clinical_indicators', {}))
                history_json = json.dumps(patient.get('history', {}))
                
                cursor.execute('''
                    INSERT OR REPLACE INTO patients 
                    (patient_id, name, age, gender, medications, prescriptions, 
                     clinical_indicators, history, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    patient['patient_id'], patient['name'], patient['age'], 
                    patient['gender'], medications_json, prescriptions_json,
                    clinical_indicators_json, history_json
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully saved {len(patients)} patient records to database")
            
        except Exception as e:
            logger.error(f"Error saving patient data to database: {str(e)}")
            raise
    
    def get_patient_count(self) -> int:
        """
        Get the total count of patients in the database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM patients")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.error(f"Error getting patient count: {str(e)}")
            return 0


# Global instance
data_loader = DataLoader()