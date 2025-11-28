import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import re

from config import settings

# Configure logging
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Class to handle data processing for medication diversion detection
    """
    
    def __init__(self):
        self.feature_columns = [
            'age', 'gender_encoded', 'num_medications', 'high_risk_medications',
            'frequent_prescriptions', 'prescription_frequency', 'medication_cost',
            'controlled_substances', 'opioid_prescriptions', 'benzodiazepine_prescriptions',
            'polypharmacy_score', 'prescriber_count', 'pharmacy_count',
            'early_refill_events', 'out_of_region_prescriptions', 'duplicate_prescriptions',
            'high_dose_prescriptions', 'concurrent_prescriptions', 'history_of_diversion',
            'substance_abuse_history', 'mental_health_history', 'chronic_pain_history',
            'emergency_department_visits', 'doctor_shopping_indicators'
        ]
    
    def process_patient_data(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Process raw patient data into features for ML model
        """
        try:
            logger.info("Processing patient data")
            
            # Initialize feature dictionary
            features = {}
            
            # Age-based features
            features['age'] = raw_data.get('age', 0)
            
            # Gender encoding
            gender = raw_data.get('gender', 'unknown').lower()
            features['gender_encoded'] = 1 if gender == 'male' else 0
            
            # Medication-related features
            medications = raw_data.get('medications', [])
            features['num_medications'] = len(medications)
            
            # Count high-risk medications
            high_risk_keywords = ['opioid', 'benzodiazepine', 'stimulant', 'sedative']
            features['high_risk_medications'] = sum(
                1 for med in medications 
                if any(keyword in med.get('name', '').lower() for keyword in high_risk_keywords)
            )
            
            # Prescription frequency analysis
            prescriptions = raw_data.get('prescriptions', [])
            features['frequent_prescriptions'] = len(prescriptions)
            
            # Calculate prescription frequency (prescriptions per month)
            if prescriptions:
                # Assuming prescriptions have date information
                start_date = min(p.get('date', datetime.now()) for p in prescriptions)
                end_date = max(p.get('date', datetime.now()) for p in prescriptions)
                months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                features['prescription_frequency'] = len(prescriptions) / max(months, 1)
            else:
                features['prescription_frequency'] = 0
            
            # Calculate total medication cost
            features['medication_cost'] = sum(p.get('cost', 0) for p in prescriptions)
            
            # Controlled substance indicators
            controlled_keywords = ['schedule', 'controlled', 'narcotic', 'opioid', 'benzo']
            features['controlled_substances'] = sum(
                1 for p in prescriptions
                if any(keyword in p.get('drug_name', '').lower() for keyword in controlled_keywords)
            )
            
            # Specific medication types
            features['opioid_prescriptions'] = sum(
                1 for p in prescriptions
                if 'opioid' in p.get('drug_name', '').lower()
            )
            
            features['benzodiazepine_prescriptions'] = sum(
                1 for p in prescriptions
                if 'benzo' in p.get('drug_name', '').lower() or 'diazepam' in p.get('drug_name', '').lower()
            )
            
            # Polypharmacy score (taking multiple medications simultaneously)
            features['polypharmacy_score'] = self._calculate_polypharmacy_score(medications, prescriptions)
            
            # Healthcare provider patterns
            prescribers = set(p.get('prescriber_id', '') for p in prescriptions)
            features['prescriber_count'] = len(prescribers)
            
            pharmacies = set(p.get('pharmacy_id', '') for p in prescriptions)
            features['pharmacy_count'] = len(pharmacies)
            
            # Early refill events
            features['early_refill_events'] = self._count_early_refills(prescriptions)
            
            # Out-of-region prescriptions
            features['out_of_region_prescriptions'] = self._count_out_of_region_prescriptions(prescriptions)
            
            # Duplicate prescriptions
            features['duplicate_prescriptions'] = self._count_duplicate_prescriptions(prescriptions)
            
            # High-dose prescriptions
            features['high_dose_prescriptions'] = self._count_high_dose_prescriptions(prescriptions)
            
            # Concurrent prescriptions
            features['concurrent_prescriptions'] = self._count_concurrent_prescriptions(prescriptions)
            
            # Clinical history indicators
            clinical_indicators = raw_data.get('clinical_indicators', {})
            features['history_of_diversion'] = 1 if clinical_indicators.get('history_of_diversion', False) else 0
            features['substance_abuse_history'] = 1 if clinical_indicators.get('substance_abuse_history', False) else 0
            features['mental_health_history'] = 1 if clinical_indicators.get('mental_health_history', False) else 0
            features['chronic_pain_history'] = 1 if clinical_indicators.get('chronic_pain_history', False) else 0
            
            # Healthcare utilization patterns
            history = raw_data.get('history', {})
            features['emergency_department_visits'] = history.get('ed_visits', 0)
            features['doctor_shopping_indicators'] = self._calculate_doctor_shopping_score(prescribers, pharmacies)
            
            # Create DataFrame with all features
            df = pd.DataFrame([features])
            
            # Ensure all required columns exist, fill missing with 0
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Select only the required columns in the right order
            df = df[self.feature_columns]
            
            logger.info("Patient data processing completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error processing patient data: {str(e)}")
            raise
    
    def _calculate_polypharmacy_score(self, medications: List[Dict], prescriptions: List[Dict]) -> float:
        """
        Calculate a polypharmacy score based on simultaneous medication use
        """
        try:
            # This is a simplified polypharmacy calculation
            # In reality, this would involve more complex temporal analysis
            num_medications = len(medications)
            
            # Additional risk if multiple medications are taken simultaneously
            if num_medications > 5:
                return num_medications * 0.2
            elif num_medications > 10:
                return num_medications * 0.3
            else:
                return num_medications * 0.1
                
        except Exception:
            return 0.0
    
    def _count_early_refills(self, prescriptions: List[Dict]) -> int:
        """
        Count early refill events (refills before expected date)
        """
        try:
            early_refill_count = 0
            for p in prescriptions:
                # This would involve comparing actual fill date with expected refill date
                # For demonstration, we'll simulate this
                if p.get('early_refill', False):
                    early_refill_count += 1
            return early_refill_count
        except Exception:
            return 0
    
    def _count_out_of_region_prescriptions(self, prescriptions: List[Dict]) -> int:
        """
        Count prescriptions from out-of-region providers
        """
        try:
            out_of_region_count = 0
            for p in prescriptions:
                # This would involve checking provider location against patient location
                if p.get('out_of_region', False):
                    out_of_region_count += 1
            return out_of_region_count
        except Exception:
            return 0
    
    def _count_duplicate_prescriptions(self, prescriptions: List[Dict]) -> int:
        """
        Count potential duplicate prescriptions
        """
        try:
            # Simple logic to count multiple prescriptions of the same drug
            drug_counts = {}
            for p in prescriptions:
                drug_name = p.get('drug_name', '').lower()
                drug_counts[drug_name] = drug_counts.get(drug_name, 0) + 1
            
            # Count drugs that appear more than once
            duplicate_count = sum(1 for count in drug_counts.values() if count > 1)
            return duplicate_count
        except Exception:
            return 0
    
    def _count_high_dose_prescriptions(self, prescriptions: List[Dict]) -> int:
        """
        Count prescriptions with high doses
        """
        try:
            high_dose_count = 0
            for p in prescriptions:
                # This would involve comparing dose to standard thresholds
                # For opioids, this might involve morphine milligram equivalents (MME)
                dose = p.get('dose', 0)
                if dose > 100:  # Example threshold
                    high_dose_count += 1
            return high_dose_count
        except Exception:
            return 0
    
    def _count_concurrent_prescriptions(self, prescriptions: List[Dict]) -> int:
        """
        Count concurrent prescriptions (overlapping)
        """
        try:
            # This would involve temporal analysis to identify overlapping prescriptions
            # For demonstration, we'll simulate this
            concurrent_count = 0
            for p in prescriptions:
                if p.get('concurrent', False):
                    concurrent_count += 1
            return concurrent_count
        except Exception:
            return 0
    
    def _calculate_doctor_shopping_score(self, prescribers: set, pharmacies: set) -> float:
        """
        Calculate a doctor shopping score based on multiple providers
        """
        try:
            # Higher scores for more prescribers relative to pharmacies
            if len(prescribers) > 1 and len(pharmacies) > 1:
                return min(len(prescribers) * 0.2, 1.0)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def preprocess_dataset(self, raw_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a full dataset for ML training
        """
        try:
            logger.info("Preprocessing dataset for ML training")
            
            # Create a copy to avoid modifying original data
            df = raw_dataset.copy()
            
            # Handle missing values
            df = df.fillna(0)
            
            # Encode categorical variables
            df = pd.get_dummies(df, columns=['gender'], prefix='gender')
            
            # Normalize/standardize numerical features
            # This would typically involve using sklearn's preprocessing tools
            
            logger.info("Dataset preprocessing completed")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing dataset: {str(e)}")
            raise


# Global instance
data_processor = DataProcessor()