import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import re

from config import settings

# Configure logging
logger = logging.getLogger(__name__)

class EvidenceBasedEvaluation:
    """
    Evidence-based evaluation framework using DEA, CDC, and SAMHSA guidelines
    """
    
    def __init__(self):
        # DEA, CDC, and SAMHSA guideline thresholds and criteria
        self.dea_guidelines = {
            'high_volume_prescriber': 100,  # Patients per month
            'controlled_substance_threshold': 5,  # Number of controlled substances
            'schedule_II_threshold': 3,  # Number of Schedule II drugs
            'morphine_milligram_equivalent': 90,  # MME per day threshold
        }
        
        self.cdc_guidelines = {
            'opioid_prescribing_threshold': 50,  # MME per day
            'concurrent_benzos_opioids': True,  # Flag when both prescribed
            'long_term_opioid_use': 90,  # Days threshold for long-term use
            'high_dose_opioids': 120,  # MME per day for high-dose category
        }
        
        self.samhsa_guidelines = {
            'risk_factors': [
                'substance_abuse_history',
                'mental_health_history',
                'family_history_substance_abuse',
                'history_of_diversion'
            ],
            'vulnerable_populations': [
                'young_adults',
                'history_mental_health',
                'social_stressors'
            ]
        }
    
    def evaluate_patient_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate patient risk based on evidence-based guidelines
        """
        try:
            logger.info(f"Evaluating risk for patient {patient_data.get('patient_id', 'unknown')}")
            
            # Initialize result dictionary
            evaluation_result = {
                'patient_id': patient_data.get('patient_id'),
                'timestamp': datetime.now().isoformat(),
                'red_flags': [],
                'risk_factors': [],
                'guideline_violations': [],
                'overall_risk_level': 'LOW',
                'dea_compliance': True,
                'cdc_compliance': True,
                'samhsa_compliance': True
            }
            
            # Evaluate against DEA guidelines
            dea_flags = self._evaluate_dea_guidelines(patient_data)
            evaluation_result['de'] = dea_flags
            
            # Evaluate against CDC guidelines
            cdc_flags = self._evaluate_cdc_guidelines(patient_data)
            evaluation_result['cdc'] = cdc_flags
            
            # Evaluate against SAMHSA guidelines
            samhsa_flags = self._evaluate_samhsa_guidelines(patient_data)
            evaluation_result['samhsa'] = samhsa_flags
            
            # Combine all flags
            all_flags = dea_flags['red_flags'] + cdc_flags['red_flags'] + samhsa_flags['red_flags']
            evaluation_result['red_flags'] = all_flags
            
            # Calculate overall risk based on flags
            evaluation_result['overall_risk_level'] = self._calculate_overall_risk(all_flags)
            
            # Determine compliance status
            evaluation_result['dea_compliance'] = len(dea_flags['red_flags']) == 0
            evaluation_result['cdc_compliance'] = len(cdc_flags['red_flags']) == 0
            evaluation_result['samhsa_compliance'] = len(samhsa_flags['red_flags']) == 0
            
            logger.info(f"Risk evaluation completed for patient {patient_data.get('patient_id', 'unknown')}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error in patient risk evaluation: {str(e)}")
            raise
    
    def _evaluate_dea_guidelines(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate patient against DEA guidelines
        """
        try:
            red_flags = []
            violations = []
            
            # Check for high volume prescriptions
            prescriptions = patient_data.get('prescriptions', [])
            if len(prescriptions) > self.dea_guidelines['high_volume_prescriber']:
                red_flags.append({
                    'type': 'HIGH_VOLUME',
                    'description': f'Patient has {len(prescriptions)} prescriptions (threshold: {self.dea_guidelines["high_volume_prescriber"]})',
                    'severity': 'MEDIUM'
                })
                violations.append('high_volume_prescriptions')
            
            # Check for controlled substances
            controlled_count = sum(1 for p in prescriptions 
                                 if any(keyword in p.get('drug_name', '').lower() 
                                       for keyword in ['schedule', 'controlled', 'narcotic']))
            
            if controlled_count > self.dea_guidelines['controlled_substance_threshold']:
                red_flags.append({
                    'type': 'CONTROLLED_SUBSTANCES',
                    'description': f'Patient has {controlled_count} controlled substances (threshold: {self.dea_guidelines["controlled_substance_threshold"]})',
                    'severity': 'HIGH'
                })
                violations.append('excessive_controlled_substances')
            
            # Check for Schedule II drugs
            schedule_II_count = sum(1 for p in prescriptions 
                                  if 'schedule ii' in p.get('drug_name', '').lower() or 
                                     'schedule 2' in p.get('drug_name', '').lower())
            
            if schedule_II_count > self.dea_guidelines['schedule_II_threshold']:
                red_flags.append({
                    'type': 'SCHEDULE_II',
                    'description': f'Patient has {schedule_II_count} Schedule II drugs (threshold: {self.dea_guidelines["schedule_II_threshold"]})',
                    'severity': 'HIGH'
                })
                violations.append('excessive_schedule_II_drugs')
            
            # Calculate morphine milligram equivalent (simplified)
            mme = self._calculate_mme(prescriptions)
            if mme > self.dea_guidelines['morphine_milligram_equivalent']:
                red_flags.append({
                    'type': 'HIGH_MME',
                    'description': f'Patient has MME of {mme} (threshold: {self.dea_guidelines["morphine_milligram_equivalent"]})',
                    'severity': 'HIGH'
                })
                violations.append('high_mme_prescribing')
            
            # Check for duplicate prescriptions
            duplicate_rx = patient_data.get('duplicate_prescriptions', 0)
            if duplicate_rx > 1:
                red_flags.append({
                    'type': 'DUPLICATE_PRESCRIPTIONS',
                    'description': f'Patient has {duplicate_rx} duplicate prescriptions',
                    'severity': 'MEDIUM'
                })
                violations.append('duplicate_prescriptions')
            
            return {
                'red_flags': red_flags,
                'violations': violations,
                'controlled_substances': controlled_count,
                'schedule_II_drugs': schedule_II_count
            }
            
        except Exception as e:
            logger.error(f"Error evaluating DEA guidelines: {str(e)}")
            return {'red_flags': [], 'violations': [], 'controlled_substances': 0, 'schedule_II_drugs': 0}
    
    def _evaluate_cdc_guidelines(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate patient against CDC guidelines
        """
        try:
            red_flags = []
            violations = []
            
            prescriptions = patient_data.get('prescriptions', [])
            
            # Calculate MME (Morphine Milligram Equivalent)
            mme = self._calculate_mme(prescriptions)
            
            # Check for opioid prescribing threshold
            if mme > self.cdc_guidelines['opioid_prescribing_threshold']:
                red_flags.append({
                    'type': 'HIGH_DOSE_OPIOID',
                    'description': f'Patient has MME of {mme} (CDC threshold: {self.cdc_guidelines["opioid_prescribing_threshold"]})',
                    'severity': 'HIGH'
                })
                violations.append('high_dose_opioid_prescribing')
            
            # Check for concurrent benzodiazepine and opioid use
            opioid_prescriptions = sum(1 for p in prescriptions if 'opioid' in p.get('drug_name', '').lower())
            benzo_prescriptions = sum(1 for p in prescriptions if 'benzo' in p.get('drug_name', '').lower() or 'diazepam' in p.get('drug_name', '').lower())
            
            if opioid_prescriptions > 0 and benzo_prescriptions > 0 and self.cdc_guidelines['concurrent_benzos_opioids']:
                red_flags.append({
                    'type': 'CONCURRENT_BENZOS_OPIOIDS',
                    'description': 'Patient has concurrent benzodiazepine and opioid prescriptions',
                    'severity': 'HIGH'
                })
                violations.append('concurrent_benzos_opioids')
            
            # Check for long-term opioid use
            opioid_prescriptions_count = sum(1 for p in prescriptions if 'opioid' in p.get('drug_name', '').lower())
            if opioid_prescriptions_count > self.cdc_guidelines['long_term_opioid_use']:
                red_flags.append({
                    'type': 'LONG_TERM_OPIOID_USE',
                    'description': f'Long-term opioid use detected ({opioid_prescriptions_count} prescriptions)',
                    'severity': 'MEDIUM'
                })
                violations.append('long_term_opioid_use')
            
            # Check for high-dose opioids
            if mme > self.cdc_guidelines['high_dose_opioids']:
                red_flags.append({
                    'type': 'VERY_HIGH_DOSE_OPIOID',
                    'description': f'Very high dose opioid prescribing: {mme} MME (threshold: {self.cdc_guidelines["high_dose_opioids"]})',
                    'severity': 'CRITICAL'
                })
                violations.append('very_high_dose_opioids')
            
            return {
                'red_flags': red_flags,
                'violations': violations,
                'mme': mme,
                'opioid_prescriptions': opioid_prescriptions,
                'benzo_prescriptions': benzo_prescriptions
            }
            
        except Exception as e:
            logger.error(f"Error evaluating CDC guidelines: {str(e)}")
            return {'red_flags': [], 'violations': [], 'mme': 0, 'opioid_prescriptions': 0, 'benzo_prescriptions': 0}
    
    def _evaluate_samhsa_guidelines(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate patient against SAMHSA guidelines
        """
        try:
            red_flags = []
            violations = []
            
            clinical_indicators = patient_data.get('clinical_indicators', {})
            
            # Check for SAMHSA risk factors
            risk_factor_count = 0
            for risk_factor in self.samhsa_guidelines['risk_factors']:
                if clinical_indicators.get(risk_factor, False):
                    risk_factor_count += 1
                    red_flags.append({
                        'type': 'SAMHSA_RISK_FACTOR',
                        'description': f'SAMHSA risk factor present: {risk_factor}',
                        'severity': 'MEDIUM'
                    })
                    violations.append(risk_factor)
            
            # Check for vulnerable populations
            age = patient_data.get('age', 0)
            if 18 <= age <= 25:  # Young adults
                red_flags.append({
                    'type': 'VULNERABLE_POPULATION',
                    'description': 'Patient is in vulnerable young adult population (18-25 years)',
                    'severity': 'MEDIUM'
                })
                violations.append('young_adult_vulnerability')
            
            # Mental health history
            if clinical_indicators.get('mental_health_history', False):
                red_flags.append({
                    'type': 'MENTAL_HEALTH_HISTORY',
                    'description': 'Patient has mental health history (SAMHSA risk factor)',
                    'severity': 'MEDIUM'
                })
                violations.append('mental_health_history')
            
            # Substance abuse history
            if clinical_indicators.get('substance_abuse_history', False):
                red_flags.append({
                    'type': 'SUBSTANCE_ABUSE_HISTORY',
                    'description': 'Patient has substance abuse history (major SAMHSA risk factor)',
                    'severity': 'HIGH'
                })
                violations.append('substance_abuse_history')
            
            # History of diversion
            if clinical_indicators.get('history_of_diversion', False):
                red_flags.append({
                    'type': 'HISTORY_OF_DIVERSION',
                    'description': 'Patient has history of medication diversion',
                    'severity': 'CRITICAL'
                })
                violations.append('history_of_diversion')
            
            return {
                'red_flags': red_flags,
                'violations': violations,
                'risk_factor_count': risk_factor_count
            }
            
        except Exception as e:
            logger.error(f"Error evaluating SAMHSA guidelines: {str(e)}")
            return {'red_flags': [], 'violations': [], 'risk_factor_count': 0}
    
    def _calculate_mme(self, prescriptions: List[Dict[str, Any]]) -> float:
        """
        Calculate Morphine Milligram Equivalent (simplified approach)
        In a real implementation, this would use standard MME conversion factors
        """
        try:
            mme = 0.0
            
            # Simplified MME calculation based on drug type and dose
            for p in prescriptions:
                drug_name = p.get('drug_name', '').lower()
                dose = p.get('dose', 0)
                
                # Simplified conversion factors (in practice, these would be more precise)
                if 'fentanyl' in drug_name:
                    # Fentanyl is much more potent - convert to MME
                    mme += dose * 100  # Simplified conversion
                elif 'oxycodone' in drug_name:
                    mme += dose * 1.5
                elif 'hydrocodone' in drug_name:
                    mme += dose * 1.0
                elif 'morphine' in drug_name:
                    mme += dose
                elif 'hydromorphone' in drug_name:
                    mme += dose * 4.0
                elif 'methadone' in drug_name:
                    mme += dose * 3.0  # Simplified - actual conversion is variable
            
            return mme
            
        except Exception:
            return 0.0
    
    def _calculate_overall_risk(self, red_flags: List[Dict[str, Any]]) -> str:
        """
        Calculate overall risk level based on red flags
        """
        try:
            if not red_flags:
                return 'LOW'
            
            # Count severity levels
            critical_count = sum(1 for flag in red_flags if flag.get('severity') == 'CRITICAL')
            high_count = sum(1 for flag in red_flags if flag.get('severity') == 'HIGH')
            medium_count = sum(1 for flag in red_flags if flag.get('severity') == 'MEDIUM')
            
            if critical_count > 0:
                return 'CRITICAL'
            elif high_count >= 3 or (high_count >= 1 and medium_count >= 3):
                return 'HIGH'
            elif high_count >= 1 or (medium_count >= 2):
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception:
            return 'LOW'


# Global instance
evaluation_framework = EvidenceBasedEvaluation()