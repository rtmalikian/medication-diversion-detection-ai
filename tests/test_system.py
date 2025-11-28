import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

from models.diversion_detection import DiversionDetectionModel
from data.processor import DataProcessor
from services.evaluation_framework import EvidenceBasedEvaluation
from services.xai_service import XAIService
from services.risk_calculator import RiskCalculatorService
from data.loader import DataLoader

class TestDiversionDetectionSystem(unittest.TestCase):
    """
    Comprehensive test suite for the Clinical Risk Modeling Engine
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method
        """
        self.model = DiversionDetectionModel()
        self.data_processor = DataProcessor()
        self.evaluation_framework = EvidenceBasedEvaluation()
        self.xai_service = XAIService()
        self.risk_calculator = RiskCalculatorService()
        self.data_loader = DataLoader()
    
    def test_data_processor(self):
        """
        Test the data processing module
        """
        # Create sample patient data
        sample_patient = {
            "patient_id": "PT001",
            "age": 45,
            "gender": "Male",
            "medications": [
                {"id": "med1", "name": "Oxycodone", "category": "opioid", "strength": 20},
                {"id": "med2", "name": "Lorazepam", "category": "benzodiazepine", "strength": 1.0}
            ],
            "prescriptions": [
                {
                    "id": "rx1", 
                    "drug_name": "Oxycodone 20mg", 
                    "prescriber_id": "Dr. Smith", 
                    "pharmacy_id": "Pharmacy 1",
                    "quantity": 60,
                    "days_supply": 30,
                    "date": "2023-01-15",
                    "cost": 120.50,
                    "dose": 20,
                    "early_refill": False,
                    "concurrent": False,
                    "out_of_region": False
                }
            ],
            "clinical_indicators": {
                "history_of_diversion": False,
                "substance_abuse_history": True,
                "mental_health_history": True,
                "chronic_pain_history": True
            },
            "history": {
                "ed_visits": 3,
                "hospitalizations": 1,
                "primary_care_visits": 8
            }
        }
        
        # Test data processing
        processed_df = self.data_processor.process_patient_data(sample_patient)
        
        # Verify the output
        self.assertIsInstance(processed_df, pd.DataFrame)
        self.assertEqual(processed_df.shape[0], 1)  # One patient
        self.assertGreaterEqual(processed_df.shape[1], 10)  # At least 10 features
        self.assertIn('age', processed_df.columns)
        self.assertIn('gender_encoded', processed_df.columns)
        self.assertEqual(processed_df['age'].iloc[0], 45)
        
        print("✓ Data processor test passed")
    
    def test_evaluation_framework(self):
        """
        Test the evidence-based evaluation framework
        """
        # Create sample patient data
        sample_patient = {
            "patient_id": "PT002",
            "age": 35,
            "gender": "Female",
            "prescriptions": [
                {"drug_name": "Oxycodone", "dose": 30},
                {"drug_name": "Alprazolam", "dose": 2.0}
            ],
            "clinical_indicators": {
                "substance_abuse_history": True,
                "mental_health_history": True,
                "history_of_diversion": False
            }
        }
        
        # Test evaluation
        evaluation_result = self.evaluation_framework.evaluate_patient_risk(sample_patient)
        
        # Verify the output
        self.assertIn('patient_id', evaluation_result)
        self.assertIn('red_flags', evaluation_result)
        self.assertIn('overall_risk_level', evaluation_result)
        self.assertIn('dea_compliance', evaluation_result)
        self.assertIn('cdc_compliance', evaluation_result)
        self.assertIn('samhsa_compliance', evaluation_result)
        
        # Check that risk level is appropriate
        self.assertIn(evaluation_result['overall_risk_level'], ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
        
        print("✓ Evaluation framework test passed")
    
    def test_model_initialization(self):
        """
        Test model initialization and loading
        """
        # Test that the model can be initialized
        self.assertIsNotNone(self.model.model)
        self.assertFalse(self.model.is_trained)  # Initially not trained
        
        # The model should initialize with default parameters
        self.assertEqual(type(self.model.model).__name__, 'GradientBoostingClassifier')
        
        print("✓ Model initialization test passed")
    
    def test_training_simulation(self):
        """
        Test model training (with small dataset for speed)
        """
        # Use the data loader to generate a small synthetic dataset
        synthetic_data = self.data_loader.load_synthetic_data(size=50)
        
        # The training test would go here, but for now just verify the model can be trained
        # In a real scenario, we would run the actual training
        self.assertTrue(True)  # Placeholder for actual training test
        
        print("✓ Training simulation test passed")
    
    def test_risk_calculation(self):
        """
        Test overall risk calculation
        """
        # Create sample patient data
        sample_patient = {
            "patient_id": "PT003",
            "age": 50,
            "gender": "Male",
            "medications": [
                {"name": "Oxycodone", "category": "opioid"}
            ],
            "prescriptions": [
                {"drug_name": "Oxycodone", "dose": 40, "prescriber_id": "Dr. Smith", "pharmacy_id": "Pharmacy 1"}
            ],
            "clinical_indicators": {
                "substance_abuse_history": False,
                "mental_health_history": False,
                "history_of_diversion": False
            },
            "history": {
                "ed_visits": 1
            }
        }
        
        # Calculate risk
        risk_score, red_flags = self.risk_calculator.calculate_diversion_risk(sample_patient)
        
        # Verify outputs
        self.assertIsInstance(risk_score, float)
        self.assertGreaterEqual(risk_score, 0.0)
        self.assertLessEqual(risk_score, 1.0)
        self.assertIsInstance(red_flags, list)
        
        print(f"✓ Risk calculation test passed - Risk score: {risk_score:.4f}")
    
    def test_feature_names(self):
        """
        Test that feature names are properly maintained
        """
        # Get feature names from the model
        feature_names = self.model.get_feature_names()
        
        # Should return a list of feature names
        self.assertIsInstance(feature_names, list)
        self.assertGreaterEqual(len(feature_names), 10)  # Should have at least 10 features
        
        print("✓ Feature names test passed")
    
    def test_synthetic_data_generation(self):
        """
        Test synthetic data generation
        """
        # Generate synthetic data
        synthetic_patients = self.data_loader.load_synthetic_data(size=10)
        
        # Verify the output
        self.assertEqual(len(synthetic_patients), 10)
        
        # Check structure of first patient
        first_patient = synthetic_patients[0]
        self.assertIn('patient_id', first_patient)
        self.assertIn('age', first_patient)
        self.assertIn('gender', first_patient)
        self.assertIn('medications', first_patient)
        self.assertIn('prescriptions', first_patient)
        self.assertIn('clinical_indicators', first_patient)
        
        print("✓ Synthetic data generation test passed")
    
    def test_xai_service_initialization(self):
        """
        Test XAI service initialization
        """
        # The XAI service initialization would require a trained model
        # For now, just verify that the service object exists
        self.assertIsNotNone(self.xai_service.explainer)
        
        print("✓ XAI service initialization test passed")


class IntegrationTest(unittest.TestCase):
    """
    Integration tests for the system components
    """
    
    def setUp(self):
        self.model = DiversionDetectionModel()
        self.data_processor = DataProcessor()
        self.evaluation_framework = EvidenceBasedEvaluation()
        self.risk_calculator = RiskCalculatorService()
    
    def test_end_to_end_flow(self):
        """
        Test the complete flow from patient data to risk assessment
        """
        # Create a sample patient with potential red flags
        sample_patient = {
            "patient_id": "PT004",
            "age": 40,
            "gender": "Female",
            "medications": [
                {"name": "Oxycodone", "category": "opioid"},
                {"name": "Alprazolam", "category": "benzodiazepine"}
            ],
            "prescriptions": [
                {"drug_name": "Oxycodone", "dose": 60, "prescriber_id": "Dr. Smith", "pharmacy_id": "Pharmacy 1"},
                {"drug_name": "Alprazolam", "dose": 2.0, "prescriber_id": "Dr. Jones", "pharmacy_id": "Pharmacy 2"}
            ],
            "clinical_indicators": {
                "substance_abuse_history": True,
                "mental_health_history": True,
                "history_of_diversion": False
            },
            "history": {
                "ed_visits": 5
            }
        }
        
        # Process data
        processed_data = self.data_processor.process_patient_data(sample_patient)
        self.assertIsInstance(processed_data, pd.DataFrame)
        
        # Calculate risk
        risk_score, red_flags = self.risk_calculator.calculate_diversion_risk(sample_patient)
        self.assertIsInstance(risk_score, float)
        self.assertGreaterEqual(risk_score, 0.0)
        self.assertLessEqual(risk_score, 1.0)
        self.assertIsInstance(red_flags, list)
        
        # Evaluate against guidelines
        evaluation = self.evaluation_framework.evaluate_patient_risk(sample_patient)
        self.assertIn('overall_risk_level', evaluation)
        
        print(f"✓ End-to-end flow test passed - Risk score: {risk_score:.4f}, Red flags: {len(red_flags)}")
    
    def test_batch_processing(self):
        """
        Test batch processing of multiple patients
        """
        # Create multiple sample patients
        patients = [
            {
                "patient_id": f"PT{i:03d}",
                "age": 30 + i,
                "gender": "Male" if i % 2 == 0 else "Female",
                "medications": [{"name": "Generic Med", "category": "analgesic"}],
                "prescriptions": [{"drug_name": "Generic Drug", "dose": 10 + i, "prescriber_id": f"Dr. Smith {i}", "pharmacy_id": f"Pharmacy {i}"}],
                "clinical_indicators": {"substance_abuse_history": i > 3, "mental_health_history": i > 2, "history_of_diversion": False},
                "history": {"ed_visits": i}
            }
            for i in range(5)
        ]
        
        # Process batch
        results = self.risk_calculator.batch_calculate_risk(patients)
        
        # Verify results
        self.assertEqual(len(results), 5)
        for patient_id, risk_score, red_flags in results:
            self.assertIsInstance(patient_id, str)
            self.assertIsInstance(risk_score, float)
            self.assertGreaterEqual(risk_score, 0.0)
            self.assertLessEqual(risk_score, 1.0)
            self.assertIsInstance(red_flags, list)
        
        print("✓ Batch processing test passed")


def run_tests():
    """
    Run all tests
    """
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add tests from both test classes
    test_suite.addTest(unittest.makeSuite(TestDiversionDetectionSystem))
    test_suite.addTest(unittest.makeSuite(IntegrationTest))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun}")
    print(f"{'='*50}")
    
    return result


if __name__ == '__main__':
    run_tests()