#!/usr/bin/env python3
"""
Basic functionality test for Clinical Risk Modeling Engine
"""

import sys
import os

def test_imports():
    """Test that all main modules can be imported"""
    print("Testing module imports...")
    
    modules_to_test = [
        "main",
        "config",
        "models.diversion_detection",
        "data.processor", 
        "services.evaluation_framework",
        "services.xai_service",
        "services.risk_calculator",
        "services.database",
        "api.risk_assessment",
        "api.patient_data",
        "api.model_management",
        "data.loader"
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module} - Error: {e}")
            failed_imports.append((module, str(e)))
    
    return len(failed_imports) == 0

def test_model_creation():
    """Test basic model functionality"""
    print("\nTesting model creation...")
    
    try:
        from models.diversion_detection import DiversionDetectionModel
        model = DiversionDetectionModel()
        print("✓ Model created successfully")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_data_processor():
    """Test data processor functionality"""
    print("\nTesting data processor...")
    
    try:
        from data.processor import DataProcessor
        processor = DataProcessor()
        print("✓ Data processor created successfully")
        return True
    except Exception as e:
        print(f"✗ Data processor failed: {e}")
        return False

def test_evaluation_framework():
    """Test evaluation framework"""
    print("\nTesting evaluation framework...")
    
    try:
        from services.evaluation_framework import EvidenceBasedEvaluation
        evaluator = EvidenceBasedEvaluation()
        print("✓ Evaluation framework created successfully")
        return True
    except Exception as e:
        print(f"✗ Evaluation framework failed: {e}")
        return False

def test_risk_calculator():
    """Test risk calculator"""
    print("\nTesting risk calculator...")
    
    try:
        from services.risk_calculator import RiskCalculatorService
        calculator = RiskCalculatorService()
        print("✓ Risk calculator created successfully")
        return True
    except Exception as e:
        print(f"✗ Risk calculator failed: {e}")
        return False

def run_basic_tests():
    """Run all basic functionality tests"""
    print("Running basic functionality tests for Clinical Risk Modeling Engine...\n")
    
    all_passed = True
    
    # Test imports
    all_passed &= test_imports()
    
    # Test individual components
    all_passed &= test_model_creation()
    all_passed &= test_data_processor()
    all_passed &= test_evaluation_framework()
    all_passed &= test_risk_calculator()
    
    print(f"\n{'='*50}")
    if all_passed:
        print("✓ All basic functionality tests PASSED!")
        print("The system is ready for prototyping.")
    else:
        print("✗ Some tests FAILED!")
        print("Please check the error messages above.")
    print(f"{'='*50}")
    
    return all_passed

if __name__ == "__main__":
    run_basic_tests()