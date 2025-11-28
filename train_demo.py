#!/usr/bin/env python3
"""
Model training script for Clinical Risk Modeling Engine
Trains the ML model with synthetic data based on public health statistics
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.diversion_detection import DiversionDetectionModel
from data.loader import DataLoader
from data.processor import DataProcessor
from services.risk_calculator import risk_calculator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_demo_training_data():
    """
    Create demonstration training data based on public health statistics
    """
    logger.info("Creating synthetic training data based on public health statistics")
    
    # Base parameters based on public health data
    n_patients = 500  # Smaller set for demo purposes
    np.random.seed(42)  # For reproducible results
    
    patients = []
    
    for i in range(n_patients):
        # Demographic information based on general population statistics
        age = np.random.normal(50, 20)
        age = max(18, min(100, int(age)))  # Keep within realistic range
        gender = np.random.choice(['Male', 'Female'], p=[0.48, 0.52])
        
        # High-risk factors based on CDC/health statistics
        substance_abuse_history = np.random.random() < 0.08  # ~8% baseline
        mental_health_history = np.random.random() < 0.20    # ~20% baseline
        chronic_pain_history = np.random.random() < 0.25     # ~25% baseline
        
        # Medication patterns
        num_medications = np.random.poisson(5)  # Average 5 medications
        num_prescriptions = np.random.poisson(8)  # Average 8 prescriptions
        
        # High-risk medication indicators
        opioid_prescriptions = int(np.random.binomial(num_prescriptions, 0.3))  # 30% chance per prescription
        benzo_prescriptions = int(np.random.binomial(num_prescriptions, 0.15))  # 15% chance per prescription
        
        # Risk indicators
        num_prescribers = np.random.poisson(2) + 1  # At least 1 prescriber
        num_pharmacies = np.random.poisson(1.5) + 1  # At least 1 pharmacy
        early_refill_events = np.random.poisson(0.5)  # Average 0.5 early refills
        concurrent_prescriptions = np.random.poisson(1)  # Average 1 concurrent prescription
        
        # Generate medications list
        medications = []
        for med_idx in range(num_medications):
            med_types = [
                {"name": "Acetaminophen", "category": "analgesic"},
                {"name": "Ibuprofen", "category": "nsaid"},
                {"name": "Oxycodone", "category": "opioid"},
                {"name": "Hydrocodone", "category": "opioid"},
                {"name": "Alprazolam", "category": "benzodiazepine"},
                {"name": "Lorazepam", "category": "benzodiazepine"},
                {"name": "Gabapentin", "category": "anticonvulsant"}
            ]
            med = np.random.choice(med_types)
            medications.append({
                "id": f"med_{i}_{med_idx}",
                "name": med["name"],
                "category": med["category"],
                "strength": np.random.uniform(5, 200) if med["category"] in ["opioid", "benzodiazepine"] else np.random.uniform(100, 1000)
            })
        
        # Generate prescriptions list
        prescriptions = []
        for rx_idx in range(num_prescriptions):
            prescriber = f"Dr. Smith {rx_idx % 5}" if num_prescribers <= 3 else f"Dr. Johnson {rx_idx % 7}"
            pharmacy = f"Pharmacy {np.random.randint(1, 100)}"
            
            # Select a medication for this prescription
            selected_med = np.random.choice(medications) if medications else {"name": "Generic Medication", "category": "analgesic"}
            
            prescriptions.append({
                "id": f"rx_{i}_{rx_idx}",
                "drug_name": selected_med["name"],
                "prescriber_id": prescriber,
                "pharmacy_id": pharmacy,
                "quantity": np.random.randint(10, 120),
                "days_supply": np.random.randint(7, 90),
                "date": datetime.now().strftime('%Y-%m-%d'),
                "cost": np.random.uniform(10, 500),
                "dose": selected_med.get("strength", 10),
                "early_refill": np.random.random() < (early_refill_events / max(num_prescriptions, 1)),
                "concurrent": rx_idx < concurrent_prescriptions,
                "out_of_region": np.random.random() < 0.02  # 2% chance
            })
        
        # Create the patient record
        patient = {
            "patient_id": f"DEMO_PT{i:04d}",
            "name": f"Demo Patient {i}",
            "age": age,
            "gender": gender,
            "medications": medications,
            "prescriptions": prescriptions,
            "clinical_indicators": {
                "history_of_diversion": np.random.random() < 0.02,  # 2% baseline
                "substance_abuse_history": substance_abuse_history,
                "mental_health_history": mental_health_history,
                "chronic_pain_history": chronic_pain_history,
                "high_risk_behavior_patterns": np.random.random() < 0.05  # 5% baseline
            },
            "history": {
                "ed_visits": np.random.poisson(2),  # Average 2 ED visits
                "hospitalizations": np.random.poisson(0.5),  # Average 0.5 hospitalizations
                "primary_care_visits": np.random.poisson(6),  # Average 6 PCP visits
                "specialist_visits": np.random.poisson(4)  # Average 4 specialist visits
            },
            # Additional features for risk calculation
            "opioid_prescriptions": opioid_prescriptions,
            "benzodiazepine_prescriptions": benzo_prescriptions,
            "num_prescribers": num_prescribers,
            "num_pharmacies": num_pharmacies,
            "early_refill_events": early_refill_events,
            "concurrent_prescriptions": concurrent_prescriptions,
            "out_of_region_prescriptions": sum(1 for p in prescriptions if p.get('out_of_region', False))
        }
        
        patients.append(patient)
    
    logger.info(f"Created {len(patients)} synthetic patient records for training")
    return patients

def create_training_labels(patients):
    """
    Create training labels based on risk factors (simulating ground truth)
    """
    logger.info("Creating training labels based on risk factors")
    
    labels = []
    for patient in patients:
        # Create a risk score based on multiple factors
        risk_score = 0
        
        # Clinical indicators
        if patient['clinical_indicators']['history_of_diversion']:
            risk_score += 3
        if patient['clinical_indicators']['substance_abuse_history']:
            risk_score += 2
        if patient['clinical_indicators']['mental_health_history']:
            risk_score += 1
        if patient['clinical_indicators']['chronic_pain_history']:
            risk_score += 1
            
        # Medication patterns
        if patient['opioid_prescriptions'] > 2:
            risk_score += 2
        if patient['benzodiazepine_prescriptions'] > 1:
            risk_score += 1
            
        # Healthcare utilization patterns
        if patient['early_refill_events'] > 1:
            risk_score += 1
        if patient['num_prescribers'] > 3:
            risk_score += 1
        if patient['num_pharmacies'] > 3:
            risk_score += 1
        if patient['concurrent_prescriptions'] > 2:
            risk_score += 2
            
        # Convert to binary label (high risk vs low risk)
        # Using threshold to create balanced dataset
        label = 1 if risk_score >= 4 else 0  # Adjust threshold as needed
        labels.append(label)
    
    # Print distribution
    high_risk_count = sum(labels)
    low_risk_count = len(labels) - high_risk_count
    logger.info(f"Label distribution: {high_risk_count} high-risk, {low_risk_count} low-risk patients")
    
    return np.array(labels)

def train_model_with_demo_data():
    """
    Train the model with demonstration data
    """
    logger.info("Starting model training with synthetic data...")
    
    # Create demo training data
    patients = create_demo_training_data()
    labels = create_training_labels(patients)
    
    # Initialize model
    model = DiversionDetectionModel()
    
    # Prepare training data
    logger.info("Processing training data...")
    X, y = model._prepare_training_data(patients)
    
    # Update labels to match processed data (in case any preprocessing changed the count)
    y = labels[:X.shape[0]] if len(labels) >= X.shape[0] else np.concatenate([labels, np.zeros(X.shape[0] - len(labels))])
    
    logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
    
    # Split data for training and validation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info(f"Training set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
    
    # Train the model
    from sklearn.ensemble import GradientBoostingClassifier
    model.model = GradientBoostingClassifier(
        n_estimators=50,  # Reduced for demo
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model.model.fit(X_train, y_train)
    model.is_trained = True
    
    # Evaluate the model
    train_score = model.model.score(X_train, y_train)
    test_score = model.model.score(X_test, y_test)
    
    logger.info(f"Training accuracy: {train_score:.4f}")
    logger.info(f"Test accuracy: {test_score:.4f}")
    
    # Save the trained model
    model.save_model()
    logger.info("Model training completed and saved!")
    
    # Test with a few sample patients
    logger.info("\nTesting model with sample patients...")
    sample_patients = patients[:5]  # Test with first 5 patients
    
    for patient in sample_patients:
        risk_score, red_flags = risk_calculator.calculate_diversion_risk(patient)
        print(f"Patient {patient['patient_id']}: Risk Score = {risk_score:.4f}, Red Flags = {len(red_flags)}")
    
    return model

def main():
    """
    Main function to run the demo training
    """
    logger.info("Starting Clinical Risk Modeling Engine - Demo Training")
    
    try:
        # Train the model
        trained_model = train_model_with_demo_data()
        
        logger.info("\nDemo training completed successfully!")
        logger.info("The model is now ready for use with the API.")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()