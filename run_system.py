#!/usr/bin/env python3
"""
Clinical Risk Modeling Engine for Medication Diversion Detection
Entry point to run the application
"""

import os
import sys
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """
    Set up the environment for the application
    """
    # Create necessary directories if they don't exist
    directories = ['models', 'data', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory '{directory}' is ready")

def initialize_system():
    """
    Initialize the clinical decision support system
    """
    logger.info("Initializing Clinical Risk Modeling Engine...")
    
    # Import required modules
    from models.diversion_detection import diversion_model
    from services.xai_service import xai_service
    from data.loader import data_loader
    
    # Initialize model
    logger.info("Loading ML model...")
    diversion_model.load_model()
    
    # Initialize XAI service
    logger.info("Initializing XAI service...")
    try:
        xai_service.initialize()
    except Exception as e:
        logger.warning(f"Could not initialize XAI service: {e}")
    
    # Load sample data if needed
    logger.info("System initialization completed")
    
    return {
        'model': diversion_model,
        'xai_service': xai_service,
        'data_loader': data_loader
    }

def run_api_server():
    """
    Run the FastAPI server
    """
    import uvicorn
    from main import app
    
    logger.info("Starting FastAPI server...")
    logger.info("Clinical Risk Modeling Engine is now running on http://0.0.0.0:8000")
    logger.info("API documentation available at http://0.0.0.0:8000/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )

def run_demo():
    """
    Run a demonstration of the system
    """
    logger.info("Running system demonstration...")
    
    # Initialize system components
    system = initialize_system()
    model = system['model']
    xai_service = system['xai_service']
    data_loader = system['data_loader']
    
    # Generate some synthetic patient data
    logger.info("Generating synthetic patient data for demonstration...")
    patients = data_loader.load_synthetic_data(size=5)
    
    # Import risk calculator
    from services.risk_calculator import risk_calculator
    
    logger.info("Calculating risk for sample patients...")
    for i, patient in enumerate(patients[:3]):  # Only first 3 for demo
        try:
            risk_score, red_flags = risk_calculator.calculate_diversion_risk(patient)
            
            print(f"\n--- Patient {i+1}: {patient['name']} ---")
            print(f"Age: {patient['age']}, Gender: {patient['gender']}")
            print(f"Risk Score: {risk_score:.4f}")
            print(f"Risk Level: {'CRITICAL' if risk_score >= 0.8 else 'HIGH' if risk_score >= 0.6 else 'MEDIUM' if risk_score >= 0.4 else 'LOW'}")
            print(f"Red Flags: {len(red_flags)}")
            for flag in red_flags[:3]:  # Show first 3 flags
                print(f"  - {flag}")
            if len(red_flags) > 3:
                print(f"  ... and {len(red_flags) - 3} more")
                
        except Exception as e:
            logger.error(f"Error processing patient {patient['patient_id']}: {e}")
    
    print(f"\nDemonstration completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    setup_environment()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            run_demo()
        elif sys.argv[1] == "api":
            run_api_server()
        else:
            print("Usage: python run_system.py [demo|api]")
            print("  demo - Run system demonstration")
            print("  api  - Start API server")
    else:
        # Default: ask user what they want to do
        print("Clinical Risk Modeling Engine for Medication Diversion Detection")
        print("1. Run demonstration (demo)")
        print("2. Start API server (api)")
        choice = input("Enter your choice (demo/api): ").strip().lower()
        
        if choice == "demo":
            run_demo()
        elif choice == "api":
            run_api_server()
        else:
            print("Invalid choice. Usage: python run_system.py [demo|api]")


if __name__ == "__main__":
    main()