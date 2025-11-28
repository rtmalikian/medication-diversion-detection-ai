# Clinical Risk Modeling Engine for Medication Diversion Detection | AI-Powered Opioid Diversion Prevention Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/rtmalikian/diversion-detection/graphs/commit-activity)

## ⚠️ IMPORTANT DISCLAIMER & INTELLECTUAL PROPERTY NOTICE
**THIS IS A PROTOTYPE DEMONSTRATION ONLY. This software is NOT intended for clinical use, is not designed to treat or diagnose any medical condition, and should not be used for actual clinical decision making. This prototype is for demonstration and research purposes only. Any use for actual patient care is strictly prohibited until proper validation, regulatory approval, and clinical trials have been completed.**

**INTELLECTUAL PROPERTY RIGHTS**: This Clinical Risk Modeling Engine is the intellectual property of Raphael Tomas Malikian. Any use of this tool must provide proper attribution to the creator. Commercial use is strictly prohibited without explicit written permission and licensing. For permission inquiries, please contact the author directly.

## Overview

The **Clinical Risk Modeling Engine** is an AI-powered clinical decision support tool (CDST) designed to proactively identify and flag high-risk polypharmacy and potential medication safety/diversion scenarios. Built with machine learning and evidence-based guidelines from DEA, CDC, and SAMHSA, this system addresses the critical challenge of prescription drug diversion, particularly opioids and controlled substances.

### Available in Two Formats:
1. **Full Enterprise System**: Backend API with EHR integration capabilities (Under Development)
2. **Standalone Web Tool**: Browser-based assessment tool (Prototype Ready)

## Key Features

### 🔍 Comprehensive Risk Assessment
- **Evidence-based evaluation framework** using DEA, CDC, and SAMHSA guidelines
- **XAI (Explainable AI)** with SHAP/LIME analysis for clinical transparency
- **FastAPI-based REST API** for enterprise deployment (Planned)
- **Standalone web-based assessment tool** for immediate use
- **Machine learning model** for medication diversion detection using Gradient Boosting
- **Professional verification system** for enhanced security

### 📊 Risk Factors Considered

#### Clinical Indicators
- **Substance abuse history** - History of drug or alcohol abuse
- **Mental health history** - Depression, anxiety, PTSD, or other mental health conditions
- **Chronic pain history** - Long-term pain conditions requiring medication
- **Pain out of proportion to physical exam** - Pain levels inconsistent with physical findings

#### Prescription Patterns
- **Opioid prescriptions** - Number and frequency of opioid medications
- **Benzodiazepine prescriptions** - Count and duration of benzodiazepine use
- **High-dose prescriptions** - Medications exceeding recommended dosages
- **Concurrent prescriptions** - Multiple high-risk medications simultaneously
- **Early refill requests** - Requesting refills before scheduled time (Red Flag)
- **Unusual requests** - Early refills, large quantities, specific formulations
- **Prescription frequency** - Number of prescriptions per month

#### Healthcare Utilization
- **Emergency department visits** - Frequent ED visits in pattern recognition
- **Hospitalization history** - Hospital admissions and inpatient stays
- **Primary care visits** - Regular healthcare engagement patterns
- **Emergency requests** - Unscheduled or urgent care visits
- **Lost/stolen reports** - Number of reported lost/stolen prescriptions

#### Provider and Pharmacy Patterns
- **Number of prescribers** - Multiple healthcare providers prescribing controlled substances
- **Number of pharmacies** - Using multiple pharmacies (pharmacy hopping)
- **Geographic dispersion** - Distance between prescribers and pharmacies
- **Travel distance to provider** - Long distances traveled for prescriptions
- **Travel distance to pharmacy** - Extended travel for medication pickup
- **Prescriber pressure tactics** - Aggressive demands or threats toward providers
- **Prescription shopping patterns** - Visiting multiple providers for same medications

#### Insurance and Payment
- **Insurance status** - Uninsured or cash-paying patients
- **Payment method** - Cash payments as a red flag for avoiding tracking
- **Doctor shopping admissions** - Patient self-reported shopping behavior
- **Diversion admissions** - Patient acknowledgment of medication diversion

#### Behavioral Indicators
- **Provider clinical intuition** - Gut feeling assessment by healthcare provider
- **Inconsistent complaints** - Changing reasons for medication requests
- **Time since last prescription** - Short intervals between similar prescriptions
- **History of diversion** - Previous documented medication diversion incidents

## System Architecture

### Full Enterprise System (Under Development)
- **Backend**: Python, FastAPI, scikit-learn, pandas
- **Database**: SQLAlchemy ORM with PostgreSQL/MySQL support
- **Machine Learning**: Gradient Boosting, XAI with SHAP/LIME
- **API**: RESTful endpoints with authentication
- **Integration**: EHR connectivity (EPIC, Cerner, Allscripts)
- **PDMP Integration**: Prescription drug monitoring program connections

### Standalone Web Tool (Prototype Ready)
- **Frontend**: HTML, CSS, JavaScript (No server required)
- **Security**: Professional verification with license validation
- **Assessment**: Dual-level (Basic/Detailed) patient risk evaluation
- **Offline Capability**: Functions without internet connection

## Technologies & Stack

### Enterprise System
- **Backend**: Python 3.8+, FastAPI, Uvicorn
- **Machine Learning**: scikit-learn, XGBoost, SHAP, LIME
- **Data Processing**: pandas, NumPy
- **Database**: SQLAlchemy, PostgreSQL
- **Authentication**: OAuth2, JWT tokens
- **Containerization**: Docker, Docker Compose

### Standalone Prototype
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **No External Dependencies**: Pure browser-based operation

## Installation (Enterprise System - Planned)

For the full enterprise system (currently under development):

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Full Enterprise System (Planned)
```bash
python run_system.py api
```
Or directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Standalone Prototype (Ready Now)
1. Download `ui_demo.html` from the repository
2. Open in any modern web browser
3. Complete professional verification with medical license
4. Begin patient risk assessments

### Running a Demo
```bash
python run_system.py demo
```

## Project Status

### ✅ Standalone Web Tool - Ready for Use
- [x] Complete risk factor integration
- [x] Professional verification system
- [x] Evidence-based risk scoring
- [x] XAI feature importance visualization
- [x] Dual assessment levels (Basic/Detailed)
- [x] Comprehensive risk factor coverage
- [x] Mandatory disclaimer popup
- [x] Research citations and accountability

### 🔄 Full Enterprise System - Under Development
- [ ] Backend API integration
- [ ] EHR connectivity (EPIC, Cerner, Allscripts)
- [ ] PDMP (Prescription Drug Monitoring Program) integration
- [ ] Database connectivity and management
- [ ] Production-level security implementation
- [ ] Clinical validation and testing
- [ ] Regulatory compliance preparation

## Project Structure
```
diversion/
├── api/                 # API endpoints (Enterprise system)
├── data/               # Data processing modules
├── models/             # ML models and training algorithms
├── services/           # Business logic and service layers
├── utils/              # Utility functions and helpers
├── tests/              # Unit and integration tests
├── ui_demo.html        # Standalone web assessment tool (Prototype Ready)
├── requirements.txt    # Python dependencies
├── requirements-dev.txt # Development dependencies
├── LICENSE             # MIT License
├── README.md           # Project documentation
├── CHANGELOG.md        # Version history
├── CONTRIBUTING.md     # Contribution guidelines
├── EHR_INTEGRATION.md  # EHR integration planning
├── PUBLIC_DATA_SOURCES.md # Testing data resources
└── main.py             # Main application entry point
```

## API Documentation (Enterprise System - Planned)
After deploying the full system, API documentation will be available at `http://localhost:8000/docs`

## Standalone Tool Features

### Professional Verification System
- Medical license number verification
- Provider name and specialty validation
- Prevents unauthorized access by non-medical professionals

### Dual Assessment Levels
- **Basic Assessment**: Quick screening with core risk factors
- **Detailed Assessment**: Comprehensive evaluation with all risk indicators

### Risk Scoring Algorithm
- Evidence-based risk calculation
- Weighted risk factors based on research
- Feature importance visualization
- Red flag identification and reporting

### Security Measures
- Mandatory professional verification
- Protection against cherry-picking of potential patients
- Clinical decision support for legitimate healthcare providers only

## Testing
Run the test suite:
```bash
python -m pytest tests/
```

## Data Sources for Testing and Validation
For testing and validation, this project can utilize various publicly available healthcare datasets. See [PUBLIC_DATA_SOURCES.md](PUBLIC_DATA_SOURCES.md) for a comprehensive list of resources including:
- Government health datasets (HCUP, CDC, SAMHSA)
- Synthetic patient data generators (Synthea)
- Clinical research datasets (MIMIC, PhysioNet)
- Substance abuse and opioid-related statistics

## Research Citations and Accountability
This tool incorporates evidence-based risk factors from peer-reviewed research including:
- CDC Guidelines for Prescribing Opioids for Chronic Pain
- DEA guidelines on controlled substance diversion
- SAMHSA substance use indicators
- Studies on prescription drug monitoring programs
- Research on "doctor shopping" behavior
- Pain management and addiction studies

All risk factors are weighted based on clinical evidence and research findings.

## Creator Information

**Raphael Tomas Malikian**
Email: rtmalikian@gmail.com

Raphael has experience working in the Family Medicine field as a provider and has lived experience of how damaging drug diversion is on providers, patients, pharmacies, and the greater community in the context of the opioid epidemic and other controlled substance diversion. This personal and professional experience has informed the development of this Clinical Risk Modeling Engine to help identify potential medication diversion risks.

## Contributing
We welcome contributions to enhance the Clinical Risk Modeling Engine. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to participate.

## Acknowledgments

This project was created with the assistance of **Qwen Code in Visual Studio Code** using the coder model. The AI coding assistant helped with code generation, debugging, documentation, and project structure development.

## License
MIT License (see LICENSE file)

## Keywords
- medication diversion detection
- opioid abuse prevention
- prescription drug monitoring
- clinical decision support
- AI healthcare tool
- drug diversion prevention
- controlled substance monitoring
- physician safety
- patient safety
- prescription safety
- healthcare AI
- machine learning healthcare
- opioid crisis solution
- drug abuse detection
- medical informatics
- clinical risk assessment