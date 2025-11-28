# Changelog
All notable changes to the Clinical Risk Modeling Engine for medication diversion detection will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-11-28
### Added
- Initial project structure with api, data, models, services, utils, and tests directories
- FastAPI application with main.py entry point
- Configuration management with config.py
- Requirements file with dependencies (FastAPI, scikit-learn, pandas, SHAP, LIME, etc.)
- ML model implementation using Gradient Boosting for diversion detection
- Data processing module for feature engineering
- Evidence-based evaluation framework using DEA, CDC, and SAMHSA guidelines
- XAI service with SHAP and LIME integration
- Database integration with SQLAlchemy
- API endpoints for risk assessment, patient data, and model management
- Risk calculation service with evidence-based and ML risk scoring
- Patient management service with CRUD operations
- Comprehensive README documentation
- License file (MIT)
- .gitignore for GitHub compatibility
- Contribution guidelines
- Setup configuration for Python packaging
- Public data sources documentation for testing
- EHR integration planning documentation
- Installation script and basic functionality tests
- Synthetic data generation capabilities
- API endpoint testing framework

## [0.1.1] - 2025-11-28
### Fixed
- Changed disclaimer from popup modal to persistent banner on every screen
- Fixed verifyProfessional function to properly show assessment section after verification
- Added proper error checking in verifyProfessional function with element existence validation
- Fixed syntax error in calculateFeatureImportance function that was preventing verification
### Changed
- Removed modal-based disclaimer system
- Updated CSS and JavaScript for persistent disclaimer display
- Enhanced verification flow with better element handling

## [0.0.1] - 2025-11-28
### Added
- Initial project creation with basic directory structure
- Core ML model architecture using Gradient Boosting
- Basic API endpoints for risk assessment
- Initial data processing capabilities
- Basic evaluation framework based on clinical guidelines