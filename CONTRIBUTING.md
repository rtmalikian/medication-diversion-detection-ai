# Contributing to Clinical Risk Modeling Engine

Thank you for your interest in contributing to the Clinical Risk Modeling Engine! This project aims to improve patient safety by detecting potential medication diversion.

## Code of Conduct

Please follow our Code of Conduct to foster an open and welcoming environment.

## How to Contribute

### Reporting Bugs
- Use the issue tracker to report bugs
- Describe the issue clearly with steps to reproduce
- Include environment information (OS, Python version, etc.)

### Suggesting Features
- Open an issue to discuss new features before implementing
- Explain the use case and benefits
- Consider the clinical impact and regulatory requirements

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation if needed
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a pull request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/clinical-risk-modeling-engine.git
cd clinical-risk-modeling-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all functions, classes, and modules
- Keep functions and methods focused and small

## Clinical Considerations

This is a clinical decision support tool. When contributing:

- Ensure all changes maintain clinical accuracy
- Consider regulatory requirements (HIPAA, FDA, etc.)
- Maintain explainability and interpretability
- Preserve patient privacy in all examples and tests
- Follow evidence-based guidelines (DEA, CDC, SAMHSA)