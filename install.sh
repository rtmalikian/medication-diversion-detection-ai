#!/bin/bash
# Installation script for Clinical Risk Modeling Engine

echo "Setting up Clinical Risk Modeling Engine..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "Installation complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To start the API server, run: python run_system.py api"
echo "To run a demo, run: python run_system.py demo"