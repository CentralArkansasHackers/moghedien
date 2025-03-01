# Moghedien

Moghedien is an AI-driven attack path analysis engine that enhances BloodHound's capabilities. It automates the discovery and execution of attack paths in Active Directory environments.

## Features

- AI-powered attack path generation
- Automated exploitation recommendations
- Risk scoring and prioritization
- Integration with BloodHound data

## Installation

```bash
pip install -e .
Usage
bashCopymoghedien analyze --data-dir /path/to/bloodhound/data
Development Setup
bashCopy# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -e ".[dev]"
