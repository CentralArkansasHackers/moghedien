#### setup.py
```python
from setuptools import setup, find_packages

setup(
    name="moghedien",
    version="0.1.0",
    description="AI-driven attack path analysis for Active Directory",
    author="Security Team",
    packages=find_packages(),
    install_requires=[
        "networkx>=2.8.0",
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "torch>=1.12.0",
        "stable-baselines3>=1.6.0",
        "neo4j>=5.3.0",
        "click>=8.1.0",
        "rich>=12.0.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.3.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "moghedien=moghedien.cli.main:cli",
        ],
    },
)
