# Moghedien

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Moghedien is an AI-driven attack path analysis engine that enhances BloodHound's capabilities. It automates the discovery and execution of attack paths in Active Directory environments using advanced graph algorithms and machine learning techniques.

## Features

- **AI-powered attack path generation** - Automatically identify optimal attack paths through AD environments
- **Risk-based path scoring** - Score paths based on success probability, stealth, and complexity
- **Attack technique recommendations** - Get specific techniques for each step in an attack path
- **BloodHound integration** - Seamlessly work with BloodHound data to enhance red team operations
- **Interactive exploration** - Search and analyze the AD graph through a CLI interface

## Project Status

This project is currently in active development. Phase 1 of the development roadmap is nearing completion, with work on the AI decision engine in progress.

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/moghedien.git
cd moghedien

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e ".[dev]"
```

### Dependencies

The project requires several Python libraries, including:

- networkx - For graph representation and algorithms
- numpy - For numerical computations
- pandas - For data manipulation
- scikit-learn - For machine learning components
- torch - For deep learning models (reinforcement learning)
- neo4j - For database connectivity (optional)
- click and rich - For the CLI interface

All dependencies are automatically installed with the package.

## Usage

### Command Line Interface

```bash
# Analyze BloodHound data
moghedien analyze --data-dir /path/to/bloodhound/data

# Other commands will be added as development progresses
```

### Interactive Mode

After loading BloodHound data, Moghedien enters an interactive mode where you can:

1. Find paths to Domain Admin
2. Find paths to high-value targets
3. Search for specific objects in the AD graph
4. Exit the application

## Development Setup

For development work, install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_bloodhound_data.py
```

### Project Structure

- `moghedien/`
  - `bloodhound/` - BloodHound data parsing and loading
  - `cli/` - Command line interface
  - `core/` - Core functionality (graph, pathfinder, models)
  - `utils/` - Utility functions and logging

## Roadmap

See the [project plan](docs/project_plan.md) for the detailed development roadmap.

- Phase 1: BloodHound AI - Graph Analysis & Attack Path Automation (Months 0-3)
- Phase 2: Automation of Exploitation & Post-Exploitation (Months 4-6)
- Phase 3: Adaptive Command & Control (Months 7-9)
- Phase 4: AI-Powered Phishing & Social Engineering (Months 10-12)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and authorized security testing purposes only. Use responsibly and only on systems you have permission to test.