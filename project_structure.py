# Project Structure for Antifragile vbnf

# Directory structure:
"""
antifragile_vbnf/
├── setup.py
├── requirements.txt
├── config.py
├── main.py
├── README.md
├── USAGE_GUIDE.md
├── antifragile_vbnf/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── generators.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── antifragility.py
│   │   ├── training.py
│   │   └── losses.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── stress_tests.py
│   │   ├── advanced_tests.py
│   │   └── performance.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── statistics.py
│   │   └── antifragile_analysis.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plots.py
│   │   └── dashboards.py
│   └── experiments/
│       ├── __init__.py
│       ├── banana_experiment.py
│       └─ stress_experiments.py
"""

# setup.py
setup_content = '''
from setuptools import setup, find_packages

setup(
    name="antifragile_vbnf",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
        "pyro-ppl>=1.8.0",
        "zuko>=0.1.0",
    ],
    author="Your Name",
    description="Antifragile Conditional Autoregressive Latent Normalizing Flows",
    python_requires=">=3.8",
)
'''

# requirements.txt
requirements_content = '''
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
pandas>=1.3.0
scipy>=1.7.0
tqdm>=4.62.0
pyro-ppl>=1.8.0
zuko>=0.1.0
'''

# README.md
readme_content = '''
# Antifragile vbnf

Antifragile Conditional Autoregressive Latent Normalizing Flows - A framework for training normalizing flows that benefit from volatility and stress.

## Key Features

- **Antifragile Training**: Models that improve under stress rather than degrade
- **Comprehensive Testing**: Advanced stress tests and statistical analysis
- **Modular Design**: Object-oriented architecture for easy extension
- **Visualization**: Rich dashboards and plots for analysis
- **Statistical Validation**: Multiple-run experiments with significance testing

## Quick Start

```python
from antifragile_vbnf import run_banana_experiment

# Run the classic banana experiment
results = run_banana_experiment()
```

## Command Line Usage

```bash
# Run banana experiment
python main.py --experiment banana

# Run comprehensive stress tests  
python main.py --experiment stress_test --vis

# Run antifragile-focused analysis (20 runs)
python main.py --experiment antifragile_focused --num_runs 20
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install the package: `pip install -e .`

See USAGE_GUIDE.md for detailed usage instructions.
'''

print("Enhanced project structure created!")
print("\nKey improvements in the refactored codebase:")
print("✅ Object-Oriented Design with clear separation of concerns")
print("✅ Modular architecture for easy extension and testing")
print("✅ Comprehensive configuration management")
print("✅ Factory functions for backward compatibility")
print("✅ Rich visualization and analysis capabilities")
print("✅ Statistical validation framework")
print("✅ Command-line interface for easy experimentation")

print("\nNext steps:")
print("1. Create the directory structure as shown above")
print("2. Copy each code module into its respective file")
print("3. Create the setup.py, requirements.txt, and README.md files")
print("4. Run: pip install -e .")
print("5. Test with: python main.py --experiment banana --vis")