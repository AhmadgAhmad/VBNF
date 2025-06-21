# Antifragile CALNF

**Antifragile Conditional Autoregressive Latent Normalizing Flows**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> *Training normalizing flows that benefit from volatility and stress rather than being harmed by them*

## 🎯 Overview

Antifragile CALNF implements Nassim Taleb's concept of antifragility in normalizing flows - creating models that don't just survive volatility and stress, but actually **improve** from it. Unlike traditional robust models that merely resist perturbations, antifragile models gain strength from disorder.

### Key Features

- 🛡️ **Antifragile Training**: Models that improve under stress rather than degrade
- 🧪 **Comprehensive Testing**: Advanced stress tests including black swan events, distribution shifts, and dynamic environments  
- 📊 **Statistical Validation**: Multi-run experiments with significance testing and effect size analysis
- 🎨 **Rich Visualization**: Interactive dashboards and publication-ready plots
- ⚙️ **Modular Design**: Object-oriented architecture for easy extension and research
- 🔧 **Easy Configuration**: Type-safe configuration management with predefined setups

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/antifragile-calnf.git
cd antifragile-calnf
pip install -e .
```

### Basic Usage

```python
from antifragile_calnf import run_banana_experiment

# Run the classic banana experiment
results = run_banana_experiment()
```

### Command Line Interface

```bash
# Quick banana experiment
python main.py --experiment banana --vis

# Comprehensive stress testing  
python main.py --experiment stress_test --vis

# Statistical analysis across 20 runs
python main.py --experiment antifragile_focused --num_runs 20

# Custom configuration
python main.py --experiment banana --config custom_config.json --device cuda
```

## 📖 Documentation

### Core Concepts

**Antifragility** in machine learning means creating models that:
- **Benefit from volatility** (positive Jensen's gap)
- **Improve under stress** rather than degrade
- **Exploit diversity** instead of being harmed by it
- **Prepare for tail events** beyond observed data

### Architecture

```
antifragile_calnf/
├── data/           # Data generation and augmentation
├── core/           # Training algorithms and antifragile loss functions  
├── evaluation/     # Stress testing and performance evaluation
├── analysis/       # Statistical analysis and antifragile metrics
├── visualization/  # Plotting and dashboard generation
└── experiments/    # High-level experiment runners
```

### Key Components

- **`FlowTrainer`**: Implements antifragile training algorithms
- **`StressTester`**: Comprehensive stress testing framework
- **`AntifragileAnalyzer`**: Specialized metrics for antifragile properties
- **`AntifragileDashboard`**: Rich visualization and reporting

## 🧪 Experiments & Results

### Banana Dataset Experiment

Classic antifragile demonstration on horizontally-shifted semicircular data:

```python
from antifragile_calnf.experiments import BananaExperiment
from config import get_config

experiment = BananaExperiment(get_config('banana'))
results = experiment.run_experiment()

print(f"Antifragility ratio: {results['convexity']['ratio']:.3f}")
```

### Stress Testing Suite

Comprehensive evaluation across multiple stress scenarios:

```python
from antifragile_calnf.evaluation import StressTester

tester = StressTester()
results = tester.run_all_stress_tests(standard_flow, antifragile_flow, 
                                     nominal_data, target_data)
```

**Stress Tests Include:**
- 📈 Distribution shift tests (horizontal shift, noise, deformation)
- 🦢 Black swan events (extreme outliers, unexpected clusters)  
- 🌊 Dynamic environments (progressive drift, oscillations)
- 📊 Transfer function analysis (Taleb's H-function)
- ⚡ Advanced antifragility tests (Jensen's gap, convexity, recovery speed)

### Statistical Analysis

Multi-run statistical validation with significance testing:

```python
from antifragile_calnf.experiments import run_multiple_experiments

all_metrics, stats_results = run_multiple_experiments(
    num_runs=50, save_results=True
)

# Automatic significance testing and effect size calculation
# Results saved to CSV with comprehensive statistical analysis
```

## 📊 Sample Results

![Banana Experiment](docs/images/banana_comparison.png)
*Comparison of standard vs antifragile CALNF on banana-shaped data*

![Stress Test Results](docs/images/stress_test_dashboard.png)
*Comprehensive stress testing dashboard showing antifragile benefits*

## 🛠️ Advanced Usage

### Custom Training

```python
from antifragile_calnf.core import FlowTrainer, AntifragileLossCalculator
import zuko

# Create custom trainer
trainer = FlowTrainer(device="cuda")

# Train with antifragile loss
flow = zuko.flows.NSF(features=2, context=1, transforms=5)
trained_flow, losses = trainer.train_flow_antifragile_centered(
    flow, nominal_data, target_data, num_steps=1000
)
```

### Custom Stress Tests

```python
from antifragile_calnf.evaluation import AdvancedAntifragilityTester

# Create custom stress tester
tester = AdvancedAntifragilityTester()

# Run specific antifragile tests
volatility_results = tester.test_volatility_response(
    standard_flow, antifragile_flow, nominal_data, target_data
)
```

### Configuration Management

```python
from config import ExperimentConfig, DataConfig, TrainingConfig

# Create custom configuration
config = ExperimentConfig(
    name="my_experiment",
    data=DataConfig(n_nominal=2000, n_target=100),
    training=TrainingConfig(num_steps=1500, lr=5e-4)
)

# Save for later use
config.save('my_experiment.json')

# Load and use
config = ExperimentConfig.load('my_experiment.json')
```

## 📈 Key Metrics

The framework evaluates antifragile properties through specialized metrics:

- **Jensen's Gap**: `E[f(stress)] - f(E[stress])` (positive = antifragile)
- **Convexity Response**: Second-order benefits from volatility
- **Stress Benefit Ratio**: Relative improvement under extreme conditions
- **Recovery Superiority**: Speed of adaptation after perturbations
- **Tail Preparedness**: Performance on events beyond training distribution

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/yourusername/antifragile-calnf.git
cd antifragile-calnf
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
pytest tests/
python -m pytest tests/ --cov=antifragile_calnf --cov-report=html
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{antifragile_calnf2024,
  title={Antifragile Conditional Autoregressive Latent Normalizing Flows},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/antifragile-calnf}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Nassim Nicholas Taleb's work on antifragility
- Built on PyTorch and the Zuko normalizing flows library
- Statistical analysis powered by SciPy and pandas

## 🔗 Related Work

- [Normalizing Flows](https://arxiv.org/abs/1505.05770)
- [Antifragile: Things That Gain from Disorder](https://www.goodreads.com/book/show/13530973-antifragile)
- [Zuko: Normalizing Flows in PyTorch](https://github.com/probabilists/zuko)

---

**Made with ❤️ for the machine learning research community**