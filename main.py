# main.py
import argparse
import sys
from config import get_config
from experiments.banana_experiment import run_banana_experiment
from experiments.stress_experiments import run_all_stress_tests, run_multiple_experiments
from analysis.antifragile_analysis import AntifragileExperimentRunner

def main():
    """Main entry point for the antifragile vbnf experiments."""
    parser = argparse.ArgumentParser(description='Antifragile vbnf Experiments')
    
    parser.add_argument('--experiment', type=str, default='banana',
                       choices=['banana', 'stress_test', 'antifragile_focused', 'multiple_runs'],
                       help='Type of experiment to run')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration name or path to config file')
    
    parser.add_argument('--num_runs', type=int, default=10,
                       help='Number of runs for statistical analysis')
    
    parser.add_argument('--vis', action='store_true',
                       help='Enable visualizations')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    parser.add_argument('--save_models', action='store_true',
                       help='Save trained models')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load configuration
    if args.config:
        if args.config.endswith('.json'):
            from config import ExperimentConfig
            config = ExperimentConfig.load(args.config)
        else:
            config = get_config(args.config)
    else:
        # Use default config based on experiment type
        if args.experiment in ['banana']:
            config = get_config('banana')
        elif args.experiment in ['stress_test', 'multiple_runs']:
            config = get_config('stress_test')
        elif args.experiment == 'antifragile_focused':
            config = get_config('banana')  # Use banana as base
        else:
            config = get_config('quick_test')
    
    print(f"Running experiment: {args.experiment}")
    print(f"Configuration: {config.name}")
    
    # Run the appropriate experiment
    if args.experiment == 'banana':
        results = run_banana_experiment(config, device)
        print("Banana experiment completed successfully!")
        
    elif args.experiment == 'stress_test':
        results = run_all_stress_tests(config, args.vis, device)
        print("Stress test completed successfully!")
        
    elif args.experiment == 'multiple_runs':
        all_metrics, stats_results = run_multiple_experiments(
            args.num_runs, config, save_results=True, device=device
        )
        print(f"Multiple runs experiment ({args.num_runs} runs) completed successfully!")
        
    elif args.experiment == 'antifragile_focused':
        runner = AntifragileExperimentRunner()
        all_metrics, antifragile_analysis = runner.run_antifragile_focused_tests(
            vis_flg=args.vis, num_runs=args.num_runs
        )
        
        # Print comprehensive summary
        runner.analyzer.print_antifragile_summary(antifragile_analysis)
        print(f"Antifragile-focused experiment ({args.num_runs} runs) completed successfully!")
    
    print("\nExperiment completed! Check the generated files and visualizations.")

if __name__ == "__main__":
    main()


# antifragile_vbnf/__init__.py
"""
Antifragile Conditional Autoregressive Latent Normalizing Flows (A-vbnf)

A comprehensive framework for training and evaluating antifragile normalizing flows
that benefit from volatility and stress rather than being harmed by them.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main components for easy access
from .data.generators import generate_banana_dataset, generate_semicircle
from .core.training import train_flow, train_flow_antifragile_centered
from .core.antifragility import AntifragilityCalculator, AdaptiveWeightCalculator
from .evaluation.stress_tests import StressTester
from .analysis.statistics import extract_metric_values, compute_antifragile_stats
from .analysis.antifragile_analysis import calculate_antifragile_metrics, analyze_antifragile_results
from .visualization.plots import plot_flow_samples, visualize_stress_test_results
from .visualization.dashboards import create_antifragile_visualizations
from .experiments.banana_experiment import run_banana_experiment
from .experiments.stress_experiments import run_all_stress_tests, run_multiple_experiments

# Define what gets imported with "from antifragile_vbnf import *"
__all__ = [
    # Data generation
    'generate_banana_dataset',
    'generate_semicircle',
    
    # Core training
    'train_flow',
    'train_flow_antifragile_centered',
    'AntifragilityCalculator',
    'AdaptiveWeightCalculator',
    
    # Evaluation
    'StressTester',
    
    # Analysis
    'extract_metric_values',
    'compute_antifragile_stats',
    'calculate_antifragile_metrics',
    'analyze_antifragile_results',
    
    # Visualization
    'plot_flow_samples',
    'visualize_stress_test_results',
    'create_antifragile_visualizations',
    
    # Experiments
    'run_banana_experiment',
    'run_all_stress_tests',
    'run_multiple_experiments',
]


# antifragile_vbnf/data/__init__.py
"""Data generation utilities for antifragile vbnf experiments."""

from .generators import (
    generate_banana_dataset,
    generate_semicircle,
    BananaDataGenerator,
    SyntheticDataGenerator
)

__all__ = [
    'generate_banana_dataset',
    'generate_semicircle', 
    'BananaDataGenerator',
    'SyntheticDataGenerator'
]


# antifragile_vbnf/core/__init__.py
"""Core training and antifragility components."""

from .training import train_flow, train_flow_antifragile_centered, FlowTrainer
from .antifragility import AntifragilityCalculator, AdaptiveWeightCalculator
from .losses import AntifragileLossCalculator

__all__ = [
    'train_flow',
    'train_flow_antifragile_centered',
    'FlowTrainer',
    'AntifragilityCalculator', 
    'AdaptiveWeightCalculator',
    'AntifragileLossCalculator'
]


# antifragile_vbnf/evaluation/__init__.py
"""Evaluation and testing utilities."""

from .stress_tests import StressTester
from .advanced_tests import AdvancedAntifragilityTester

try:
    from .performance import evaluate_performance_under_disruption
    __all__ = ['StressTester', 'AdvancedAntifragilityTester', 'evaluate_performance_under_disruption']
except ImportError:
    __all__ = ['StressTester', 'AdvancedAntifragilityTester']


# antifragile_vbnf/analysis/__init__.py
"""Analysis and statistics utilities."""

from .statistics import StatisticalAnalyzer, extract_metric_values, compute_antifragile_stats
from .antifragile_analysis import (
    AntifragileAnalyzer, 
    AntifragileExperimentRunner,
    calculate_antifragile_metrics,
    analyze_antifragile_results
)

__all__ = [
    'StatisticalAnalyzer',
    'extract_metric_values',
    'compute_antifragile_stats',
    'AntifragileAnalyzer',
    'AntifragileExperimentRunner', 
    'calculate_antifragile_metrics',
    'analyze_antifragile_results'
]


# antifragile_vbnf/visualization/__init__.py
"""Visualization utilities."""

from .plots import FlowVisualizer, StressTestVisualizer, plot_flow_samples, visualize_stress_test_results
from .dashboards import AntifragileDashboard, create_antifragile_visualizations

__all__ = [
    'FlowVisualizer',
    'StressTestVisualizer',
    'plot_flow_samples',
    'visualize_stress_test_results',
    'AntifragileDashboard',
    'create_antifragile_visualizations'
]


# antifragile_vbnf/experiments/__init__.py
"""Experiment runners and utilities."""

from .banana_experiment import BananaExperiment, run_banana_experiment
from .stress_experiments import StressExperimentRunner, run_all_stress_tests, run_multiple_experiments

__all__ = [
    'BananaExperiment',
    'run_banana_experiment',
    'StressExperimentRunner', 
    'run_all_stress_tests',
    'run_multiple_experiments'
]