# antifragile_vbnf/experiments/stress_experiments.py
import torch
import numpy as np
import pyro
from typing import Dict, List, Tuple

from evaluation.stress_tests import StressTester
from data.generators import generate_banana_dataset
from core.training import train_flow, train_flow_antifragile_centered
from analysis.statistics import StatisticalAnalyzer
from analysis.antifragile_analysis import AntifragileAnalyzer
from visualization.dashboards import create_antifragile_visualizations
import zuko

class StressExperimentRunner:
    """Runner for comprehensive stress testing experiments."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.stress_tester = StressTester(device)
        self.stats_analyzer = StatisticalAnalyzer()
        self.antifragile_analyzer = AntifragileAnalyzer()
    
    def run_all_stress_tests(self, config=None, vis_flg: bool = True) -> Dict:
        """Run all stress tests and generate a comprehensive report."""
        print("Starting comprehensive stress testing for Antifragile vbnf...")
        
        # Use config if provided, otherwise use defaults
        if config:
            n_nominal = config.data.n_nominal
            n_target = config.data.n_target
            num_steps = config.training.num_steps
            lr = config.training.lr
            subsets = config.training.subsets
        else:
            n_nominal, n_target = 1000, 5
            num_steps, lr, subsets = 800, 1e-3, 5
        
        # Load or train models first
        data = generate_banana_dataset(n_nominal=n_nominal, n_target=n_target)
        nominal_data = data['nominal'].to(self.device)
        target_data = data['target'].to(self.device)
        
        # Train the models
        print("Training models...")
        standard_flow, antifragile_flow = self._train_models(
            nominal_data, target_data, num_steps, lr, subsets
        )
        
        # Run all stress tests
        print("Running stress tests...")
        all_results = self.stress_tester.run_all_stress_tests(
            standard_flow, antifragile_flow, nominal_data, target_data, vis_flg
        )
        
        return all_results
    
    def run_multiple_experiments(self, num_runs: int = 20, config=None, 
                                save_results: bool = True) -> Tuple[Dict, List[Dict]]:
        """Run multiple experiments for statistical analysis."""
        print(f"Running {num_runs} experiments for statistical analysis...")
        
        all_run_metrics = {}
        
        for i in range(num_runs):
            # Set seeds for reproducibility
            seed_value = i
            torch.manual_seed(seed_value)
            np.random.seed(seed_value)
            pyro.set_rng_seed(seed_value)
            
            print(f"Run {i+1}/{num_runs}")
            
            # Run stress tests for this run
            all_results = self.run_all_stress_tests(config, vis_flg=False)
            
            # Extract metrics
            extracted_metrics = self.stats_analyzer.extract_metric_values(all_results)
            
            # Add antifragile-specific metrics
            antifragile_metrics = self.antifragile_analyzer.calculate_antifragile_metrics(all_results)
            extracted_metrics.update(antifragile_metrics)
            
            # Collect results
            for metric_name, metric_data in extracted_metrics.items():
                if metric_name not in all_run_metrics:
                    all_run_metrics[metric_name] = {'std': [], 'anti': []}
                
                if len(metric_data['std']) > 0:
                    all_run_metrics[metric_name]['std'].append(np.mean(metric_data['std']))
                if len(metric_data['anti']) > 0:
                    all_run_metrics[metric_name]['anti'].append(np.mean(metric_data['anti']))
        
        # Statistical analysis
        stats_results = self.stats_analyzer.analyze_multiple_runs(all_run_metrics)
        
        # Antifragile-focused analysis
        antifragile_analysis = self.antifragile_analyzer.analyze_antifragile_results(all_run_metrics)
        
        if save_results:
            self._save_results(stats_results, all_run_metrics, antifragile_analysis)
        
        # Print summary
        self._print_summary(stats_results)
        self.antifragile_analyzer.print_antifragile_summary(antifragile_analysis)
        
        return all_run_metrics, stats_results
    
    def _train_models(self, nominal_data, target_data, num_steps, lr, subsets):
        """Train both standard and antifragile models."""
        # Train standard model
        standard_flow = zuko.flows.NSF(features=2, context=1, transforms=5, hidden_features=[64, 64])
        standard_flow, _ = train_flow(
            standard_flow, nominal_data, target_data, 
            context_dim=1, num_steps=num_steps, lr=lr, subsets=subsets, 
            use_antifragile=False, device=self.device
        )
        
        # Train antifragile model
        antifragile_flow = zuko.flows.NSF(features=2, context=1, transforms=5, hidden_features=[64, 64])
        antifragile_flow, _ = train_flow_antifragile_centered(
            antifragile_flow, nominal_data, target_data, 
            context_dim=1, num_steps=num_steps, lr=lr, subsets=subsets,
            device=self.device
        )
        
        return standard_flow, antifragile_flow
    
    def _save_results(self, stats_results, all_run_metrics, antifragile_analysis):
        """Save results to files."""
        import pandas as pd
        import json
        
        # Save statistical results
        df_results = pd.DataFrame(stats_results)
        df_results.to_csv('experiment_results.csv', index=False)
        
        # Save detailed metrics (convert to JSON-serializable format)
        metrics_to_save = {}
        for metric, data in all_run_metrics.items():
            metrics_to_save[metric] = {
                'std': [float(x) for x in data['std']],
                'anti': [float(x) for x in data['anti']]
            }
        
        with open('detailed_metrics.json', 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        # Save antifragile analysis
        with open('antifragile_analysis.json', 'w') as f:
            # Convert tensors and other non-serializable objects to floats
            def convert_for_json(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_for_json(antifragile_analysis), f, indent=2)
        
        print("Results saved to:")
        print("- experiment_results.csv")
        print("- detailed_metrics.json") 
        print("- antifragile_analysis.json")
    
    def _print_summary(self, stats_results):
        """Print summary of statistical results."""
        print("\n" + "="*100)
        print(f"{'Metric':<25} {'Std Mean':<12} {'Anti Mean':<12} {'Improv %':<12} {'p-value':<12} {'Significant':<10} {'Better':<15}")
        print("="*100)
        
        for result in stats_results:
            print(f"{result['metric']:<25} {result['std_mean']:<12.4f} {result['anti_mean']:<12.4f} "
                  f"{result['improvement']:<12.2f} {result['p_value']:<12.4f} "
                  f"{'Yes' if result['is_significant'] else 'No':<10} {result['better_method']:<15}")
        
        # Calculate overall summary
        significant_improvements = [r for r in stats_results if r['is_significant'] and r['better_method'] == 'A-vbnf']
        significant_deteriorations = [r for r in stats_results if r['is_significant'] and r['better_method'] == 'Standard vbnf']
        
        print("\n" + "="*100)
        print(f"SUMMARY:")
        print(f"Metrics with significant improvement in A-vbnf: {len(significant_improvements)}/{len(stats_results)}")
        print(f"Metrics with significant deterioration in A-vbnf: {len(significant_deteriorations)}/{len(stats_results)}")
        print(f"Metrics with no significant difference: {len(stats_results) - len(significant_improvements) - len(significant_deteriorations)}/{len(stats_results)}")
        print("="*100)

# Factory functions for backward compatibility
def run_all_stress_tests(config=None, vis_flg: bool = True, device: str = "cpu") -> Dict:
    """Factory function to run all stress tests."""
    runner = StressExperimentRunner(device)
    return runner.run_all_stress_tests(config, vis_flg)

def run_multiple_experiments(num_runs: int = 20, config=None, save_results: bool = True,
                           device: str = "cpu") -> Tuple[Dict, List[Dict]]:
    """Factory function to run multiple experiments."""
    runner = StressExperimentRunner(device)
    return runner.run_multiple_experiments(num_runs, config, save_results)