# antifragile_calnf/experiments/banana_experiment.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zuko
from typing import Dict, Any #

from ..data.generators import generate_banana_dataset, generate_semicircle
from ..core.training import train_flow, train_flow_antifragile_centered
from ..evaluation.performance import evaluate_performance_under_disruption
from ..visualization.plots import plot_flow_samples

class BananaExperiment:
    """Main experiment class for banana-shaped data."""
    
    def __init__(self, config=None, device: str = "cpu"):
        self.config = config
        self.device = device
        
    def run_experiment(self) -> Dict[str, Any]:
        """Run experiment comparing standard and antifragile CALNF on banana data."""
        
        # Use config if provided, otherwise use defaults
        if self.config:
            n_nominal = self.config.data.n_nominal
            n_target = self.config.data.n_target
            noise_scale = self.config.data.noise_scale
            horizontal_shift = self.config.data.horizontal_shift
            num_steps = self.config.training.num_steps
            lr = self.config.training.lr
            subsets = self.config.training.subsets
        else:
            n_nominal, n_target = 1000, 3
            noise_scale, horizontal_shift = 0.1, 1.0
            num_steps, lr, subsets = 800, 1e-3, 5
        
        print("Generating banana dataset...")
        data = generate_banana_dataset(n_nominal=n_nominal, n_target=n_target, 
                                      noise_scale=noise_scale, horizontal_shift=horizontal_shift)
        df = data['df']
        nominal_data = data['nominal'].to(self.device)
        target_data = data['target'].to(self.device)
        
        # Plot the dataset
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='x', y='y', hue='type', data=df)
        plt.title("Horizontally Shifted Banana-Shaped Dataset")
        plt.savefig("banana_dataset.png")
        plt.close()
        
        # Train standard CALNF
        print("\nTraining standard CALNF...")
        standard_flow = zuko.flows.NSF(features=2, context=1, transforms=5, hidden_features=[64, 64])
        standard_flow, standard_losses = train_flow(
            standard_flow, nominal_data, target_data, 
            context_dim=1, num_steps=num_steps, lr=lr, subsets=subsets, 
            use_antifragile=False, device=self.device
        )
        
        # Train antifragile CALNF
        print("\nTraining antifragile CALNF...")
        antifragile_flow = zuko.flows.NSF(features=2, context=1, transforms=5, hidden_features=[64, 64])
        antifragile_flow, antifragile_losses = train_flow_antifragile_centered(
            antifragile_flow, nominal_data, target_data,
            context_dim=1, num_steps=num_steps, lr=lr, subsets=subsets,
            device=self.device
        )
        
        # Evaluate under disruption
        print("\nEvaluating performance under disruption...")
        std_magnitudes, std_log_probs = evaluate_performance_under_disruption(
            standard_flow, nominal_data, target_data, device=self.device
        )
        anti_magnitudes, anti_log_probs = evaluate_performance_under_disruption(
            antifragile_flow, nominal_data, target_data, device=self.device
        )
        
        # Create comprehensive visualization
        self._create_comprehensive_visualization(
            nominal_data, target_data, standard_flow, antifragile_flow,
            standard_losses, antifragile_losses, std_magnitudes, std_log_probs,
            anti_magnitudes, anti_log_probs
        )
        
        # Calculate convexity metrics
        std_poly = np.polyfit(std_magnitudes, std_log_probs, 2)
        anti_poly = np.polyfit(anti_magnitudes, anti_log_probs, 2)
        std_convexity = std_poly[0]
        anti_convexity = anti_poly[0]
        
        print(f"\nVisualization saved as 'banana_comparison.png', 'banana_analysis.png', and 'banana_disruption.png'")
        print(f"Standard model convexity: {std_convexity:.4f}")
        print(f"Antifragile model convexity: {anti_convexity:.4f}")
        print(f"Antifragility ratio: {anti_convexity/std_convexity:.4f}")
        
        return {
            'standard_flow': standard_flow,
            'antifragile_flow': antifragile_flow,
            'standard_losses': standard_losses,
            'antifragile_losses': antifragile_losses,
            'convexity': {
                'standard': std_convexity,
                'antifragile': anti_convexity,
                'ratio': anti_convexity/std_convexity
            }
        }
    
    def _create_comprehensive_visualization(self, nominal_data, target_data, 
                                          standard_flow, antifragile_flow,
                                          standard_losses, antifragile_losses,
                                          std_magnitudes, std_log_probs,
                                          anti_magnitudes, anti_log_probs):
        """Create comprehensive visualization similar to Figure 1."""
        
        # Main comparison figure
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        fig.suptitle("Inference in data-constrained environments with Banana-shaped data", fontsize=16)
        
        # Plot ground truth
        ax = axs[0]
        nominal_gt = generate_semicircle([0, 0], 1.0, 1000, [0, np.pi], 0.1)
        target_gt = generate_semicircle([1.0, 0], 1.0, 1000, [np.pi, 2*np.pi], 0.1)
        ax.scatter(nominal_gt[:, 0], nominal_gt[:, 1], c='blue', s=1, alpha=0.5, label="Nominal")
        ax.scatter(target_gt[:, 0], target_gt[:, 1], c='red', s=1, alpha=0.5, label="Target")
        ax.set_title("(a) GT")
        ax.set_xlim(-2, 3)
        ax.set_ylim(-2, 2)
        
        # Plot imbalanced dataset
        ax = axs[1]
        ax.scatter(nominal_data[:, 0].cpu().numpy(), nominal_data[:, 1].cpu().numpy(), 
                  c='blue', s=1, alpha=0.5, label="Nominal")
        ax.scatter(target_data[:, 0].cpu().numpy(), target_data[:, 1].cpu().numpy(), 
                  c='red', s=1, alpha=0.5, label="Target")
        ax.set_title("(b) Imbalanced dataset")
        ax.set_xlim(-2, 3)
        ax.set_ylim(-2, 2)
        
        # Plot standard CALNF samples
        ax = axs[2]
        self._plot_flow_samples_on_axis(ax, standard_flow, nominal_data, target_data, "(c) Standard CALNF")
        
        # Skip subfigure (d) as mentioned in the request
        ax = axs[3]
        ax.set_visible(False)
        
        # Plot antifragile CALNF samples
        ax = axs[4]
        self._plot_flow_samples_on_axis(ax, antifragile_flow, nominal_data, target_data, "(e) Antifragile CALNF")
        
        plt.tight_layout()
        plt.savefig("banana_comparison.png", dpi=300)
        plt.close()
        
        # Additional analysis plots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot training losses
        ax = axs[0, 0]
        ax.plot(standard_losses['step'], standard_losses['nominal_loss'], label='Standard Nominal')
        ax.plot(standard_losses['step'], standard_losses['target_loss'], label='Standard Target')
        if 'nominal_loss' in antifragile_losses.columns:
            ax.plot(antifragile_losses['step'], antifragile_losses['nominal_loss'], label='Antifragile Nominal')
        if 'target_loss' in antifragile_losses.columns:
            ax.plot(antifragile_losses['step'], antifragile_losses['target_loss'], label='Antifragile Target')
        ax.set_title("Training Losses")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.legend()
        
        # Plot antifragile loss component
        ax = axs[0, 1]
        if 'antifragile_loss' in antifragile_losses.columns:
            ax.plot(antifragile_losses['step'], antifragile_losses['antifragile_loss'], label='Antifragile Term')
        elif 'volatility_gain' in antifragile_losses.columns:
            ax.plot(antifragile_losses['step'], antifragile_losses['volatility_gain'], label='Volatility Gain')
        ax.set_title("Antifragile Loss Component")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.legend()
        
        # Plot samples from both flows
        ax = axs[1, 0]
        plot_flow_samples(standard_flow, [0.0, 0.5, 1.0], ax, title="Standard CALNF Samples", device=self.device)
        
        ax = axs[1, 1]
        plot_flow_samples(antifragile_flow, [0.0, 0.5, 1.0], ax, title="Antifragile CALNF Samples", device=self.device)
        
        plt.tight_layout()
        plt.savefig("banana_analysis.png", dpi=300)
        plt.close()
        
        # Plot performance under disruption
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(std_magnitudes, std_log_probs, 'b-', label="Standard CALNF", linewidth=2)
        ax.plot(anti_magnitudes, anti_log_probs, 'r-', label="Antifragile CALNF", linewidth=2)
        
        # Fit polynomial to highlight convexity/concavity
        std_poly = np.polyfit(std_magnitudes, std_log_probs, 2)
        anti_poly = np.polyfit(anti_magnitudes, anti_log_probs, 2)
        x_poly = np.linspace(0, 1, 100)
        y_std_poly = np.polyval(std_poly, x_poly)
        y_anti_poly = np.polyval(anti_poly, x_poly)
        
        ax.plot(x_poly, y_std_poly, 'b--', alpha=0.7)
        ax.plot(x_poly, y_anti_poly, 'r--', alpha=0.7)
        
        # Calculate and display convexity coefficients
        std_convexity = std_poly[0]
        anti_convexity = anti_poly[0]
        ax.text(0.05, 0.95, f"Standard convexity: {std_convexity:.4f}", 
               transform=ax.transAxes, color='blue')
        ax.text(0.05, 0.90, f"Antifragile convexity: {anti_convexity:.4f}", 
               transform=ax.transAxes, color='red')
        
        ax.set_title("Performance Under Disruption")
        ax.set_xlabel("Disruption Magnitude")
        ax.set_ylabel("Log Probability")
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig("banana_disruption.png", dpi=300)
        plt.close()
    
    def _plot_flow_samples_on_axis(self, ax, flow, nominal_data, target_data, title):
        """Plot flow samples on a given axis."""
        with torch.no_grad():
            # Generate samples at different context values
            contexts = [0.0, 0.25, 0.5, 0.75, 1.0]
            colors = plt.cm.viridis(np.linspace(0, 1, len(contexts)))
            
            for i, context in enumerate(contexts):
                ctx = torch.tensor([[context]], device=self.device)
                dist = flow(ctx)
                samples = dist.sample((200,)).cpu().numpy()
                ax.scatter(samples[:, 0, 0], samples[:, 0, 1], c=[colors[i]], s=1, alpha=0.6)
        
        # Add reference data
        ax.scatter(nominal_data[:50, 0].cpu().numpy(), nominal_data[:50, 1].cpu().numpy(), 
                  c='blue', s=1, alpha=0.1)
        ax.scatter(target_data[:, 0].cpu().numpy(), target_data[:, 1].cpu().numpy(), 
                  c='red', s=1, alpha=0.1)
        ax.set_title(title)
        ax.set_xlim(-2, 3)
        ax.set_ylim(-2, 2)

# Factory function for backward compatibility
def run_banana_experiment(config=None, device: str = "cpu") -> Dict[str, Any]:
    """Factory function to run banana experiment."""
    experiment = BananaExperiment(config, device)
    return experiment.run_experiment()


# antifragile_calnf/experiments/stress_experiments.py
import torch
import numpy as np
import pyro
from typing import Dict, List, Tuple

from ..evaluation.stress_tests import StressTester
from ..data.generators import generate_banana_dataset
from ..core.training import train_flow, train_flow_antifragile_centered
from ..analysis.statistics import StatisticalAnalyzer
from ..analysis.antifragile_analysis import AntifragileAnalyzer
from ..visualization.dashboards import create_antifragile_visualizations
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
        print("Starting comprehensive stress testing for Antifragile CALNF...")
        
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
        significant_improvements = [r for r in stats_results if r['is_significant'] and r['better_method'] == 'A-CALNF']
        significant_deteriorations = [r for r in stats_results if r['is_significant'] and r['better_method'] == 'Standard CALNF']
        
        print("\n" + "="*100)
        print(f"SUMMARY:")
        print(f"Metrics with significant improvement in A-CALNF: {len(significant_improvements)}/{len(stats_results)}")
        print(f"Metrics with significant deterioration in A-CALNF: {len(significant_deteriorations)}/{len(stats_results)}")
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