# antifragile_vbnf/experiments/banana_experiment.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zuko
from typing import Dict, Any #

from data.generators import generate_banana_dataset, generate_semicircle
from core.training import train_flow, train_flow_antifragile_centered
from evaluation.performance import evaluate_performance_under_disruption
from visualization.plots import plot_flow_samples
from evaluation.antifragile_testing_suite import run_antifragile_tests
class BananaExperiment:
    """Main experiment class for banana-shaped data."""
    
    def __init__(self, config=None, device: str = "cpu"):
        self.config = config
        self.device = device
        
    def run_experiment(self) -> Dict[str, Any]:
        """Run experiment comparing standard and antifragile vbnf on banana data."""
        
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
        
        # Train standard vbnf
        print("\nTraining standard vbnf...")
        standard_flow = zuko.flows.NSF(features=2, context=1, transforms=5, hidden_features=[64, 64])
        standard_flow, standard_losses = train_flow(
            standard_flow, nominal_data, target_data, 
            context_dim=1, num_steps=num_steps, lr=lr, subsets=subsets, 
            use_antifragile=False, device=self.device
        )
        
        # Train antifragile vbnf
        print("\nTraining antifragile vbnf...")
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
        

        print("\nRunning antifragile testing suite for antifragile vbnf...")

        _ = run_antifragile_tests(
                        trained_flow=antifragile_flow,
                        nominal_data=nominal_data, 
                        target_data=target_data,
                        context_dim=1
                    )
        
        print("\nRunning antifragile testing suite for standard vbnf...")

        _ = run_antifragile_tests(
                        trained_flow=standard_flow,
                        nominal_data=nominal_data, 
                        target_data=target_data,
                        context_dim=1
                    )


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
        
        # Plot standard vbnf samples
        ax = axs[2]
        self._plot_flow_samples_on_axis(ax, standard_flow, nominal_data, target_data, "(c) Standard vbnf")
        
        # Skip subfigure (d) as mentioned in the request
        ax = axs[3]
        ax.set_visible(False)
        
        # Plot antifragile vbnf samples
        ax = axs[4]
        self._plot_flow_samples_on_axis(ax, antifragile_flow, nominal_data, target_data, "(e) Antifragile vbnf")
        
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
        if 'antifragile_gain' in antifragile_losses.columns:
            ax.plot(antifragile_losses['step'], antifragile_losses['antifragile_gain'], label='Antifragile Term')
        elif 'volatility_gain' in antifragile_losses.columns:
            ax.plot(antifragile_losses['step'], antifragile_losses['volatility_gain'], label='Volatility Gain')
        ax.set_title("Antifragile Loss Component")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.legend()
        
        # Plot samples from both flows
        ax = axs[1, 0]
        plot_flow_samples(standard_flow, [0.0, 0.5, 1.0], ax, title="Standard vbnf Samples", device=self.device)
        
        ax = axs[1, 1]
        plot_flow_samples(antifragile_flow, [0.0, 0.5, 1.0], ax, title="Antifragile vbnf Samples", device=self.device)
        
        plt.tight_layout()
        plt.savefig("banana_analysis.png", dpi=300)
        plt.close()
        
        # Plot performance under disruption
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(std_magnitudes, std_log_probs, 'b-', label="Standard vbnf", linewidth=2)
        ax.plot(anti_magnitudes, anti_log_probs, 'r-', label="Antifragile vbnf", linewidth=2)
        
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


