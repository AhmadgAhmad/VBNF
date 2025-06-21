# antifragile_calnf/visualization/plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional, Tuple

class FlowVisualizer:
    """Visualizer for normalizing flows and their samples."""
    
    def __init__(self, style: str = "seaborn-v0_8"):
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
    def plot_flow_samples(self, flow, context_values: List[float], ax=None, 
                         n_samples: int = 1000, title: Optional[str] = None,
                         device: str = "cpu") -> plt.Axes:
        """Plot samples from the flow at different context values."""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        
        with torch.no_grad():
            for context in context_values:
                ctx = torch.tensor([[context]], device=device)
                dist = flow(ctx)
                samples = dist.sample((n_samples,)).cpu().numpy()
                ax.scatter(samples[:,0, 0], samples[:, 0,1], alpha=0.3, 
                          label=f"Context={context:.1f}")
        
        ax.legend()
        if title:
            ax.set_title(title)
        ax.set_xlim(-2, 3)  # Expanded to show shifted target class
        ax.set_ylim(-2, 2)
        
        return ax
    
    def plot_dataset(self, data_df: pd.DataFrame, title: str = "Dataset", 
                    ax=None) -> plt.Axes:
        """Plot the generated dataset."""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        
        sns.scatterplot(x='x', y='y', hue='type', data=data_df, ax=ax)
        ax.set_title(title)
        return ax
    
    def plot_training_losses(self, losses_df: pd.DataFrame, title: str = "Training Losses",
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot training losses over time."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title)
        
        # Main losses
        ax = axes[0, 0]
        if 'nominal_loss' in losses_df.columns:
            ax.plot(losses_df['step'], losses_df['nominal_loss'], label='Nominal Loss')
        if 'target_loss' in losses_df.columns:
            ax.plot(losses_df['step'], losses_df['target_loss'], label='Target Loss')
        if 'total_loss' in losses_df.columns:
            ax.plot(losses_df['step'], losses_df['total_loss'], label='Total Loss')
        ax.set_title("Main Losses")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
        
        # Antifragile components
        ax = axes[0, 1]
        antifragile_cols = [col for col in losses_df.columns if 'antifragile' in col]
        for col in antifragile_cols:
            ax.plot(losses_df['step'], losses_df[col], label=col.replace('antifragile_', ''))
        ax.set_title("Antifragile Components")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
        
        # Regularization
        ax = axes[1, 0]
        if 'reg_loss' in losses_df.columns:
            ax.plot(losses_df['step'], losses_df['reg_loss'], label='Regularization')
        ax.set_title("Regularization Loss")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
        
        # Total overview
        ax = axes[1, 1]
        if 'total_loss' in losses_df.columns:
            ax.plot(losses_df['step'], losses_df['total_loss'], 'k-', linewidth=2)
        ax.set_title("Total Loss Overview")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_performance_comparison(self, std_magnitudes: np.ndarray, std_log_probs: np.ndarray,
                                   anti_magnitudes: np.ndarray, anti_log_probs: np.ndarray,
                                   title: str = "Performance Under Disruption",
                                   figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Plot performance comparison under disruption."""
        fig, ax = plt.subplots(figsize=figsize)
        
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
        
        ax.set_title(title)
        ax.set_xlabel("Disruption Magnitude")
        ax.set_ylabel("Log Probability")
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig

class StressTestVisualizer:
    """Visualizer for stress test results."""
    
    def __init__(self):
        pass
    
    def plot_distribution_shift_results(self, results: Dict, 
                                       figsize: Tuple[int, int] = (12, 18)) -> plt.Figure:
        """Visualize distribution shift test results."""
        fig, axs = plt.subplots(3, 1, figsize=figsize)
        
        # Horizontal shift
        if 'horizontal_shift' in results:
            df = pd.DataFrame(results['horizontal_shift'])
            axs[0].plot(df['shift'], df['std_log_prob'], 'b-', marker='o', label='Standard CALNF')
            axs[0].plot(df['shift'], df['anti_log_prob'], 'r-', marker='s', label='Antifragile CALNF')
            axs[0].set_title('Performance Under Horizontal Shift')
            axs[0].set_xlabel('Shift Magnitude')
            axs[0].set_ylabel('Log Probability')
            axs[0].grid(True)
            axs[0].legend()
        
        # Noise variation
        if 'noise_variation' in results:
            df = pd.DataFrame(results['noise_variation'])
            axs[1].plot(df['noise'], df['std_log_prob'], 'b-', marker='o', label='Standard CALNF')
            axs[1].plot(df['noise'], df['anti_log_prob'], 'r-', marker='s', label='Antifragile CALNF')
            axs[1].set_title('Performance Under Increasing Noise')
            axs[1].set_xlabel('Noise Level')
            axs[1].set_ylabel('Log Probability')
            axs[1].grid(True)
            axs[1].legend()
        
        # Shape deformation
        if 'shape_deformation' in results:
            df = pd.DataFrame(results['shape_deformation'])
            axs[2].plot(df['ratio'], df['std_log_prob'], 'b-', marker='o', label='Standard CALNF')
            axs[2].plot(df['ratio'], df['anti_log_prob'], 'r-', marker='s', label='Antifragile CALNF')
            axs[2].set_title('Performance Under Shape Deformation')
            axs[2].set_xlabel('Deformation Ratio')
            axs[2].set_ylabel('Log Probability')
            axs[2].grid(True)
            axs[2].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_black_swan_results(self, results: Dict, 
                               figsize: Tuple[int, int] = (12, 12)) -> plt.Figure:
        """Visualize black swan test results."""
        fig, axs = plt.subplots(2, 1, figsize=figsize)
        
        # Single outlier
        if 'single_outlier' in results:
            df = pd.DataFrame(results['single_outlier'])
            axs[0].plot(df['magnitude'], df['std_impact_pct'], 'b-', marker='o', label='Standard CALNF')
            axs[0].plot(df['magnitude'], df['anti_impact_pct'], 'r-', marker='s', label='Antifragile CALNF')
            axs[0].set_title('Impact of Single Extreme Outlier')
            axs[0].set_xlabel('Outlier Magnitude')
            axs[0].set_ylabel('Impact on Log Probability (%)')
            axs[0].grid(True)
            axs[0].legend()
        
        # Outlier cluster
        if 'outlier_cluster' in results:
            df = pd.DataFrame(results['outlier_cluster'])
            axs[1].plot(df['cluster_size'], df['std_impact_pct'], 'b-', marker='o', label='Standard CALNF')
            axs[1].plot(df['cluster_size'], df['anti_impact_pct'], 'r-', marker='s', label='Antifragile CALNF')
            axs[1].set_title('Impact of Outlier Clusters')
            axs[1].set_xlabel('Cluster Size')
            axs[1].set_ylabel('Impact on Log Probability (%)')
            axs[1].grid(True)
            axs[1].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_dynamic_environment_results(self, results: Dict, 
                                        figsize: Tuple[int, int] = (12, 12)) -> plt.Figure:
        """Visualize dynamic environment test results."""
        fig, axs = plt.subplots(2, 1, figsize=figsize)
        
        # Distribution drift
        if 'distribution_drift' in results:
            df = pd.DataFrame(results['distribution_drift'])
            axs[0].plot(df['t'], df['std_log_prob'], 'b-', marker='o', label='Standard CALNF')
            axs[0].plot(df['t'], df['anti_log_prob'], 'r-', marker='s', label='Antifragile CALNF')
            axs[0].set_title('Performance Under Progressive Distribution Drift')
            axs[0].set_xlabel('Drift Parameter')
            axs[0].set_ylabel('Log Probability')
            axs[0].grid(True)
            axs[0].legend()
        
        # Oscillating environment
        if 'oscillation' in results:
            df = pd.DataFrame(results['oscillation'])
            axs[1].plot(df['step'], df['std_log_prob'], 'b-', marker='o', label='Standard CALNF')
            axs[1].plot(df['step'], df['anti_log_prob'], 'r-', marker='s', label='Antifragile CALNF')
            axs[1].set_title('Performance Under Oscillating Environment')
            axs[1].set_xlabel('Step')
            axs[1].set_ylabel('Log Probability')
            axs[1].grid(True)
            axs[1].legend()
        
        plt.tight_layout()
        return fig
    
    def create_overall_improvements_plot(self, improvements: Dict[str, float], 
                                        figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create summary visualization of relative improvements."""
        fig, ax = plt.subplots(figsize=figsize)
        
        categories = list(improvements.keys())
        values = list(improvements.values())
        
        # Color code by improvement (green positive, red negative)
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title('Relative Improvement of Antifragile CALNF vs Standard CALNF')
        ax.set_xlabel('Test Category')
        ax.set_ylabel('Relative Improvement (%)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (5 if height > 0 else -5),
                   f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        return fig

# Factory functions for easy access
def plot_flow_samples(flow, context_values: List[float], ax=None, 
                     n_samples: int = 1000, title: Optional[str] = None,
                     device: str = "cpu") -> plt.Axes:
    """Factory function to plot flow samples."""
    visualizer = FlowVisualizer()
    return visualizer.plot_flow_samples(flow, context_values, ax, n_samples, title, device)

def visualize_stress_test_results(all_results: Dict):
    """Factory function to visualize stress test results."""
    visualizer = StressTestVisualizer()
    
    if 'distribution_shift' in all_results:
        fig = visualizer.plot_distribution_shift_results(all_results['distribution_shift'])
        fig.savefig('distribution_shift_results.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    if 'black_swan' in all_results:
        fig = visualizer.plot_black_swan_results(all_results['black_swan'])
        fig.savefig('black_swan_results.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    if 'dynamic_environment' in all_results:
        fig = visualizer.plot_dynamic_environment_results(all_results['dynamic_environment'])
        fig.savefig('dynamic_environment_results.png', dpi=300, bbox_inches='tight')
        plt.close(fig)