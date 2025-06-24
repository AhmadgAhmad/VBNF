# antifragile_vbnf/visualization/dashboards.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List

class AntifragileDashboard:
    """Dashboard for antifragile analysis visualization."""
    
    def __init__(self):
        self.setup_style()
    
    def setup_style(self):
        """Setup plotting style."""
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        sns.set_palette("husl")
    
    def create_antifragile_visualizations(self, all_run_metrics: Dict, 
                                         antifragile_analysis: Dict):
        """Create antifragile-focused visualizations."""
        
        # 1. Antifragile Hypothesis Test Results
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Core antifragile metrics
        core_metrics = antifragile_analysis.get('antifragile_hypothesis_tests', {})
        if core_metrics:
            self.plot_antifragile_core_results(axes[0, 0], core_metrics, all_run_metrics)
        
        # Trade-off analysis
        trade_offs = antifragile_analysis.get('trade_off_analysis', {})
        if trade_offs:
            self.plot_trade_off_analysis(axes[0, 1], trade_offs, all_run_metrics)
        
        # Stress response curves
        self.plot_stress_response_curves(axes[1, 0], all_run_metrics)
        
        # Recovery trajectories
        self.plot_recovery_trajectories(axes[1, 1], all_run_metrics)
        
        plt.tight_layout()
        plt.savefig('antifragile_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Jensen's Gap Visualization
        self.create_jensen_gap_visualization(all_run_metrics)
        
        # 3. Conditional Performance Heatmap
        self.create_conditional_performance_heatmap(antifragile_analysis)
        
        # 4. Summary Dashboard
        self.create_antifragile_summary_dashboard(antifragile_analysis)
    
    def plot_antifragile_core_results(self, ax, core_metrics: Dict, 
                                     all_run_metrics: Dict):
        """Plot core antifragile metrics results."""
        metrics = list(core_metrics.keys())
        improvements = [core_metrics[m]['improvement_pct'] for m in metrics]
        p_values = [core_metrics[m]['p_value'] for m in metrics]
        
        # Color by significance and direction
        colors = []
        for i, metric in enumerate(metrics):
            if core_metrics[metric]['is_significant']:
                if improvements[i] > 0:
                    colors.append('green')
                else:
                    colors.append('red')
            else:
                colors.append('gray')
        
        bars = ax.bar(range(len(metrics)), improvements, color=colors, alpha=0.7)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Core Antifragile Metrics Performance')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add significance stars
        for i, (improvement, p_val) in enumerate(zip(improvements, p_values)):
            if p_val < 0.05:
                ax.text(i, improvement + (5 if improvement > 0 else -5), 
                       '*', ha='center', fontsize=16, fontweight='bold')
    
    def plot_trade_off_analysis(self, ax, trade_offs: Dict, all_run_metrics: Dict):
        """Plot trade-off analysis showing acceptable deteriorations."""
        if not trade_offs:
            ax.text(0.5, 0.5, 'No trade-off metrics available', 
                    ha='center', va='center', transform=ax.transAxes)
            return
        
        metrics = list(trade_offs.keys())
        improvements = [trade_offs[m]['improvement_pct'] for m in metrics]
        
        # Color by acceptability of trade-off
        colors = []
        for i, metric in enumerate(metrics):
            if trade_offs[metric]['is_significant']:
                if improvements[i] < -50:  # Large deterioration
                    colors.append('darkred')
                elif improvements[i] < 0:  # Moderate deterioration
                    colors.append('orange')
                else:
                    colors.append('lightgreen')
            else:
                colors.append('lightgray')
        
        ax.bar(range(len(metrics)), improvements, color=colors, alpha=0.7)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylabel('Change (%)')
        ax.set_title('Trade-off Analysis\n(Orange = Acceptable, Dark Red = Concerning)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    def plot_stress_response_curves(self, ax, all_run_metrics: Dict):
        """Plot stress response curves showing antifragile behavior."""
        # Mock stress levels for visualization
        stress_levels = np.linspace(0, 1, 5)
        
        # Use available metrics to simulate stress response
        stress_metrics = ['noise_variation', 'stress_level_variation', 'extreme_context']
        
        for model_type in ['std', 'anti']:
            responses = []
            for metric in stress_metrics[:3]:  # Use first 3 as stress levels
                if metric in all_run_metrics and model_type in all_run_metrics[metric]:
                    responses.append(np.mean(all_run_metrics[metric][model_type]))
            
            if len(responses) >= 3:
                ax.plot(stress_levels[:len(responses)], responses, 
                       marker='o', label=f'{"A-vbnf" if model_type == "anti" else "Standard vbnf"}',
                       linewidth=2)
        
        ax.set_xlabel('Stress Level')
        ax.set_ylabel('Performance')
        ax.set_title('Stress Response Curves\n(Antifragile should show convex response)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_recovery_trajectories(self, ax, all_run_metrics: Dict):
        """Plot recovery trajectories from disruptions."""
        if 'oscillation' in all_run_metrics:
            for model_type in ['std', 'anti']:
                if model_type in all_run_metrics['oscillation']:
                    trajectory = all_run_metrics['oscillation'][model_type]
                    if len(trajectory) > 0:
                        # Simulate time series
                        time_points = np.linspace(0, 1, len(trajectory))
                        ax.plot(time_points, trajectory, 
                               marker='o', label=f'{"A-vbnf" if model_type == "anti" else "Standard vbnf"}',
                               linewidth=2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Performance')
        ax.set_title('Recovery Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def create_jensen_gap_visualization(self, all_run_metrics: Dict):
        """Create specific Jensen's gap visualization."""
        if 'jensen_gap' not in all_run_metrics:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Box plot of Jensen's gap
        data_to_plot = []
        labels = []
        
        for model_type in ['std', 'anti']:
            if model_type in all_run_metrics['jensen_gap']:
                data_to_plot.append(all_run_metrics['jensen_gap'][model_type])
                labels.append('Standard vbnf' if model_type == 'std' else 'A-vbnf')
        
        if data_to_plot:
            box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, 
                      label='Fragile/Antifragile Threshold')
            ax.set_ylabel("Jensen's Gap")
            ax.set_title("Jensen's Gap Comparison\n(Positive = Antifragile, Negative = Fragile)")
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('jensen_gap_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_conditional_performance_heatmap(self, antifragile_analysis: Dict):
        """Create heatmap showing conditional performance across different scenarios."""
        # Collect all metrics and their improvements
        all_metrics = {}
        all_metrics.update(antifragile_analysis.get('antifragile_hypothesis_tests', {}))
        all_metrics.update(antifragile_analysis.get('trade_off_analysis', {}))
        all_metrics.update(antifragile_analysis.get('conditional_performance', {}))
        
        if not all_metrics:
            return
        
        # Create matrix for heatmap
        metrics = list(all_metrics.keys())
        improvements = [all_metrics[m]['improvement_pct'] for m in metrics]
        significance = [all_metrics[m]['is_significant'] for m in metrics]
        
        # Create heatmap data
        heatmap_data = np.array(improvements).reshape(1, -1)
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 3))
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-100, vmax=100)
        
        # Set labels
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticks([])
        
        # Add text annotations
        for i, (improvement, is_sig) in enumerate(zip(improvements, significance)):
            text = f'{improvement:.1f}%'
            if is_sig:
                text += '*'
            ax.text(i, 0, text, ha='center', va='center', 
                   color='white' if abs(improvement) > 50 else 'black',
                   fontweight='bold' if is_sig else 'normal')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Improvement (%)')
        
        ax.set_title('Conditional Performance Heatmap\n(* = Statistically Significant)')
        
        plt.tight_layout()
        plt.savefig('conditional_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_antifragile_summary_dashboard(self, antifragile_analysis: Dict):
        """Create summary dashboard with key antifragile insights."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Summary statistics
        core_metrics = antifragile_analysis.get('antifragile_hypothesis_tests', {})
        trade_metrics = antifragile_analysis.get('trade_off_analysis', {})
        
        # Count successes
        core_successes = sum(1 for m in core_metrics.values() 
                            if m['is_significant'] and m['improvement_pct'] > 0)
        core_total = len(core_metrics)
        
        trade_acceptable = sum(1 for m in trade_metrics.values() 
                              if not m['is_significant'] or m['improvement_pct'] > -20)
        trade_total = len(trade_metrics)
        
        # Success rate pie chart
        if core_total > 0:
            ax1.pie([core_successes, core_total - core_successes], 
                   labels=['Antifragile Success', 'No Benefit'], 
                   autopct='%1.1f%%', colors=['green', 'lightgray'])
            ax1.set_title(f'Core Antifragile Metrics\n({core_successes}/{core_total} successful)')
        
        # Trade-off acceptability
        if trade_total > 0:
            ax2.pie([trade_acceptable, trade_total - trade_acceptable],
                   labels=['Acceptable', 'Concerning'], 
                   autopct='%1.1f%%', colors=['orange', 'red'])
            ax2.set_title(f'Trade-off Analysis\n({trade_acceptable}/{trade_total} acceptable)')
        
        # Effect sizes
        all_metrics = {**core_metrics, **trade_metrics}
        if all_metrics:
            effect_sizes = [abs(m['cohens_d']) for m in all_metrics.values()]
            metric_names = list(all_metrics.keys())
            
            ax3.barh(range(len(metric_names)), effect_sizes)
            ax3.set_yticks(range(len(metric_names)))
            ax3.set_yticklabels(metric_names)
            ax3.set_xlabel("Effect Size (|Cohen's d|)")
            ax3.set_title('Effect Sizes Across Metrics')
            ax3.axvline(x=0.5, color='orange', linestyle='--', label='Medium Effect')
            ax3.axvline(x=0.8, color='red', linestyle='--', label='Large Effect')
            ax3.legend()
        
        # Key insights text
        insights = []
        if core_successes > 0:
            insights.append(f"✅ {core_successes} antifragile benefits confirmed")
        if core_total > 0 and core_successes / core_total > 0.5:
            insights.append("✅ Majority of antifragile hypotheses supported")
        if trade_total > 0 and trade_acceptable / trade_total > 0.7:
            insights.append("✅ Trade-offs mostly acceptable")
        
        ax4.text(0.1, 0.9, "Key Insights:", fontsize=14, fontweight='bold', 
                 transform=ax4.transAxes)
        
        for i, insight in enumerate(insights):
            ax4.text(0.1, 0.8 - i*0.1, insight, fontsize=12, 
                    transform=ax4.transAxes)
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Summary Assessment')
        
        plt.tight_layout()
        plt.savefig('antifragile_summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

class AdvancedVisualizationDashboard:
    """Advanced visualization dashboard for detailed antifragile analysis."""
    
    def __init__(self):
        self.setup_style()
    
    def setup_style(self):
        """Setup advanced plotting style."""
        plt.rcParams['figure.figsize'] = (14, 10)
        plt.rcParams['font.size'] = 11
        sns.set_style("whitegrid")
    
    def visualize_advanced_antifragility_results(self, results: Dict):
        """Visualize the results of the advanced antifragility tests."""
        
        # 1. Volatility Response (Jensen's Gap)
        if 'volatility_response' in results:
            self.plot_volatility_response(results['volatility_response'])
        
        # 2. Convexity/Concavity Analysis
        if 'convexity' in results:
            self.plot_convexity_analysis(results['convexity'])
        
        # 3. Barbell Strategy
        if 'barbell_strategy' in results:
            self.plot_barbell_strategy(results['barbell_strategy'])
        
        # 4. Recovery Speed
        if 'recovery_speed' in results:
            self.plot_recovery_speed(results['recovery_speed'])
    
    def plot_volatility_response(self, volatility_data: List[Dict]):
        """Plot volatility response analysis."""
        df = pd.DataFrame(volatility_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['volatility'], df['std_jensen_gap'], 'b-', marker='o', label='Standard vbnf')
        ax.plot(df['volatility'], df['anti_jensen_gap'], 'r-', marker='s', label='Antifragile vbnf')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Shade regions where each model benefits from volatility
        ax.fill_between(df['volatility'], 0, df['std_jensen_gap'], 
                       where=(df['std_jensen_gap'] > 0), color='blue', alpha=0.2)
        ax.fill_between(df['volatility'], 0, df['anti_jensen_gap'], 
                       where=(df['anti_jensen_gap'] > 0), color='red', alpha=0.2)
        
        ax.set_title("Volatility Response (Jensen's Gap)")
        ax.set_xlabel("Volatility Level")
        ax.set_ylabel("Jensen's Gap (E[f(X)] - f(E[X]))")
        ax.legend()
        ax.grid(True)
        
        # Add annotations explaining the interpretation
        ax.text(0.02, 0.98, "Positive values indicate antifragility (benefit from volatility)",
               transform=ax.transAxes, va='top', fontsize=10)
        ax.text(0.02, 0.93, "Negative values indicate fragility (harm from volatility)",
               transform=ax.transAxes, va='top', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('volatility_response.png', dpi=300)
        plt.show()
    
    def plot_convexity_analysis(self, convexity_data: List[Dict]):
        """Plot convexity/concavity analysis."""
        df = pd.DataFrame(convexity_data)
        
        # Create a 2D heatmap of convexity differences
        contexts = np.sort(np.unique(np.concatenate([df['ctx_i'], df['ctx_j']])))
        n_contexts = len(contexts)
        
        diff_matrix = np.zeros((n_contexts, n_contexts))
        std_matrix = np.zeros((n_contexts, n_contexts))
        anti_matrix = np.zeros((n_contexts, n_contexts))
        
        # Fill matrices
        for _, row in df.iterrows():
            i_idx = np.where(contexts == row['ctx_i'])[0][0]
            j_idx = np.where(contexts == row['ctx_j'])[0][0]
            
            diff_matrix[i_idx, j_idx] = row['difference']
            diff_matrix[j_idx, i_idx] = row['difference']  # Symmetric
            
            std_matrix[i_idx, j_idx] = row['std_convexity']
            std_matrix[j_idx, i_idx] = row['std_convexity']
            
            anti_matrix[i_idx, j_idx] = row['anti_convexity']
            anti_matrix[j_idx, i_idx] = row['anti_convexity']
        
        # Plot convexity maps
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Standard vbnf convexity
        im0 = axs[0].imshow(std_matrix, cmap='RdBu', origin='lower')
        axs[0].set_title('Standard vbnf Convexity')
        axs[0].set_xlabel('Context Value')
        axs[0].set_ylabel('Context Value')
        axs[0].set_xticks(np.arange(n_contexts))
        axs[0].set_yticks(np.arange(n_contexts))
        axs[0].set_xticklabels([f'{c:.1f}' for c in contexts])
        axs[0].set_yticklabels([f'{c:.1f}' for c in contexts])
        fig.colorbar(im0, ax=axs[0])
        
        # Antifragile vbnf convexity
        im1 = axs[1].imshow(anti_matrix, cmap='RdBu', origin='lower')
        axs[1].set_title('Antifragile vbnf Convexity')
        axs[1].set_xlabel('Context Value')
        axs[1].set_xticklabels([f'{c:.1f}' for c in contexts])
        axs[1].set_yticks(np.arange(n_contexts))
        axs[1].set_yticklabels([f'{c:.1f}' for c in contexts])
        fig.colorbar(im1, ax=axs[1])
        
        # Difference
        im2 = axs[2].imshow(diff_matrix, cmap='RdBu', origin='lower')
        axs[2].set_title('Convexity Difference (Anti - Std)')
        axs[2].set_xlabel('Context Value')
        axs[2].set_xticklabels([f'{c:.1f}' for c in contexts])
        axs[2].set_yticks(np.arange(n_contexts))
        axs[2].set_yticklabels([f'{c:.1f}' for c in contexts])
        fig.colorbar(im2, ax=axs[2])
        
        plt.tight_layout()
        plt.savefig('convexity_map.png', dpi=300)
        plt.show()
    
    def plot_barbell_strategy(self, barbell_data: List[Dict]):
        """Plot barbell strategy analysis."""
        df = pd.DataFrame(barbell_data)
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        
        # Variance plot
        ax1 = axs[0]
        ax1.plot(df['context'], df['std_variance'], 'b-', marker='o', label='Standard vbnf')
        ax1.plot(df['context'], df['anti_variance'], 'r-', marker='s', label='Antifragile vbnf')
        ax1.set_title('Variance Across Context Values')
        ax1.set_xlabel('Context Value')
        ax1.set_ylabel('Variance')
        ax1.grid(True)
        ax1.legend()
        
        # Variance ratio plot
        ax2 = axs[1]
        ax2.plot(df['context'], df['ratio'], 'g-', marker='o')
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Variance Ratio (Anti/Std)')
        ax2.set_xlabel('Context Value')
        ax2.set_ylabel('Ratio')
        ax2.grid(True)
        
        # Add annotations for barbell interpretation
        ax2.text(0.02, 0.95, "Ratio > 1: Antifragile model has higher variance (exploration)",
               transform=ax2.transAxes, va='top', fontsize=10)
        ax2.text(0.02, 0.90, "Ratio < 1: Antifragile model has lower variance (exploitation)",
               transform=ax2.transAxes, va='top', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('barbell_strategy.png', dpi=300)
        plt.show()
    
    def plot_recovery_speed(self, recovery_data: List[Dict]):
        """Plot recovery speed analysis."""
        df = pd.DataFrame(recovery_data)
        
        # Create pivot table for heatmap
        pivot_diff = df.pivot_table(
            values='difference', 
            index='magnitude', 
            columns='steps'
        )
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        
        # Line plot by perturbation magnitude
        for magnitude in df['magnitude'].unique():
            subset = df[df['magnitude'] == magnitude]
            axs[0].plot(subset['steps'], subset['std_recovery_pct'], 'b--', marker='o', alpha=0.5)
            axs[0].plot(subset['steps'], subset['anti_recovery_pct'], 'r--', marker='s', alpha=0.5)
        
        # Add average lines
        avg_std = df.groupby('steps')['std_recovery_pct'].mean()
        avg_anti = df.groupby('steps')['anti_recovery_pct'].mean()
        axs[0].plot(avg_std.index, avg_std.values, 'b-', marker='o', linewidth=3, label='Standard vbnf Avg')
        axs[0].plot(avg_anti.index, avg_anti.values, 'r-', marker='s', linewidth=3, label='Antifragile vbnf Avg')
        
        axs[0].set_title('Recovery Progress After Perturbation')
        axs[0].set_xlabel('Fine-tuning Steps')
        axs[0].set_ylabel('Recovery Percentage')
        axs[0].grid(True)
        axs[0].legend()
        
        # Heatmap of differences
        im = axs[1].imshow(pivot_diff, cmap='RdBu_r', aspect='auto')
        axs[1].set_title('Recovery Speed Difference (Anti - Std)')
        axs[1].set_xlabel('Fine-tuning Steps')
        axs[1].set_ylabel('Perturbation Magnitude')
        
        # Set tick labels
        axs[1].set_xticks(np.arange(len(pivot_diff.columns)))
        axs[1].set_yticks(np.arange(len(pivot_diff.index)))
        axs[1].set_xticklabels(pivot_diff.columns)
        axs[1].set_yticklabels(pivot_diff.index)
        
        # Add colorbar
        fig.colorbar(im, ax=axs[1])
        
        plt.tight_layout()
        plt.savefig('recovery_speed.png', dpi=300)
        plt.show()

# Factory functions for easy access
def create_antifragile_visualizations(all_run_metrics: Dict, antifragile_analysis: Dict):
    """Factory function to create antifragile visualizations."""
    dashboard = AntifragileDashboard()
    dashboard.create_antifragile_visualizations(all_run_metrics, antifragile_analysis)

def visualize_advanced_antifragility_results(results: Dict):
    """Factory function to visualize advanced antifragility results."""
    dashboard = AdvancedVisualizationDashboard()
    dashboard.visualize_advanced_antifragility_results(results)