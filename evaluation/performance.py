# antifragile_vbnf/evaluation/performance.py
import torch
import numpy as np
from typing import Tuple, List

class PerformanceEvaluator:
    """Evaluator for model performance under various conditions."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def evaluate_performance_under_disruption(self, flow, nominal_data: torch.Tensor, 
                                             target_data: torch.Tensor, 
                                             disruption_levels: int = 10, 
                                             batch_size: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate flow performance under increasing disruption levels with datasets of different sizes."""
        disruption_magnitudes = np.linspace(0, 1, disruption_levels)
        log_probs = []
        
        # Use smaller dataset size for evaluation to ensure consistency
        eval_size = min(len(nominal_data), len(target_data), batch_size)
        
        with torch.no_grad():
            for magnitude in disruption_magnitudes:
                # Sample from both datasets to get equal sizes
                nominal_indices = torch.randperm(len(nominal_data))[:eval_size]
                nominal_sample = nominal_data[nominal_indices]
                
                # For target data, we might need to sample with replacement if it's smaller
                if len(target_data) >= eval_size:
                    target_indices = torch.randperm(len(target_data))[:eval_size]
                    target_sample = target_data[target_indices]
                else:
                    # Sample with replacement
                    target_indices = torch.randint(0, len(target_data), (eval_size,), device=self.device)
                    target_sample = target_data[target_indices]
                
                # Interpolate between sampled datasets
                mixed_data = nominal_sample * (1-magnitude) + target_sample * magnitude
                
                # Try different calibrations
                best_log_prob = -float('inf')
                for cal in torch.linspace(0, 1, 5, device=self.device):
                    context = torch.ones(len(mixed_data), 1, device=self.device) * cal
                    dist = flow(context)
                    log_prob = torch.mean(dist.log_prob(mixed_data)).item()
                    best_log_prob = max(best_log_prob, log_prob)
                
                log_probs.append(best_log_prob)
        
        return disruption_magnitudes, np.array(log_probs)
    
    def calculate_risk_measures(self, flow, context_values: List[float], 
                               alpha: float = 0.05, num_samples: int = 10000) -> dict:
        """
        Calculate VaR and CVaR (Expected Shortfall) for the distribution.
        
        Args:
            flow: The normalizing flow model
            context_values: List of context values to evaluate
            alpha: Confidence level for VaR/CVaR (e.g., 0.05 for 95% confidence)
            num_samples: Number of samples to generate
            
        Returns:
            Dictionary with VaR and CVaR values
        """
        risk_measures = {}
        
        with torch.no_grad():
            for context in context_values:
                ctx = torch.tensor([[context]], device=self.device)
                dist = flow(ctx)
                
                # Generate samples
                samples = dist.sample((num_samples,))
                
                # For each dimension
                dim_measures = {}
                for dim in range(samples.shape[-1]):
                    # Sort samples for this dimension
                    sorted_samples, _ = torch.sort(samples[:, 0, dim])
                    
                    # Calculate VaR
                    var_index = int(alpha * num_samples)
                    var = sorted_samples[var_index].item()
                    
                    # Calculate CVaR (Expected Shortfall)
                    cvar = sorted_samples[:var_index+1].mean().item()
                    
                    dim_measures[f'dim_{dim}'] = {
                        'VaR': var,
                        'CVaR': cvar
                    }
                    
                risk_measures[context] = dim_measures
        
        return risk_measures
    
    def compare_with_traditional_risk_measures(self, standard_flow, antifragile_flow) -> dict:
        """Compare antifragile vbnf with traditional risk measures."""
        context_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        # Calculate risk measures for both models
        standard_risk = self.calculate_risk_measures(standard_flow, context_values)
        antifragile_risk = self.calculate_risk_measures(antifragile_flow, context_values)
        
        # Compare the results
        comparison = {}
        for context in context_values:
            std_measures = standard_risk[context]
            anti_measures = antifragile_risk[context]
            
            context_comparison = {}
            for dim in std_measures.keys():
                # Calculate percentage differences
                var_diff = (anti_measures[dim]['VaR'] - std_measures[dim]['VaR']) / abs(std_measures[dim]['VaR'])
                cvar_diff = (anti_measures[dim]['CVaR'] - std_measures[dim]['CVaR']) / abs(std_measures[dim]['CVaR'])
                
                context_comparison[dim] = {
                    'VaR_std': std_measures[dim]['VaR'],
                    'VaR_anti': anti_measures[dim]['VaR'],
                    'VaR_diff_pct': var_diff * 100,
                    'CVaR_std': std_measures[dim]['CVaR'],
                    'CVaR_anti': anti_measures[dim]['CVaR'], 
                    'CVaR_diff_pct': cvar_diff * 100
                }
            
            comparison[context] = context_comparison
        
        return comparison
    
    def evaluate_interpolation_quality(self, flow, nominal_data: torch.Tensor, 
                                     target_data: torch.Tensor, num_contexts: int = 11) -> dict:
        """Evaluate the quality of interpolation between nominal and target distributions."""
        contexts = torch.linspace(0, 1, num_contexts, device=self.device)
        interpolation_metrics = {}
        
        with torch.no_grad():
            for i, ctx in enumerate(contexts):
                ctx_tensor = ctx.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1]
                dist = flow(ctx_tensor)
                
                # Generate samples
                samples = dist.sample((1000,))
                
                # Calculate metrics
                sample_mean = samples.mean(dim=0).squeeze()
                sample_std = samples.std(dim=0).squeeze()
                
                # Expected interpolated mean and std
                expected_mean = (1 - ctx) * nominal_data.mean(dim=0) + ctx * target_data.mean(dim=0)
                expected_std = (1 - ctx) * nominal_data.std(dim=0) + ctx * target_data.std(dim=0)
                
                # Distance from expected
                mean_distance = torch.norm(sample_mean - expected_mean).item()
                std_distance = torch.norm(sample_std - expected_std).item()
                
                interpolation_metrics[ctx.item()] = {
                    'mean_distance': mean_distance,
                    'std_distance': std_distance,
                    'sample_mean': sample_mean.cpu().numpy(),
                    'sample_std': sample_std.cpu().numpy(),
                    'expected_mean': expected_mean.cpu().numpy(),
                    'expected_std': expected_std.cpu().numpy()
                }
        
        return interpolation_metrics
    
    def evaluate_model_stability(self, flow, test_data: torch.Tensor, 
                                num_evaluations: int = 50) -> dict:
        """Evaluate model stability across multiple evaluations."""
        stability_metrics = {}
        
        all_log_probs = []
        all_sample_means = []
        all_sample_stds = []
        
        with torch.no_grad():
            for _ in range(num_evaluations):
                # Random context
                ctx = torch.rand(1, 1, device=self.device)
                dist = flow(ctx)
                
                # Evaluate on test data
                log_prob = dist.log_prob(test_data).mean().item()
                all_log_probs.append(log_prob)
                
                # Generate samples
                samples = dist.sample((100,))
                sample_mean = samples.mean(dim=0).squeeze().cpu().numpy()
                sample_std = samples.std(dim=0).squeeze().cpu().numpy()
                
                all_sample_means.append(sample_mean)
                all_sample_stds.append(sample_std)
        
        # Calculate stability metrics
        stability_metrics = {
            'log_prob_stability': {
                'mean': np.mean(all_log_probs),
                'std': np.std(all_log_probs),
                'cv': np.std(all_log_probs) / abs(np.mean(all_log_probs))  # Coefficient of variation
            },
            'sample_mean_stability': {
                'mean': np.mean(all_sample_means, axis=0),
                'std': np.std(all_sample_means, axis=0),
                'cv': np.std(all_sample_means, axis=0) / (np.abs(np.mean(all_sample_means, axis=0)) + 1e-8)
            },
            'sample_std_stability': {
                'mean': np.mean(all_sample_stds, axis=0),
                'std': np.std(all_sample_stds, axis=0),
                'cv': np.std(all_sample_stds, axis=0) / (np.abs(np.mean(all_sample_stds, axis=0)) + 1e-8)
            }
        }
        
        return stability_metrics
    
    def evaluate_context_sensitivity(self, flow, test_data: torch.Tensor, 
                                   context_range: Tuple[float, float] = (0.0, 1.0),
                                   num_contexts: int = 21) -> dict:
        """Evaluate how sensitive the model is to context changes."""
        contexts = torch.linspace(context_range[0], context_range[1], num_contexts, device=self.device)
        
        log_probs = []
        context_values = []
        
        with torch.no_grad():
            for ctx in contexts:
                ctx_tensor = torch.ones(len(test_data), 1, device=self.device) * ctx
                dist = flow(ctx_tensor)
                log_prob = dist.log_prob(test_data).mean().item()
                
                log_probs.append(log_prob)
                context_values.append(ctx.item())
        
        # Calculate sensitivity metrics
        log_probs = np.array(log_probs)
        context_values = np.array(context_values)
        
        # Gradient approximation
        gradients = np.gradient(log_probs, context_values)
        
        sensitivity_metrics = {
            'context_values': context_values,
            'log_probs': log_probs,
            'gradients': gradients,
            'max_gradient': np.max(np.abs(gradients)),
            'mean_gradient': np.mean(np.abs(gradients)),
            'gradient_variability': np.std(gradients),
            'smoothness': 1.0 / (np.std(gradients) + 1e-8)  # Higher is smoother
        }
        
        return sensitivity_metrics

# Factory functions for backward compatibility
def evaluate_performance_under_disruption(flow, nominal_data: torch.Tensor, target_data: torch.Tensor, 
                                         disruption_levels: int = 10, batch_size: int = 200,
                                         device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Factory function for evaluating performance under disruption."""
    evaluator = PerformanceEvaluator(device)
    return evaluator.evaluate_performance_under_disruption(
        flow, nominal_data, target_data, disruption_levels, batch_size
    )

def calculate_risk_measures(flow, context_values: List[float], alpha: float = 0.05, 
                          num_samples: int = 10000, device: str = "cpu") -> dict:
    """Factory function for calculating risk measures."""
    evaluator = PerformanceEvaluator(device)
    return evaluator.calculate_risk_measures(flow, context_values, alpha, num_samples)

def compare_with_traditional_risk_measures(standard_flow, antifragile_flow, device: str = "cpu") -> dict:
    """Factory function for comparing with traditional risk measures."""
    evaluator = PerformanceEvaluator(device)
    return evaluator.compare_with_traditional_risk_measures(standard_flow, antifragile_flow)