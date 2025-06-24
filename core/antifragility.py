# antifragile_vbnf/core/antifragility.py
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import math

class AntifragilityCalculator:
    """Core class for calculating antifragility-related metrics and functions."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def calculate_derivatives(self, state_history: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate first and second derivatives from state history."""
        if len(state_history) < 2:
            return None, None
        
        # First derivative
        first_deriv = state_history[-1] - state_history[-2]
        
        # Second derivative
        if len(state_history) >= 3:
            prev_deriv = state_history[-2] - state_history[-3]
            second_deriv = first_deriv - prev_deriv
        else:
            second_deriv = torch.zeros_like(first_deriv)
        
        return first_deriv, second_deriv
    
    def calculate_derivatives_smoothed(self, state_history: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate smoothed derivatives from state history."""
        # Convert to tensor if not already
        if not isinstance(state_history[0], torch.Tensor):
            states = [torch.tensor(s, device=self.device) for s in state_history]
        else:
            states = state_history
        
        # Exponential smoothing
        alpha = 0.7
        smoothed_states = [states[0]]
        for i in range(1, len(states)):
            smoothed = alpha * states[i] + (1-alpha) * smoothed_states[-1]
            smoothed_states.append(smoothed)
        
        # First derivatives
        first_derivs = []
        for i in range(1, len(smoothed_states)):
            first_derivs.append(smoothed_states[i] - smoothed_states[i-1])
        
        # Second derivatives
        second_derivs = []
        for i in range(1, len(first_derivs)):
            second_derivs.append(first_derivs[i] - first_derivs[i-1])
        
        # Return average derivatives
        first_deriv = torch.stack(first_derivs).mean(0) if first_derivs else torch.zeros_like(states[0])
        second_deriv = torch.stack(second_derivs).mean(0) if second_derivs else torch.zeros_like(states[0])
        
        return first_deriv, second_deriv
    
    def redundancy_factor(self, x: torch.Tensor, omega: torch.Tensor, 
                         k: torch.Tensor, target_mean: torch.Tensor) -> torch.Tensor:
        """Calculate redundancy factor based on distance from critical regions."""
        # Distance from reference point (center of nominal class)
        distance_nominal = torch.norm(x - omega, dim=1, keepdim=True)
        
        # Distance from target mean (center of target class)
        distance_target = torch.norm(x - target_mean, dim=1, keepdim=True)
        
        # We want redundancy to be high when we're between nominal and target
        # and low when we're far from both
        between_factor = torch.exp(-(distance_nominal * distance_target))
        
        # Apply a bell-shaped curve centered at the midpoint between nominal and target
        midpoint = (omega + target_mean) / 2
        distance_mid = torch.norm(x - midpoint, dim=1, keepdim=True)
        radial_factor = torch.exp(-2.0 * distance_mid)
        
        return between_factor * radial_factor
    
    def enhanced_redundancy_factor(self, x: torch.Tensor, omega: torch.Tensor, 
                                  k: torch.Tensor, target_mean: torch.Tensor) -> torch.Tensor:
        """Enhanced redundancy factor with adaptive scaling and tail awareness."""
        # Calculate characteristic scale of the problem
        problem_scale = torch.norm(target_mean - omega)
        
        # Calculate interpolation midpoint - slightly biased toward target for asymmetry
        midpoint = 0.4 * omega + 0.6 * target_mean
        
        # Normalized distances (for scale invariance)
        distance_nominal = torch.norm(x - omega.unsqueeze(0), dim=1, keepdim=True) / problem_scale
        distance_target = torch.norm(x - target_mean.unsqueeze(0), dim=1, keepdim=True) / problem_scale
        
        # Normalized distance from midpoint
        distance_mid = torch.norm(x - midpoint.unsqueeze(0), dim=1, keepdim=True) / problem_scale
        
        # Enhanced between factor - using sum instead of product for numerical stability
        # This creates high values when points are close to the interpolation line
        between_factor = torch.exp(-0.5 * (distance_nominal + distance_target - 
                                         torch.norm(target_mean - omega) / problem_scale))
        
        # Radial factor with adaptive scaling
        sigma = 0.3  # Controls width of focus region
        radial_factor = torch.exp(-distance_mid**2 / (2 * sigma**2))
        
        # Tail region factor - enhances importance of points beyond target in direction of shift
        # This creates redundancy in anticipation of potential distribution shifts
        direction_vector = (target_mean - omega) / problem_scale
        projection = torch.sum((x - omega.unsqueeze(0)) * direction_vector.unsqueeze(0), dim=1, keepdim=True)
        tail_factor = 1.0 + torch.sigmoid(3.0 * (projection - 1.5 * problem_scale))
        
        # Special handling for stress threshold region (k)
        k_distance = torch.norm(x - k.unsqueeze(0), dim=1, keepdim=True) / problem_scale
        stress_factor = 1.0 + 0.5 * torch.exp(-k_distance**2 / 0.1)
        
        # Combined redundancy factor
        redundancy = between_factor * radial_factor * stress_factor * tail_factor
        
        # Clip to reasonable range for stability
        redundancy = torch.clamp(redundancy, 0.0, 5.0)
        
        return redundancy
    
    def detect_oscillations(self, state_history: List[torch.Tensor]) -> float:
        """Detect oscillations in the state history."""
        if len(state_history) < 4:
            return 0.0
        
        # Check recent states
        recent = torch.stack(state_history[-4:])
        diffs = recent[1:] - recent[:-1]
        
        # Count sign changes
        sign_changes = 0
        for i in range(1, len(diffs)):
            sign_change = torch.sum(torch.sign(diffs[i]) != torch.sign(diffs[i-1])).float()
            sign_changes += sign_change
        
        # Normalize
        max_changes = (len(diffs) - 1) * diffs.shape[1]
        oscillation_score = sign_changes / max_changes if max_changes > 0 else 0.0
        
        return oscillation_score.item()
    
    def calculate_transfer_function(self, flow, context_values: List[float], 
                                   omega: torch.Tensor, k: torch.Tensor, 
                                   num_samples: int = 1000) -> Dict:
        """
        Calculate Taleb's transfer function H that maps nonlinearities to tail sensitivity.
        
        Args:
            flow: The normalizing flow model
            context_values: List of context values to evaluate
            omega: Reference point (center of nominal distribution)
            k: Stress level (threshold below which we measure tail risk)
            num_samples: Number of samples to generate per context
            
        Returns:
            Dictionary with transfer function values at different points
        """
        transfer_values = {}
        
        with torch.no_grad():
            for context in context_values:
                ctx = torch.tensor([[context]], device=self.device)
                dist = flow(ctx)
                
                # Generate samples
                samples = dist.sample((num_samples,))
                
                # Calculate probability below k (left-tail probability)
                # For multi-dimensional case, we need to check each dimension separately
                indicators = torch.zeros_like(samples, dtype=torch.float32)
                for dim in range(samples.shape[-1]):
                    indicators[:, :, dim] = (samples[:, :, dim] < k[dim]).float()
                
                # Average across samples
                tail_prob = indicators.mean(dim=0)
                
                # Calculate shortfall below k for each dimension
                shortfall = torch.zeros_like(tail_prob)
                for dim in range(samples.shape[-1]):
                    mask = indicators[:, :, dim].bool()
                    if mask.sum() > 0:  # Avoid division by zero
                        masked_samples = samples[:, :, dim][mask]
                        shortfall[0, dim] = masked_samples.mean()
                    else:
                        shortfall[0, dim] = k[dim]  # Default value if no samples below k
                
                # Calculate transfer function for each dimension
                h_values = {}
                for dim in range(samples.shape[-1]):
                    # Calculate first derivative of shortfall
                    # We'll approximate it by sampling nearby context values
                    delta = 0.05
                    ctx_plus = torch.tensor([[context + delta]], device=self.device)
                    dist_plus = flow(ctx_plus)
                    samples_plus = dist_plus.sample((num_samples,))
                    
                    # Calculate indicators for context+delta
                    indicators_plus = (samples_plus[:, :, dim] < k[dim]).float()
                    
                    # Calculate shortfall for context+delta
                    shortfall_plus = torch.tensor([k[dim]], device=self.device)  # Default
                    if indicators_plus.sum() > 0:
                        mask_plus = indicators_plus.bool()
                        masked_samples_plus = samples_plus[:, :, dim][mask_plus]
                        shortfall_plus = masked_samples_plus.mean()
                    
                    # First derivative approximation
                    derivative = (shortfall_plus - shortfall[0, dim]) / delta
                    
                    # Calculate transfer function H = d(log(shortfall))/d(log(context))
                    # Using a safe value for context to avoid division by zero
                    safe_context = max(context, 1e-10)
                    safe_shortfall = max(shortfall[0, dim].item(), 1e-10)
                    
                    h_value = (safe_context / safe_shortfall) * derivative
                    
                    # Store as scalar
                    h_values[f'dim_{dim}'] = h_value.item() if isinstance(h_value, torch.Tensor) else h_value
                
                transfer_values[context] = h_values
                
        return transfer_values

class AdaptiveWeightCalculator:
    """Calculator for adaptive weights in antifragile training."""
    
    def __init__(self):
        pass
    
    def calculate_performance_metrics(self, flow, nominal_dist, target_dist, 
                                    first_deriv: torch.Tensor, second_deriv: torch.Tensor, 
                                    oscillation_factor: float) -> Dict[str, float]:
        """Calculate current performance metrics for adaptive weighting."""
        # Recovery speed - magnitude of first derivative
        recovery_speed = torch.norm(first_deriv).item()
        
        # Volatility response - stability of second derivative
        volatility_response = (1.0 - oscillation_factor) * torch.norm(second_deriv).item()
        
        # Outlier robustness - log probability of tail samples
        tail_samples = self.get_tail_samples(nominal_dist, target_dist, 10)
        if tail_samples is not None:
            # Sample at midpoint context
            mid_context = torch.ones(len(tail_samples), 1) * 0.5
            mid_dist = flow(mid_context)
            outlier_robustness = torch.mean(mid_dist.log_prob(tail_samples)).exp().item()
        else:
            outlier_robustness = 1.0
        
        # Interpolation quality - smoothness of interpolation path
        # Using math.exp for scalar operations
        interpolation_quality = math.exp(-oscillation_factor * 3.0)
        
        return {
            'recovery_speed': recovery_speed,
            'volatility_response': volatility_response,
            'outlier_robustness': outlier_robustness,
            'interpolation_quality': interpolation_quality
        }
    
    def calculate_adaptive_weights(self, current_metrics: Dict[str, float], 
                                 target_thresholds: Dict[str, float]) -> Dict[str, float]:
        """Calculate adaptive weights based on relative performance."""
        weights = {}
        
        for metric, current in current_metrics.items():
            target = target_thresholds[metric]
            if target > 0:
                # For metrics where higher is better
                if metric in ['recovery_speed', 'volatility_response', 'outlier_robustness']:
                    ratio = current / target
                    weights[metric] = max(0.5, min(2.0, 1.0 / ratio)) if ratio > 0 else 2.0
                # For metrics where lower is better (just interpolation_quality here)
                else:
                    ratio = target / current if current > 0 else 0.5
                    weights[metric] = max(0.5, min(2.0, ratio))
            else:
                weights[metric] = 1.0
        
        # Normalize weights to sum to the number of metrics
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            num_metrics = len(weights)
            weights = {k: v * num_metrics / weight_sum for k, v in weights.items()}
        
        return weights
    
    def get_tail_samples(self, nominal_dist, target_dist, num_samples: int = 20) -> Optional[torch.Tensor]:
        """Generate samples from the tail regions for outlier testing."""
        try:
            # Sample from nominal and target
            nominal_samples = nominal_dist.sample((num_samples * 2,))
            target_samples = target_dist.sample((num_samples * 2,))
            
            # Calculate log probs under both distributions
            nom_log_prob_nom = nominal_dist.log_prob(nominal_samples)
            nom_log_prob_targ = target_dist.log_prob(nominal_samples)
            targ_log_prob_nom = nominal_dist.log_prob(target_samples)
            targ_log_prob_targ = target_dist.log_prob(target_samples)
            
            # Find samples with large log prob ratio (in tails)
            nom_ratio = nom_log_prob_nom - nom_log_prob_targ
            targ_ratio = targ_log_prob_targ - targ_log_prob_nom
            
            # Get top samples with highest ratios (most extreme)
            _, nom_indices = torch.topk(nom_ratio, min(num_samples, len(nom_ratio)))
            _, targ_indices = torch.topk(targ_ratio, min(num_samples, len(targ_ratio)))
            
            # Combine tail samples
            tail_nominal = nominal_samples[nom_indices]
            tail_target = target_samples[targ_indices]
            
            return torch.cat([tail_nominal, tail_target], dim=0)
        except:
            # Handle any exceptions (e.g., numerical issues)
            return None