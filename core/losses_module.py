# antifragile_calnf/core/losses.py
import torch
import torch.nn as nn
from typing import List, Tuple, Dict

class AntifragileLossCalculator:
    """Calculator for antifragile-specific loss functions."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def antifragile_loss_function(self, flow, nominal_data: torch.Tensor, target_data: torch.Tensor, 
                                 target_subsets: List[torch.Tensor], context_dim: int, 
                                 step: int) -> Tuple[torch.Tensor, Dict]:
        """Antifragile-centered loss function."""
        
        # 1. Minimal base reconstruction loss (just enough to maintain basic functionality)
        nominal_context = torch.zeros(len(nominal_data), context_dim, device=self.device)
        nominal_dist = flow(nominal_context)
        nominal_log_prob = nominal_dist.log_prob(nominal_data)
        base_loss = -torch.mean(nominal_log_prob) * 0.2  # Much reduced weight
        
        # Phase in antifragile components gradually
        antifragile_weight = min(1.0, step / 300.0)
        
        # 2. Volatility benefit term (core antifragility)
        volatility_gain = self.calculate_volatility_benefit(flow, nominal_data, target_data, context_dim)
        
        # 3. Progressive improvement term
        progressive_gain = self.calculate_progressive_improvement(flow, target_subsets, context_dim)
        
        # 4. Tail preparedness term
        tail_preparedness = self.calculate_tail_preparedness(flow, nominal_data, target_data, context_dim)
        
        # 5. Diversity exploitation term (replaces KL regularization)
        diversity_gain = self.calculate_diversity_exploitation(flow, target_subsets, context_dim)
        
        # 6. Stress response term (new - tests response to increasing stress)
        stress_response = self.calculate_stress_response(flow, target_data, context_dim)
        
        # Combined antifragile gain
        total_antifragile_gain = antifragile_weight * (
            volatility_gain * 1.0 +          # Core antifragility
            progressive_gain * 0.8 +         # Learning from experience
            tail_preparedness * 0.6 +        # Preparation for extremes
            diversity_gain * 0.7 +           # Exploiting diversity
            stress_response * 0.9            # Stress response
        )
        
        # Total loss: minimize base loss, maximize antifragile gains
        total_loss = base_loss - total_antifragile_gain
        
        return total_loss, {
            'base_loss': base_loss,
            'volatility_gain': volatility_gain,
            'progressive_gain': progressive_gain,
            'tail_preparedness': tail_preparedness,
            'diversity_gain': diversity_gain,
            'stress_response': stress_response,
            'total_antifragile_gain': total_antifragile_gain
        }
    
    def calculate_volatility_benefit(self, flow, nominal_data: torch.Tensor, 
                                   target_data: torch.Tensor, context_dim: int) -> torch.Tensor:
        """Core antifragility: benefit from volatility."""
        
        volatility_levels = [0.0, 0.5, 1.0, 1.5]
        performance_curve = []
        
        for vol_level in volatility_levels:
            # Add controlled volatility
            if vol_level > 0:
                noise_scale = torch.std(target_data) * vol_level * 0.1
                noisy_target = target_data + torch.randn_like(target_data) * noise_scale
            else:
                noisy_target = target_data
            
            # Measure performance at this volatility level
            context = torch.ones(len(noisy_target), context_dim, device=self.device)
            dist = flow(context)
            log_prob = dist.log_prob(noisy_target)
            performance = torch.mean(log_prob)
            performance_curve.append(performance)
        
        # Calculate benefit from volatility (positive slope = antifragile)
        performance_tensor = torch.stack(performance_curve)
        
        # Measure both linear trend and convex response
        linear_trend = performance_tensor[-1] - performance_tensor[0]
        
        # Measure convexity (antifragile systems should show convex response)
        if len(performance_tensor) >= 3:
            second_derivative = performance_tensor[:-2] - 2*performance_tensor[1:-1] + performance_tensor[2:]
            convexity_bonus = torch.mean(second_derivative)
        else:
            convexity_bonus = torch.tensor(0.0, device=self.device)
        
        volatility_benefit = linear_trend + convexity_bonus * 0.5
        
        return volatility_benefit
    
    def calculate_progressive_improvement(self, flow, target_subsets: List[torch.Tensor], 
                                        context_dim: int) -> torch.Tensor:
        """Reward improvement with increasing diversity of experience."""
        
        if len(target_subsets) < 2:
            return torch.tensor(0.0, device=self.device)
        
        cumulative_performances = []
        
        # Simulate progressive learning: performance should improve as we see more subsets
        for i in range(len(target_subsets)):
            # Context representing cumulative experience
            experience_level = (i + 1) / len(target_subsets)
            context = torch.ones(len(target_subsets[i]), context_dim, device=self.device) * experience_level
            
            dist = flow(context)
            performance = torch.mean(dist.log_prob(target_subsets[i]))
            cumulative_performances.append(performance)
        
        # Reward progressive improvement
        performances = torch.stack(cumulative_performances)
        
        # Calculate improvement trend
        if len(performances) > 1:
            improvements = performances[1:] - performances[:-1]
            progressive_gain = torch.mean(improvements)
            
            # Bonus for consistent improvement (not just final improvement)
            consistency_bonus = torch.exp(-torch.std(improvements))
            progressive_gain = progressive_gain * consistency_bonus
        else:
            progressive_gain = torch.tensor(0.0, device=self.device)
        
        return progressive_gain
    
    def calculate_tail_preparedness(self, flow, nominal_data: torch.Tensor, 
                                   target_data: torch.Tensor, context_dim: int) -> torch.Tensor:
        """Prepare for events more extreme than observed."""
        
        # Generate synthetic extreme events
        data_std = torch.std(torch.cat([nominal_data, target_data]))
        data_mean = torch.mean(torch.cat([nominal_data, target_data]), dim=0)
        
        # Create samples at 2x and 3x standard deviations from mean
        extreme_factors = [2.0, 3.0]
        tail_performances = []
        
        for factor in extreme_factors:
            # Generate extreme samples
            extreme_samples = data_mean + torch.randn_like(target_data[:20]) * data_std * factor
            
            # Test model's ability to handle these
            extreme_context = torch.ones(len(extreme_samples), context_dim, device=self.device)
            extreme_dist = flow(extreme_context)
            
            # Measure both likelihood and generative capability
            likelihood = torch.mean(extreme_dist.log_prob(extreme_samples))
            
            # Test if model can generate diverse samples (not mode collapse)
            generated_samples = extreme_dist.sample((50,))
            generation_diversity = torch.std(generated_samples)
            
            # Combined tail performance
            tail_perf = likelihood + torch.log(generation_diversity + 1e-6)
            tail_performances.append(tail_perf)
        
        # Average performance on tail events
        tail_preparedness = torch.mean(torch.stack(tail_performances))
        
        return tail_preparedness
    
    def calculate_diversity_exploitation(self, flow, target_subsets: List[torch.Tensor], 
                                        context_dim: int) -> torch.Tensor:
        """Exploit diversity instead of penalizing it (replaces KL regularization)."""
        
        if len(target_subsets) < 2:
            return torch.tensor(0.0, device=self.device)
        
        subset_capabilities = []
        
        for i, subset in enumerate(target_subsets):
            # Each subset gets its own context
            context = torch.zeros(len(subset), context_dim, device=self.device)
            context[:, 0] = (i + 1) / len(target_subsets)
            
            dist = flow(context)
            
            # Measure three aspects of subset handling:
            # 1. Likelihood of the subset data
            likelihood = torch.mean(dist.log_prob(subset))
            
            # 2. Diversity of generated samples
            samples = dist.sample((50,))
            diversity = torch.std(samples)
            
            # 3. Distinctiveness from other subsets
            subset_mean = torch.mean(samples, dim=0)
            
            subset_capability = likelihood + torch.log(diversity + 1e-6) * 0.5
            subset_capabilities.append((subset_capability, subset_mean))
        
        # Average capability across subsets
        capabilities = torch.stack([cap for cap, _ in subset_capabilities])
        avg_capability = torch.mean(capabilities)
        
        # Bonus for maintaining distinctiveness between subsets
        subset_means = torch.stack([mean for _, mean in subset_capabilities])
        distinctiveness = torch.std(subset_means)
        
        diversity_gain = avg_capability + torch.log(distinctiveness + 1e-6) * 0.3
        
        return diversity_gain
    
    def calculate_stress_response(self, flow, target_data: torch.Tensor, context_dim: int) -> torch.Tensor:
        """Test response to increasing levels of stress."""
        
        stress_levels = [0.5, 1.0, 1.5, 2.0, 2.5]
        stress_responses = []
        
        baseline_context = torch.ones(len(target_data), context_dim, device=self.device) * 0.5
        baseline_dist = flow(baseline_context)
        baseline_performance = torch.mean(baseline_dist.log_prob(target_data))
        
        for stress_level in stress_levels:
            # Apply stress by shifting the context and adding noise
            stressed_context = torch.ones(len(target_data), context_dim, device=self.device) * stress_level / 5.0
            stressed_data = target_data + torch.randn_like(target_data) * stress_level * 0.05
            
            stressed_dist = flow(stressed_context)
            stressed_performance = torch.mean(stressed_dist.log_prob(stressed_data))
            
            # Measure relative performance under stress
            relative_performance = stressed_performance - baseline_performance
            stress_responses.append(relative_performance)
        
        # Antifragile response: performance should improve or at least not degrade severely
        responses = torch.stack(stress_responses)
        
        # Reward systems that maintain or improve performance under stress
        stress_resilience = torch.mean(responses)
        
        # Bonus for showing improvement at higher stress levels
        if len(responses) >= 2:
            improvement_under_stress = responses[-1] - responses[0]  # Compare highest vs lowest stress
            stress_resilience = stress_resilience + improvement_under_stress * 0.5
        
        return stress_resilience