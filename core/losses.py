# antifragile_vbnf/core/losses.py
import torch
import torch.nn as nn
from typing import List, Tuple, Dict

class AntifragileLossCalculator_OLD:
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
        base_loss = -torch.mean(nominal_log_prob) * 0.8  # Much reduced weight
        
        # Phase in antifragile components gradually
        antifragile_weight = min(1.0, step / 300.0)
        # antifragile_weight = min(2.0, step / 100.0)  # Slower phase-in
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
        # total_antifragile_gain = antifragile_weight * (
        #     volatility_gain * 0.5 +      # Reduce from 1.0
        #     progressive_gain * 0.3 +     # Reduce from 0.8  
        #     tail_preparedness * 0.2 +    # Reduce from 0.6
        #     stress_response * 0.4        # Reduce from 0.9
        # )
        #         # Combined antifragile gain
        # total_antifragile_gain = antifragile_weight * (
        #     volatility_gain * 2.0 +          # Core antifragility
        #     progressive_gain * 0.8 +         # Learning from experience
        #     tail_preparedness * 0.6 +        # Preparation for extremes
        #     diversity_gain * 0.7 +           # Exploiting diversity
        #     stress_response * 1.5            # Stress response
        # )
        
        print(f"Volatility: {volatility_gain:.4f}")
        print(f"Stress: {stress_response:.4f}")  
        print(f"Progressive: {progressive_gain:.4f}")
        print(f"Tail: {tail_preparedness:.4f}")
        print(f"Diversity: {diversity_gain:.4f}")
        # Total loss: minimize base loss, maximize antifragile gains
        total_loss = base_loss + total_antifragile_gain 
        
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
        # volatility_levels = [0.0, 0.1, 0.3] # Reduced volatility levels for smoother response
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
    

class AntifragileLossCalculator:
    """Loss calculator implementing Taleb's mathematical definition of antifragility."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def antifragile_loss_function(self, flow, nominal_data: torch.Tensor, target_data: torch.Tensor, 
                              target_subsets: List[torch.Tensor], context_dim: int, 
                              step: int) -> Tuple[torch.Tensor, Dict]:
        """
        True antifragile loss based on Taleb's mathematical framework:
        1. Left-tail robustness (b-robust)
        2. Right-tail benefit (fat right tail)
        3. Convex response to meaningful volatility
        4. Transfer function implementation
        """
        
        # DOMINANT: Structural preservation (maintain banana shape)
        structural_loss = self.calculate_structural_preservation(flow, nominal_data, target_data, context_dim)
        
        # CRITICAL: Left-tail robustness (Taleb's b-robustness condition)
        left_tail_robustness = self.calculate_left_tail_robustness_taleb(flow, target_data, context_dim)
        
        # Conservative phase-in of antifragile components
        antifragile_weight = min(0.15, step / 1000.0)  # Much slower, lower max
        
        # Taleb's antifragile components
        volatility_benefit = self.calculate_volatility_benefit_corrected(flow, target_data, context_dim)
        convex_response = self.calculate_convex_response_taleb(flow, target_data, context_dim)
        fragility_heuristic = self.calculate_taleb_h_ratio(flow, target_data, context_dim)
        
        # Taleb's condition: Antifragile = Robust left + Volatile benefit + Convex
        antifragile_gain = antifragile_weight * (
            volatility_benefit * 1.0 +          # CORE: Benefit from volatility (Taleb's main definition)
            convex_response * 0.6 +             # Convex response supports volatility benefit
            fragility_heuristic * 0.4           # Reduce fragility (H-ratio < 1)
        )
        
        # Taleb's hierarchy: Structure >> Robustness >> Antifragility
        # total_loss = (
        #     structural_loss * 2.0 +             # DOMINANT: preserve core function
        #     left_tail_robustness * 1.8 +        # CRITICAL: robust to adverse events
        #     antifragile_gain * 1-                   # BONUS: antifragile properties
        #     volatility_benefit* 1.0
        # )
        total_loss = (
                    structural_loss * 3.0 +              # Increase structural dominance
                    left_tail_robustness * 1.8 +         
                    antifragile_gain * 0.8 -             # Reduce antifragile aggression
                    volatility_benefit * 0.5             # Reduce volatility benefit
                )
        return total_loss, {
            'structural_loss': structural_loss,
            'left_tail_robustness': left_tail_robustness,
            'volatility_benefit': volatility_benefit,
            'convex_response': convex_response,
            'fragility_heuristic': fragility_heuristic,
            'antifragile_gain': antifragile_gain
        }
    
        # return total_loss, {
        #     'base_loss': base_loss,
        #     'volatility_gain': volatility_gain,
        #     'progressive_gain': progressive_gain,
        #     'tail_preparedness': tail_preparedness,
        #     'diversity_gain': diversity_gain,
        #     'stress_response': stress_response,
        #     'total_antifragile_gain': total_antifragile_gain
        # }
    
    def calculate_structural_preservation(self, flow, nominal_data: torch.Tensor, 
                                        target_data: torch.Tensor, context_dim: int) -> torch.Tensor:
        """Strong structural preservation to maintain banana shape."""
        
        # Standard likelihood losses
        nominal_context = torch.zeros(len(nominal_data), context_dim, device=self.device)
        nominal_dist = flow(nominal_context)
        nominal_loss = -torch.mean(nominal_dist.log_prob(nominal_data))
        
        target_context = torch.ones(len(target_data), context_dim, device=self.device)
        target_dist = flow(target_context)
        target_loss = -torch.mean(target_dist.log_prob(target_data))
        
        # Bounded generation constraint
        test_context = torch.ones(100, context_dim, device=self.device) * 0.5
        test_dist = flow(test_context)
        test_samples = test_dist.sample()
        
        # Penalize samples far outside reasonable bounds
        data_bounds = torch.cat([nominal_data, target_data])
        data_min, data_max = torch.min(data_bounds, dim=0)[0], torch.max(data_bounds, dim=0)[0]
        
        bound_violations = torch.mean(torch.relu(test_samples - data_max * 1.3)) + \
                          torch.mean(torch.relu(data_min * 1.3 - test_samples))
        
        return nominal_loss + target_loss + bound_violations * 0.8
    
    def calculate_left_tail_robustness_taleb(self, flow, target_data: torch.Tensor, context_dim: int) -> torch.Tensor:
        """
        Taleb's b-robustness: V(X, f_λ, K', s^-(λ)) ≤ b for any K' ≤ K
        System must be robust to adverse perturbations (limited downside).
        """
        
        adverse_contexts = [0.1, 0.2, 0.3, 0.4]  # Increasingly adverse conditions
        degradation_scores = []
        
        # Baseline performance
        baseline_context = torch.ones(len(target_data), context_dim, device=self.device) * 0.5
        baseline_dist = flow(baseline_context)
        baseline_perf = torch.mean(baseline_dist.log_prob(target_data))
        
        for adverse_ctx in adverse_contexts:
            # Test under adverse conditions (NO NOISE ADDED TO DATA!)
            adverse_context = torch.ones(len(target_data), context_dim, device=self.device) * adverse_ctx
            adverse_dist = flow(adverse_context)
            adverse_perf = torch.mean(adverse_dist.log_prob(target_data))  # Same data!
            
            # Measure degradation (should be limited for robust systems)
            degradation = torch.clamp(baseline_perf - adverse_perf, min=0)
            degradation_scores.append(degradation)
        
        # Robustness = limited maximum degradation
        max_degradation = torch.max(torch.stack(degradation_scores))
        
        return max_degradation  # Minimize this (robust = low sensitivity to adverse conditions)
    
    def calculate_right_tail_benefit_taleb(self, flow, target_data: torch.Tensor, context_dim: int) -> torch.Tensor:
        """
        Taleb's fat right tail: system should benefit from favorable conditions.
        This implements the "long vega" property of antifragile systems.
        """
        
        favorable_contexts = [0.6, 0.7, 0.8, 0.9]  # Increasingly favorable conditions
        benefit_scores = []
        
        # Baseline performance
        baseline_context = torch.ones(len(target_data), context_dim, device=self.device) * 0.5
        baseline_dist = flow(baseline_context)
        baseline_perf = torch.mean(baseline_dist.log_prob(target_data))
        
        for favorable_ctx in favorable_contexts:
            # Test under favorable conditions (NO NOISE ADDED TO DATA!)
            favorable_context = torch.ones(len(target_data), context_dim, device=self.device) * favorable_ctx
            favorable_dist = flow(favorable_context)
            favorable_perf = torch.mean(favorable_dist.log_prob(target_data))  # Same data!
            
            # Measure benefit (should increase for antifragile systems)
            benefit = torch.clamp(favorable_perf - baseline_perf, min=0)
            benefit_scores.append(benefit)
        
        # Antifragility = increasing benefit from favorable conditions
        total_benefit = torch.sum(torch.stack(benefit_scores))
        
        return -total_benefit  # Negative because we minimize loss (want to maximize benefit)
    
    def calculate_convex_response_taleb(self, flow, target_data: torch.Tensor, context_dim: int) -> torch.Tensor:
        """
        Taleb's convexity test: f''(x) > 0
        Antifragile systems have convex response to meaningful volatility.
        """
        
        # Test response across context spectrum (meaningful volatility, not noise!)
        context_values = torch.linspace(0.2, 0.8, 7).to(self.device)
        performances = []
        
        for ctx_val in context_values:
            context = torch.ones(len(target_data), context_dim, device=self.device) * ctx_val
            dist = flow(context)
            perf = torch.mean(dist.log_prob(target_data))  # Always same clean data!
            performances.append(perf)
        
        perfs = torch.stack(performances)
        
        # Calculate second derivative (convexity)
        if len(perfs) >= 3:
            # f''(x) ≈ f(x+h) - 2f(x) + f(x-h)
            second_derivatives = perfs[2:] - 2*perfs[1:-1] + perfs[:-2]
            convexity = torch.mean(second_derivatives)
            
            # Antifragile: positive second derivative (convex response)
            return -convexity  # Negative because we minimize (want positive convexity)
        else:
            return torch.tensor(0.0, device=self.device)
    
    def calculate_taleb_h_ratio(self, flow, target_data: torch.Tensor, context_dim: int) -> torch.Tensor:
        """
        Taleb's fragility detection heuristic:
        H = (V(p + Δp) + V(p - Δp)) / (2 * V(p))
        
        H > 1: fragile
        H = 1: robust  
        H < 1: antifragile
        """
        
        base_context = 0.5
        delta_p = 0.1
        
        # Baseline
        base_ctx = torch.ones(len(target_data), context_dim, device=self.device) * base_context
        base_dist = flow(base_ctx)
        base_perf = torch.mean(base_dist.log_prob(target_data))
        
        # Perturbed contexts (NOT perturbed data!)
        up_ctx = torch.ones(len(target_data), context_dim, device=self.device) * (base_context + delta_p)
        up_dist = flow(up_ctx)
        up_perf = torch.mean(up_dist.log_prob(target_data))  # Same data!
        
        down_ctx = torch.ones(len(target_data), context_dim, device=self.device) * (base_context - delta_p)
        down_dist = flow(down_ctx)
        down_perf = torch.mean(down_dist.log_prob(target_data))  # Same data!
        
        # Taleb's H-ratio
        numerator = torch.abs(up_perf - base_perf) + torch.abs(down_perf - base_perf)
        denominator = 2 * torch.abs(base_perf) + 1e-8
        
        H_ratio = numerator / denominator
        
        # Fragility penalty: punish H > 1, reward H < 1
        fragility_penalty = torch.clamp(H_ratio - 1.0, min=0)
        
        return fragility_penalty
    
    def calculate_volatility_benefit_corrected(self, flow, target_data: torch.Tensor, context_dim: int) -> torch.Tensor:
        """
        CORRECTED volatility benefit: Test model's response to context volatility,
        NOT data noise. This is what true antifragility looks like.
        """
        
        # Base context
        base_context = 0.5
        context_volatility_levels = [0.0, 0.1, 0.2, 0.3]  # Volatility in CONTEXT, not data
        
        performance_curve = []
        
        for vol_level in context_volatility_levels:
            if vol_level > 0:
                # Add volatility to CONTEXT, not data
                context_noise = torch.randn(len(target_data), context_dim, device=self.device) * vol_level
                perturbed_context = torch.clamp(base_context + context_noise, 0, 1)
            else:
                perturbed_context = torch.ones(len(target_data), context_dim, device=self.device) * base_context
            
            # Test on CLEAN data with perturbed context
            dist = flow(perturbed_context)
            performance = torch.mean(dist.log_prob(target_data))  # Clean target data!
            performance_curve.append(performance)
        
        performances = torch.stack(performance_curve)
        
        # Antifragile: benefit from context volatility
        linear_trend = performances[-1] - performances[0]
        
        # Convexity in response
        if len(performances) >= 3:
            second_deriv = performances[:-2] - 2*performances[1:-1] + performances[2:]
            convexity = torch.mean(second_deriv)
        else:
            convexity = torch.tensor(0.0, device=self.device)
        
        # True volatility benefit: positive trend + convex response
        volatility_benefit = linear_trend + convexity * 0.5
        
        return -volatility_benefit  # Negative because we minimize loss