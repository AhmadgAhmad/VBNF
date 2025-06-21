# antifragile_calnf/evaluation/advanced_tests.py
import torch
import numpy as np
from typing import Dict, List
import zuko

class AdvancedAntifragilityTester:
    """Specialized tests focusing on specific antifragility properties."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.dtype = torch.float32
    
    def run_all_tests(self, standard_flow, antifragile_flow, 
                     nominal_data: torch.Tensor, target_data: torch.Tensor) -> Dict:
        """Run all advanced antifragility tests."""
        
        # Ensure consistent dtype throughout
        nominal_data = nominal_data.to(self.dtype).to(self.device)
        target_data = target_data.to(self.dtype).to(self.device)
        
        results = {}
        
        # 7.1. Volatility Response Test (Jensen's Gap)
        print("Running volatility response tests...")
        results['volatility_response'] = self.test_volatility_response(
            standard_flow, antifragile_flow, nominal_data, target_data
        )
        
        # 7.2. Convexity/Concavity Test
        print("Running convexity tests...")
        results['convexity'] = self.test_convexity(
            standard_flow, antifragile_flow, nominal_data, target_data
        )
        
        # 7.3. Barbell Strategy Test
        print("Running barbell strategy tests...")
        results['barbell_strategy'] = self.test_barbell_strategy(
            standard_flow, antifragile_flow
        )
        
        # 7.4. Recovery Speed Test
        print("Running recovery speed tests...")
        results['recovery_speed'] = self.test_recovery_speed(
            standard_flow, antifragile_flow, nominal_data, target_data
        )
        
        return results
    
    def test_volatility_response(self, standard_flow, antifragile_flow, 
                                nominal_data: torch.Tensor, target_data: torch.Tensor) -> List[Dict]:
        """Test if model actually *benefits* from certain types of volatility."""
        volatility_levels = np.linspace(0.05, 0.5, 10)
        volatility_results = []
        
        # Reference points
        omega = torch.mean(nominal_data, dim=0)
        target_mean = torch.mean(target_data, dim=0)
        
        for vol in volatility_levels:
            # Generate samples from both models at different context values
            contexts = [0.25, 0.5, 0.75]  # Focus on the transition region
            
            std_gains = []
            anti_gains = []
            
            for ctx in contexts:
                with torch.no_grad():
                    # Base performance (no volatility)
                    ctx_tensor = torch.tensor([[ctx]], dtype=self.dtype, device=self.device)
                    std_dist = standard_flow(ctx_tensor)
                    anti_dist = antifragile_flow(ctx_tensor)
                    
                    # Generate test points in the region between nominal and target
                    t = torch.linspace(0, 1, 100, dtype=self.dtype, device=self.device).reshape(-1, 1)
                    test_points = omega * (1-t) + target_mean * t
                    
                    # Base log probabilities
                    std_base_log_probs = std_dist.log_prob(test_points).mean()
                    anti_base_log_probs = anti_dist.log_prob(test_points).mean()
                    
                    # With volatility - use multiple context values and average
                    n_samples = 10
                    ctx_samples = torch.normal(ctx, vol, (n_samples, 1), dtype=self.dtype, device=self.device)
                    ctx_samples = torch.clamp(ctx_samples, 0.0, 1.0)  # Keep within valid range
                    
                    std_vol_log_probs = []
                    anti_vol_log_probs = []
                    
                    for c in ctx_samples:
                        c_tensor = c.reshape(1, 1)
                        std_vol_dist = standard_flow(c_tensor)
                        anti_vol_dist = antifragile_flow(c_tensor)
                        
                        std_vol_log_probs.append(std_vol_dist.log_prob(test_points).mean())
                        anti_vol_log_probs.append(anti_vol_dist.log_prob(test_points).mean())
                    
                    # Average log probs under volatility
                    std_vol_avg = torch.stack(std_vol_log_probs).mean()
                    anti_vol_avg = torch.stack(anti_vol_log_probs).mean()
                    
                    # Calculate Jensen's gap = E[f(X)] - f(E[X])
                    std_jensen_gap = std_vol_avg - std_base_log_probs
                    anti_jensen_gap = anti_vol_avg - anti_base_log_probs
                    
                    std_gains.append(std_jensen_gap.item())
                    anti_gains.append(anti_jensen_gap.item())
            
            # Average across contexts
            std_avg_gain = np.mean(std_gains)
            anti_avg_gain = np.mean(anti_gains)
            
            volatility_results.append({
                'volatility': vol,
                'std_jensen_gap': std_avg_gain,
                'anti_jensen_gap': anti_avg_gain,
                'difference': anti_avg_gain - std_avg_gain
            })
        
        return volatility_results
    
    def test_convexity(self, standard_flow, antifragile_flow, 
                      nominal_data: torch.Tensor, target_data: torch.Tensor) -> List[Dict]:
        """Test the convexity/concavity of the log-likelihood surface."""
        convexity_results = []
        
        contexts = np.linspace(0, 1, 11)
        
        # For each pair of context values, test convexity/concavity
        for i in range(len(contexts)-1):
            for j in range(i+1, len(contexts)):
                ctx_i = contexts[i]
                ctx_j = contexts[j]
                
                # Interpolation point
                ctx_mid = (ctx_i + ctx_j) / 2
                
                with torch.no_grad():
                    # Evaluate at endpoints and midpoint
                    ctx_i_tensor = torch.tensor([[ctx_i]], dtype=self.dtype, device=self.device)
                    ctx_j_tensor = torch.tensor([[ctx_j]], dtype=self.dtype, device=self.device)
                    ctx_mid_tensor = torch.tensor([[ctx_mid]], dtype=self.dtype, device=self.device)
                    
                    # Standard flow evaluations
                    std_dist_i = standard_flow(ctx_i_tensor)
                    std_dist_j = standard_flow(ctx_j_tensor)
                    std_dist_mid = standard_flow(ctx_mid_tensor)
                    
                    # Antifragile flow evaluations
                    anti_dist_i = antifragile_flow(ctx_i_tensor)
                    anti_dist_j = antifragile_flow(ctx_j_tensor)
                    anti_dist_mid = antifragile_flow(ctx_mid_tensor)
                    
                    # Generate test points
                    if len(nominal_data) >= 50:
                        nominal_subset = nominal_data[:50]
                    else:
                        nominal_subset = nominal_data
                        
                    test_points = torch.cat([nominal_subset, target_data], dim=0)
                    
                    # Log probabilities
                    std_log_prob_i = std_dist_i.log_prob(test_points).mean()
                    std_log_prob_j = std_dist_j.log_prob(test_points).mean()
                    std_log_prob_mid = std_dist_mid.log_prob(test_points).mean()
                    
                    anti_log_prob_i = anti_dist_i.log_prob(test_points).mean()
                    anti_log_prob_j = anti_dist_j.log_prob(test_points).mean()
                    anti_log_prob_mid = anti_dist_mid.log_prob(test_points).mean()
                    
                    # Convexity measure: f(mid) - (f(i) + f(j))/2
                    # Negative = convex, Positive = concave
                    std_convexity = std_log_prob_mid - (std_log_prob_i + std_log_prob_j) / 2
                    anti_convexity = anti_log_prob_mid - (anti_log_prob_i + anti_log_prob_j) / 2
                    
                    convexity_results.append({
                        'ctx_i': ctx_i,
                        'ctx_j': ctx_j,
                        'ctx_mid': ctx_mid,
                        'std_convexity': std_convexity.item(),
                        'anti_convexity': anti_convexity.item(),
                        'difference': anti_convexity.item() - std_convexity.item()
                    })
        
        return convexity_results
    
    def test_barbell_strategy(self, standard_flow, antifragile_flow) -> List[Dict]:
        """Test if the antifragile model implements a "barbell" strategy."""
        variance_results = []
        
        # We'll measure the variance of samples at different context values
        for ctx in np.linspace(0, 1, 21):
            with torch.no_grad():
                ctx_tensor = torch.tensor([[ctx]], dtype=self.dtype, device=self.device)
                
                # Generate samples
                std_dist = standard_flow(ctx_tensor)
                anti_dist = antifragile_flow(ctx_tensor)
                
                std_samples = std_dist.sample((1000,))
                anti_samples = anti_dist.sample((1000,))
                
                # Calculate variance
                std_var = std_samples.var(dim=0).mean().item()
                anti_var = anti_samples.var(dim=0).mean().item()
                
                variance_results.append({
                    'context': ctx,
                    'std_variance': std_var,
                    'anti_variance': anti_var,
                    'ratio': anti_var / std_var if std_var != 0 else float('inf')
                })
        
        return variance_results
    
    def test_recovery_speed(self, standard_flow, antifragile_flow, 
                           nominal_data: torch.Tensor, target_data: torch.Tensor) -> List[Dict]:
        """Test how quickly the models recover from perturbations."""
        recovery_results = []
        
        # Define perturbation levels
        perturbation_magnitudes = [0.5, 1.0, 1.5, 2.0]
        
        for magnitude in perturbation_magnitudes:
            # Create perturbed version of target data
            perturbed_target = target_data.clone()
            perturbation = torch.randn_like(perturbed_target, dtype=self.dtype, device=self.device) * magnitude
            perturbed_target += perturbation
            
            # Track recovery metrics at different training steps
            recovery_steps = [5, 10, 20, 50, 100]
            
            # Function to evaluate on clean target data
            def evaluate_recovery(flow):
                with torch.no_grad():
                    ctx = torch.ones(len(target_data), 1, dtype=self.dtype, device=self.device)
                    dist = flow(ctx)
                    return dist.log_prob(target_data).mean().item()
            
            # Get baseline performance
            std_baseline = evaluate_recovery(standard_flow)
            anti_baseline = evaluate_recovery(antifragile_flow)
            
            # Fine-tune on perturbed data
            for steps in recovery_steps:
                # Clone the models to avoid modifying the originals
                from ..core.training import train_flow
                
                std_flow_copy = zuko.flows.NSF(features=2, context=1, transforms=5, hidden_features=[64, 64])
                std_flow_copy.load_state_dict(standard_flow.state_dict())
                std_flow_copy = std_flow_copy.to(self.device)
                
                anti_flow_copy = zuko.flows.NSF(features=2, context=1, transforms=5, hidden_features=[64, 64])
                anti_flow_copy.load_state_dict(antifragile_flow.state_dict())
                anti_flow_copy = anti_flow_copy.to(self.device)
                
                # Fine-tune for given number of steps
                std_recovery_flow, _ = train_flow(
                    std_flow_copy, nominal_data, perturbed_target, 
                    context_dim=1, num_steps=steps, lr=1e-3, subsets=3, 
                    use_antifragile=False, device=self.device
                )
                
                anti_recovery_flow, _ = train_flow(
                    anti_flow_copy, nominal_data, perturbed_target, 
                    context_dim=1, num_steps=steps, lr=1e-3, subsets=3, 
                    use_antifragile=True, device=self.device
                )
                
                # Evaluate recovery on clean data
                std_recovery = evaluate_recovery(std_recovery_flow)
                anti_recovery = evaluate_recovery(anti_recovery_flow)
                
                # Calculate recovery percentage
                std_recovery_pct = (std_recovery - std_baseline) / abs(std_baseline) * 100
                anti_recovery_pct = (anti_recovery - anti_baseline) / abs(anti_baseline) * 100
                
                recovery_results.append({
                    'magnitude': magnitude,
                    'steps': steps,
                    'std_recovery_pct': std_recovery_pct,
                    'anti_recovery_pct': anti_recovery_pct,
                    'difference': anti_recovery_pct - std_recovery_pct
                })
        
        return recovery_results