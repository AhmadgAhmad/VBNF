# antifragile_vbnf/evaluation/stress_tests.py
import torch
import numpy as np
from typing import Dict, List, Tuple
from ..data.generators import generate_semicircle

class StressTester:
    """Main class for conducting stress tests on trained models."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def run_all_stress_tests(self, standard_flow, antifragile_flow, 
                            nominal_data: torch.Tensor, target_data: torch.Tensor,
                            vis_flg: bool = True) -> Dict:
        """Run all standard and advanced stress tests."""
        
        print("Running standard stress tests...")
        distribution_results = self.perform_distribution_shift_tests(
            standard_flow, antifragile_flow, nominal_data, target_data
        )
        
        black_swan_results = self.perform_black_swan_tests(
            standard_flow, antifragile_flow, nominal_data, target_data
        )
        
        dynamic_results = self.perform_dynamic_environment_tests(
            standard_flow, antifragile_flow, nominal_data, target_data
        )
        
        transfer_results = self.perform_transfer_function_stress_analysis(
            standard_flow, antifragile_flow, nominal_data, target_data
        )
        
        # Run advanced antifragility tests
        print("Running advanced antifragility tests...")
        advanced_results = self.perform_advanced_antifragility_tests(
            standard_flow, antifragile_flow, nominal_data, target_data
        )
        
        # Combine all results
        all_results = {
            'distribution_shift': distribution_results,
            'black_swan': black_swan_results,
            'dynamic_environment': dynamic_results,
            'transfer_function': transfer_results,
            'advanced_antifragility': advanced_results
        }
        
        return all_results
    
    def perform_distribution_shift_tests(self, standard_flow, antifragile_flow, 
                                        nominal_data: torch.Tensor, target_data: torch.Tensor) -> Dict:
        """Test model performance under various distribution shifts."""
        results = {}
        
        # 1.1. Progressive Horizontal Shift
        shifts = np.linspace(1.0, 3.0, 5)  # From normal shift to extreme shift
        horizontal_results = []
        
        for shift in shifts:
            # Create shifted test data
            shifted_center = [shift, 0]
            shifted_target = generate_semicircle(shifted_center, 1.0, 100, [np.pi, 2*np.pi], 0.1)
            shifted_target_tensor = torch.tensor(shifted_target, dtype=torch.float32, device=self.device)
            
            # Evaluate both models
            std_log_probs = self.evaluate_model_on_data(standard_flow, shifted_target_tensor)
            anti_log_probs = self.evaluate_model_on_data(antifragile_flow, shifted_target_tensor)
            
            horizontal_results.append({
                'shift': shift,
                'std_log_prob': std_log_probs.mean().item(),
                'anti_log_prob': anti_log_probs.mean().item(),
                'diff_pct': (anti_log_probs.mean() - std_log_probs.mean()) / abs(std_log_probs.mean()) * 100
            })
        
        results['horizontal_shift'] = horizontal_results
        
        # 1.2. Noise Level Variation
        noise_levels = np.linspace(0.1, 0.5, 5)  # From normal noise to high noise
        noise_results = []
        
        for noise in noise_levels:
            # Create noisier test data
            noisy_target = generate_semicircle([1.0, 0], 1.0, 100, [np.pi, 2*np.pi], noise)
            noisy_target_tensor = torch.tensor(noisy_target, dtype=torch.float32, device=self.device)
            
            # Evaluate both models
            std_log_probs = self.evaluate_model_on_data(standard_flow, noisy_target_tensor)
            anti_log_probs = self.evaluate_model_on_data(antifragile_flow, noisy_target_tensor)
            
            noise_results.append({
                'noise': noise,
                'std_log_prob': std_log_probs.mean().item(),
                'anti_log_prob': anti_log_probs.mean().item(),
                'diff_pct': (anti_log_probs.mean() - std_log_probs.mean()) / abs(std_log_probs.mean()) * 100
            })
        
        results['noise_variation'] = noise_results
        
        # 1.3. Shape Deformation
        radius_ratios = np.linspace(1.0, 2.0, 5)  # From circle to ellipse
        shape_results = []
        
        for ratio in radius_ratios:
            # Create deformed test data (elliptical instead of circular)
            deformed_target = []
            angles = np.linspace(np.pi, 2*np.pi, 100)
            for angle in angles:
                x = 1.0 + 1.0 * np.cos(angle)
                y = 0.0 + 1.0/ratio * np.sin(angle)  # Compress y-axis
                deformed_target.append([x, y])
            deformed_target = np.array(deformed_target)
            
            # Add noise
            deformed_target += np.random.normal(0, 0.1, deformed_target.shape)
            deformed_target_tensor = torch.tensor(deformed_target, dtype=torch.float32, device=self.device)
            
            # Evaluate both models
            std_log_probs = self.evaluate_model_on_data(standard_flow, deformed_target_tensor)
            anti_log_probs = self.evaluate_model_on_data(antifragile_flow, deformed_target_tensor)
            
            shape_results.append({
                'ratio': ratio,
                'std_log_prob': std_log_probs.mean().item(),
                'anti_log_prob': anti_log_probs.mean().item(),
                'diff_pct': (anti_log_probs.mean() - std_log_probs.mean()) / abs(std_log_probs.mean()) * 100
            })
        
        results['shape_deformation'] = shape_results
        
        return results
    
    def perform_black_swan_tests(self, standard_flow, antifragile_flow, 
                                nominal_data: torch.Tensor, target_data: torch.Tensor) -> Dict:
        """Test model performance when extreme outliers are injected."""
        results = {}
        
        # 2.1. Single Extreme Outlier
        outlier_magnitudes = [2.0, 3.0, 4.0, 5.0, 6.0]
        single_outlier_results = []
        
        for magnitude in outlier_magnitudes:
            # Create target data with one extreme outlier
            modified_target = target_data.clone()
            # Add extreme outlier at position [magnitude, 0]
            outlier = torch.tensor([[magnitude, 0.0]], dtype=torch.float32, device=self.device)
            outlier_target = torch.cat([modified_target, outlier], dim=0)
            
            # Evaluate both models
            std_log_probs = self.evaluate_model_on_data(standard_flow, outlier_target)
            anti_log_probs = self.evaluate_model_on_data(antifragile_flow, outlier_target)
            
            # Calculate outlier impact - how much does the outlier affect the overall log prob
            std_clean_log_probs = self.evaluate_model_on_data(standard_flow, modified_target).mean()
            anti_clean_log_probs = self.evaluate_model_on_data(antifragile_flow, modified_target).mean()
            
            std_outlier_impact = (std_log_probs.mean() - std_clean_log_probs) / abs(std_clean_log_probs) * 100
            anti_outlier_impact = (anti_log_probs.mean() - anti_clean_log_probs) / abs(anti_clean_log_probs) * 100
            
            single_outlier_results.append({
                'magnitude': magnitude,
                'std_impact_pct': std_outlier_impact.item(),
                'anti_impact_pct': anti_outlier_impact.item(),
                'relative_robustness': anti_outlier_impact.item() / std_outlier_impact.item()
            })
        
        results['single_outlier'] = single_outlier_results
        
        # 2.2. Cluster of Outliers
        cluster_sizes = [1, 2, 5, 10]
        cluster_results = []
        
        for size in cluster_sizes:
            # Create a cluster of outliers in an unexpected region
            outlier_cluster = torch.randn(size, 2, device=self.device) * 0.2 + torch.tensor([3.0, 1.5], device=self.device)
            outlier_target = torch.cat([target_data, outlier_cluster], dim=0)
            
            # Evaluate both models
            std_log_probs = self.evaluate_model_on_data(standard_flow, outlier_target)
            anti_log_probs = self.evaluate_model_on_data(antifragile_flow, outlier_target)
            
            # Calculate outlier impact
            std_clean_log_probs = self.evaluate_model_on_data(standard_flow, target_data).mean()
            anti_clean_log_probs = self.evaluate_model_on_data(antifragile_flow, target_data).mean()
            
            std_outlier_impact = (std_log_probs.mean() - std_clean_log_probs) / abs(std_clean_log_probs) * 100
            anti_outlier_impact = (anti_log_probs.mean() - anti_clean_log_probs) / abs(anti_clean_log_probs) * 100
            
            cluster_results.append({
                'cluster_size': size,
                'std_impact_pct': std_outlier_impact.item(),
                'anti_impact_pct': anti_outlier_impact.item(),
                'relative_robustness': anti_outlier_impact.item() / std_outlier_impact.item()
            })
        
        results['outlier_cluster'] = cluster_results
        
        return results
    
    def perform_dynamic_environment_tests(self, standard_flow, antifragile_flow, 
                                         nominal_data: torch.Tensor, target_data: torch.Tensor) -> Dict:
        """Test model performance in dynamically changing environments."""
        results = {}
        
        # 3.1. Progressive Distribution Drift
        drift_steps = 10
        drift_results = []
        
        # Define starting and ending distributions
        start_center = [0, 0]
        end_center = [2.5, 0.5]  # More extreme than normal target
        
        for step in range(drift_steps):
            # Interpolate between start and end
            t = step / (drift_steps - 1)
            current_center = [
                start_center[0] * (1-t) + end_center[0] * t,
                start_center[1] * (1-t) + end_center[1] * t
            ]
            
            # Create drifted data
            if step < drift_steps / 2:
                # First half: modified nominal (upper semicircle)
                angle_range = [0, np.pi]
            else:
                # Second half: modified target (lower semicircle)
                angle_range = [np.pi, 2*np.pi]
                
            drift_data = generate_semicircle(current_center, 1.0, 100, angle_range, 0.1)
            drift_tensor = torch.tensor(drift_data, dtype=torch.float32, device=self.device)
            
            # Evaluate both models
            std_log_probs = self.evaluate_model_on_data(standard_flow, drift_tensor)
            anti_log_probs = self.evaluate_model_on_data(antifragile_flow, drift_tensor)
            
            drift_results.append({
                'step': step,
                't': t,
                'center_x': current_center[0],
                'center_y': current_center[1],
                'std_log_prob': std_log_probs.mean().item(),
                'anti_log_prob': anti_log_probs.mean().item(),
                'diff_pct': (anti_log_probs.mean() - std_log_probs.mean()) / abs(std_log_probs.mean()) * 100
            })
        
        results['distribution_drift'] = drift_results
        
        # 3.2. Oscillating Environment
        oscillation_steps = 10
        oscillation_results = []
        
        for step in range(oscillation_steps):
            # Calculate oscillation parameter (0 to 1 back to 0)
            t = 0.5 * (1 - np.cos(2 * np.pi * step / oscillation_steps))
            
            # Interpolate between nominal and target
            if t < 0.5:
                # More nominal-like
                center = [t * 2, 0]
                angle_range = [t * 2 * np.pi, np.pi + t * np.pi]
            else:
                # More target-like
                center = [(2*t - 1) * 1.5, 0]
                angle_range = [np.pi * (1 - 2*(t-0.5)), 2*np.pi - np.pi * 2*(t-0.5)]
                
            oscillation_data = generate_semicircle(center, 1.0, 100, angle_range, 0.1)
            oscillation_tensor = torch.tensor(oscillation_data, dtype=torch.float32, device=self.device)
            
            # Evaluate both models
            std_log_probs = self.evaluate_model_on_data(standard_flow, oscillation_tensor)
            anti_log_probs = self.evaluate_model_on_data(antifragile_flow, oscillation_tensor)
            
            oscillation_results.append({
                'step': step,
                't': t,
                'std_log_prob': std_log_probs.mean().item(),
                'anti_log_prob': anti_log_probs.mean().item(),
                'diff_pct': (anti_log_probs.mean() - std_log_probs.mean()) / abs(std_log_probs.mean()) * 100
            })
        
        results['oscillation'] = oscillation_results
        
        return results
    
    def perform_transfer_function_stress_analysis(self, standard_flow, antifragile_flow, 
                                                 nominal_data: torch.Tensor, target_data: torch.Tensor) -> Dict:
        """Test how the transfer function changes under extreme conditions."""
        from ..core.antifragility import AntifragilityCalculator
        
        results = {}
        antifragility_calc = AntifragilityCalculator(self.device)
        
        # Calculate reference points
        omega = torch.mean(nominal_data, dim=0)
        target_mean = torch.mean(target_data, dim=0)
        k = omega - 0.5 * (target_mean - omega)  # Stress level point
        
        # 4.1. Changing Stress Level (K)
        k_factors = np.linspace(0.2, 2.0, 5)  # From less stressful to more stressful
        k_results = []
        
        for factor in k_factors:
            # Modify stress level
            modified_k = omega - factor * (target_mean - omega)
            
            # Calculate transfer function with modified k
            std_transfer = antifragility_calc.calculate_transfer_function(
                standard_flow, [0.0, 0.5, 1.0], omega, modified_k
            )
            anti_transfer = antifragility_calc.calculate_transfer_function(
                antifragile_flow, [0.0, 0.5, 1.0], omega, modified_k
            )
            
            # Extract average H values
            std_h_values = [np.mean([v for v in ctx_vals.values()]) for ctx_vals in std_transfer.values()]
            anti_h_values = [np.mean([v for v in ctx_vals.values()]) for ctx_vals in anti_transfer.values()]
            
            # Calculate fragility metrics
            std_fragility = sum(1 for h in std_h_values if h < 0) / len(std_h_values)
            anti_fragility = sum(1 for h in anti_h_values if h < 0) / len(anti_h_values)
            
            k_results.append({
                'k_factor': factor,
                'std_avg_h': np.mean(std_h_values),
                'anti_avg_h': np.mean(anti_h_values),
                'std_fragility_ratio': std_fragility,
                'anti_fragility_ratio': anti_fragility,
                'relative_improvement': (anti_fragility - std_fragility) / (std_fragility + 1e-10)
            })
        
        results['stress_level_variation'] = k_results
        
        # 4.2. Extreme Context Interpolation
        extreme_contexts = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        extreme_results = []
        
        for ctx in extreme_contexts:
            # Generate samples
            with torch.no_grad():
                std_ctx = torch.tensor([[ctx]], device=self.device)
                anti_ctx = torch.tensor([[ctx]], device=self.device)
                
                std_dist = standard_flow(std_ctx)
                anti_dist = antifragile_flow(anti_ctx)
                
                std_samples = std_dist.sample((100,))
                anti_samples = anti_dist.sample((100,))
            
            # Calculate metrics
            std_mean = std_samples.mean(dim=0).cpu().numpy()
            anti_mean = anti_samples.mean(dim=0).cpu().numpy()
            
            std_var = std_samples.var(dim=0).cpu().numpy()
            anti_var = anti_samples.var(dim=0).cpu().numpy()
            
            # Calculate distance from expected interpolation
            expected_x = (1-ctx) * omega[0].item() + ctx * target_mean[0].item()
            expected_y = (1-ctx) * omega[1].item() + ctx * target_mean[1].item()
            
            std_distance = np.sqrt((std_mean[0,0] - expected_x)**2 + (std_mean[0,1] - expected_y)**2)
            anti_distance = np.sqrt((anti_mean[0,0] - expected_x)**2 + (anti_mean[0,1] - expected_y)**2)
            
            extreme_results.append({
                'context': ctx,
                'std_distance': std_distance,
                'anti_distance': anti_distance,
                'std_variance': np.mean(std_var),
                'anti_variance': np.mean(anti_var),
                'relative_distance_improvement': (std_distance - anti_distance) / std_distance
            })
        
        results['extreme_context'] = extreme_results
        
        return results
    
    def perform_advanced_antifragility_tests(self, standard_flow, antifragile_flow, 
                                            nominal_data: torch.Tensor, target_data: torch.Tensor) -> Dict:
        """Perform specialized tests focusing on specific antifragility properties."""
        from .advanced_tests import AdvancedAntifragilityTester
        
        advanced_tester = AdvancedAntifragilityTester(self.device)
        return advanced_tester.run_all_tests(standard_flow, antifragile_flow, nominal_data, target_data)
    
    def evaluate_model_on_data(self, flow, data: torch.Tensor) -> torch.Tensor:
        """Evaluate model log probability on given data."""
        with torch.no_grad():
            # Try different context values to find best fit
            best_log_probs = None
            for context in torch.linspace(0, 1, 11, device=self.device):
                ctx = torch.ones(len(data), 1, device=self.device) * context
                dist = flow(ctx)
                log_probs = dist.log_prob(data)
                
                if best_log_probs is None or log_probs.mean() > best_log_probs.mean():
                    best_log_probs = log_probs
        
        return best_log_probs