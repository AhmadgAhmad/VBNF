# antifragile_vbnf/core/training.py
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
from core.antifragility import AntifragilityCalculator, AdaptiveWeightCalculator
from core.losses import AntifragileLossCalculator

class FlowTrainer:
    """Main trainer class for normalizing flows with antifragile capabilities."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.antifragility_calc = AntifragilityCalculator(device)
        self.adaptive_weights = AdaptiveWeightCalculator()
        self.loss_calculator = AntifragileLossCalculator(device)
        
    def train_flow(self, flow, nominal_data: torch.Tensor, target_data: torch.Tensor, 
                   context_dim: int = 1, num_steps: int = 1000, lr: float = 1e-3, 
                   subsets: int = 5, use_antifragile: bool = False) -> Tuple[nn.Module, pd.DataFrame]:
        """Train a normalizing flow with adaptive antifragile weighting."""
        
        # Move data to device
        nominal_data = nominal_data.to(self.device)
        target_data = target_data.to(self.device)
        flow = flow.to(self.device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
        
        # Track losses and metrics
        losses = []
        metrics_history = []  # For tracking performance metrics
        
        # State history for antifragility
        state_history = []
        
        # Calculate reference points for antifragility
        omega = torch.mean(nominal_data, dim=0)  # Center of nominal distribution
        target_mean = torch.mean(target_data, dim=0)  # Center of target distribution
        k = omega - 0.5 * (target_mean - omega)  # Stress level point
        
        # Target thresholds for key metrics
        target_thresholds = {
            'recovery_speed': 1.0,  # How quickly distribution recovers from perturbations
            'volatility_response': 1.0,  # How well model handles volatility 
            'outlier_robustness': 1.0,  # How well model handles outliers
            'interpolation_quality': 1.0  # Quality of interpolation between distributions
        }
        
        # Initialize adaptive weights
        adaptive_weights = {metric: 1.0 for metric in target_thresholds}
        
        # Create K random subsets of target data
        target_subsets = self._create_target_subsets(target_data, subsets)
        
        # Training loop
        pbar = tqdm(range(num_steps))
        for step in pbar:
            optimizer.zero_grad()
            
            # Calculate losses
            total_loss, loss_components = self._calculate_training_loss(
                flow, nominal_data, target_data, target_subsets, context_dim, 
                step, use_antifragile, state_history, omega, target_mean, k,
                target_thresholds, adaptive_weights, metrics_history
            )
            
            # Step optimizer
            total_loss.backward()
            optimizer.step()
            
            # Record loss
            loss_values = {
                'step': step,
                'total_loss': total_loss.item(),
                **{k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_components.items()}
            }
            
            losses.append(loss_values)
            
            # Update progress bar
            pbar_desc = f"Step {step} | Total: {total_loss.item():.3f}"
            if 'nominal_loss' in loss_components:
                pbar_desc += f" | Nominal: {loss_components['nominal_loss'].item():.3f}"
            if 'target_loss' in loss_components:
                pbar_desc += f" | Target: {loss_components['target_loss'].item():.3f}"
            if use_antifragile and 'antifragile_loss' in loss_components:
                pbar_desc += f" | Anti: {loss_components['antifragile_loss'].item():.3f}"
            pbar.set_description(pbar_desc)
        
        return flow, pd.DataFrame(losses)
    
    def train_flow_antifragile_centered(self, flow, nominal_data: torch.Tensor, 
                                       target_data: torch.Tensor, context_dim: int = 1, 
                                       num_steps: int = 800, lr: float = 1e-3, 
                                       subsets: int = 5) -> Tuple[nn.Module, pd.DataFrame]:
        """Train a normalizing flow with antifragile-centered loss function."""
        
        # Move data to device
        nominal_data = nominal_data.to(self.device)
        target_data = target_data.to(self.device)
        flow = flow.to(self.device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
        
        # Track losses and metrics
        losses = []
        
        # Create K random subsets of target data
        target_subsets = self._create_target_subsets(target_data, subsets)
        
        # Training loop
        pbar = tqdm(range(num_steps))
        for step in pbar:
            optimizer.zero_grad()
            
            # Calculate antifragile-centered loss
            total_loss, loss_components = self.loss_calculator.antifragile_loss_function(
                flow, nominal_data, target_data, target_subsets, context_dim, step
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Record losses
            loss_record = {
                'step': step,
                'total_loss': total_loss.item(),
                **{k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_components.items()}
            }
            losses.append(loss_record)
            
            # Update progress bar
            # pbar.set_description(
            #     f"Step {step} | Total: {total_loss.item():.3f} | "
            #     f"Base: {loss_components['base_loss'].item():.3f} | "
            #     f"Volatility: {loss_components['volatility_gain'].item():.3f} | "
            #     f"Progressive: {loss_components['progressive_gain'].item():.3f}"
            # )
            pbar.set_description(
                f"Step {step} | Total: {total_loss.item():.3f} | "
                f"Strcl_loss: {loss_components['structural_loss'].item():.3f} | "
                f"Volatility: {loss_components['volatility_benefit'].item():.3f} | "
                f"antifragile_g: {loss_components['antifragile_gain'].item():.3f} | "
                f"fragility_h: {loss_components['fragility_heuristic'].item():.3f} "
                f"Cvxty_rspns: {loss_components['convex_response'].item():.3f}"
            )
        # return total_loss, {
        #     'structural_loss': structural_loss.item(),
        #     'left_tail_robustness': left_tail_robustness.item(),
        #     'volatility_benefit': volatility_benefit.item(),
        #     'convex_response': convex_response.item(),
        #     'fragility_heuristic': fragility_heuristic.item(),
        #     'antifragile_gain': antifragile_gain.item()
        # }
    
        # return total_loss, {
        #     'base_loss': base_loss,
        #     'volatility_gain': volatility_gain,
        #     'progressive_gain': progressive_gain,
        #     'tail_preparedness': tail_preparedness,
        #     'diversity_gain': diversity_gain,
        #     'stress_response': stress_response,
        #     'total_antifragile_gain': total_antifragile_gain
        # }
        
        return flow, pd.DataFrame(losses)
    
    def _create_target_subsets(self, target_data: torch.Tensor, subsets: int) -> List[torch.Tensor]:
        """Create K random subsets of target data."""
        target_subsets = []
        subset_size = len(target_data)
        for _ in range(subsets):
            # Sample with replacement
            indices = torch.randint(0, subset_size, (subset_size,), device=self.device)
            target_subsets.append(target_data[indices])
        return target_subsets
    
    def _calculate_training_loss(self, flow, nominal_data: torch.Tensor, target_data: torch.Tensor,
                                target_subsets: List[torch.Tensor], context_dim: int, step: int,
                                use_antifragile: bool, state_history: List[torch.Tensor],
                                omega: torch.Tensor, target_mean: torch.Tensor, k: torch.Tensor,
                                target_thresholds: Dict[str, float], adaptive_weights: Dict[str, float],
                                metrics_history: List[Dict]) -> Tuple[torch.Tensor, Dict]:
        """Calculate the complete training loss."""
        
        # 1. Loss for nominal data with context=0
        nominal_context = torch.zeros(len(nominal_data), context_dim, device=self.device)
        nominal_dist = flow(nominal_context)
        nominal_log_prob = nominal_dist.log_prob(nominal_data)
        nominal_loss = -torch.mean(nominal_log_prob)
        
        # 2. Loss for each target subset with different one-hot context
        target_losses = []
        subsets = len(target_subsets)
        for i, subset in enumerate(target_subsets):
            context = torch.zeros(len(subset), context_dim, device=self.device)
            context[:, 0] = (i+1) / subsets  # Evenly spaced between 0 and 1
            target_dist = flow(context)
            log_prob = target_dist.log_prob(subset)
            target_losses.append(-torch.mean(log_prob))
        
        target_loss = sum(target_losses) / len(target_losses)
        
        # 3. Loss for the full target dataset with calibrated context
        target_context = torch.ones(len(target_data), context_dim, device=self.device)
        target_full_dist = flow(target_context)
        target_full_log_prob = target_full_dist.log_prob(target_data)
        target_full_loss = -torch.mean(target_full_log_prob)
        
        # 4. Regularization between target subsets
        reg_loss = self._calculate_regularization_loss(flow, target_subsets, context_dim)
        
        # 5. Antifragile loss with adaptive weighting
        antifragile_loss_components = {
            'recovery_speed': torch.tensor(0.0, device=self.device),
            'volatility_response': torch.tensor(0.0, device=self.device),
            'outlier_robustness': torch.tensor(0.0, device=self.device),
            'interpolation_quality': torch.tensor(0.0, device=self.device)
        }
        
        antifragile_loss = torch.tensor(0.0, device=self.device)
        
        if use_antifragile:
            antifragile_loss, antifragile_loss_components = self._calculate_antifragile_loss(
                flow, nominal_data, target_data, step, state_history, omega, target_mean, k,
                target_thresholds, adaptive_weights, metrics_history, context_dim
            )
        
        # Total loss
        total_loss = nominal_loss + target_loss + target_full_loss + reg_loss + antifragile_loss
        
        loss_components = {
            'nominal_loss': nominal_loss,
            'target_loss': target_loss,
            'full_target_loss': target_full_loss,
            'reg_loss': reg_loss,
            'antifragile_loss': antifragile_loss,
            **{f'antifragile_{k}': v for k, v in antifragile_loss_components.items()}
        }
        
        return total_loss, loss_components
    
    def _calculate_regularization_loss(self, flow, target_subsets: List[torch.Tensor], 
                                      context_dim: int) -> torch.Tensor:
        """Calculate regularization loss between target subsets."""
        reg_loss = torch.tensor(0.0, device=self.device)
        subsets = len(target_subsets)
        
        for i in range(subsets):
            for j in range(i+1, subsets):
                if len(target_subsets[i]) > 0 and len(target_subsets[j]) > 0:
                    subset_i_context = torch.zeros(1, context_dim, device=self.device)
                    subset_i_context[0, 0] = (i+1) / subsets
                    
                    subset_j_context = torch.zeros(1, context_dim, device=self.device)
                    subset_j_context[0, 0] = (j+1) / subsets
                    
                    # Sample from each distribution
                    subset_i_dist = flow(subset_i_context)
                    subset_j_dist = flow(subset_j_context)
                    
                    subset_i_samples = subset_i_dist.sample((100,))
                    subset_j_samples = subset_j_dist.sample((100,))
                    
                    # Calculate distance between distributions
                    dist_ij = torch.mean(torch.norm(subset_i_samples.mean(0) - subset_j_samples.mean(0)))
                    reg_loss += dist_ij
        
        reg_loss = reg_loss / (subsets * (subsets - 1) / 2) * 0.1  # Normalize and scale
        return reg_loss
    
    def _calculate_antifragile_loss(self, flow, nominal_data: torch.Tensor, target_data: torch.Tensor,
                                   step: int, state_history: List[torch.Tensor], omega: torch.Tensor,
                                   target_mean: torch.Tensor, k: torch.Tensor,
                                   target_thresholds: Dict[str, float], adaptive_weights: Dict[str, float],
                                   metrics_history: List[Dict], context_dim: int) -> Tuple[torch.Tensor, Dict]:
        """Calculate the antifragile loss components."""
        
        # Gradual phase-in of antifragility
        antifragile_phase_in = min(1.0, step / 200.0)
        
        antifragile_loss_components = {
            'recovery_speed': torch.tensor(0.0, device=self.device),
            'volatility_response': torch.tensor(0.0, device=self.device),
            'outlier_robustness': torch.tensor(0.0, device=self.device),
            'interpolation_quality': torch.tensor(0.0, device=self.device)
        }
        
        with torch.no_grad():
            # Sample from nominal and target distributions
            nominal_context = torch.zeros(1, context_dim, device=self.device)
            target_context = torch.ones(1, context_dim, device=self.device)
            
            nominal_dist = flow(nominal_context)
            target_dist = flow(target_context)
            
            nominal_samples = nominal_dist.sample((20,))
            target_samples = target_dist.sample((20,))
            
            # Update state history
            current_state = 0.6 * nominal_samples.mean(0) + 0.4 * target_samples.mean(0)
            state_history.append(current_state.detach().clone())
            
            if len(state_history) > 15:
                state_history.pop(0)
        
        # Calculate antifragile components if we have enough history
        if len(state_history) >= 3:
            # Calculate first and second derivatives
            first_deriv, second_deriv = self.antifragility_calc.calculate_derivatives_smoothed(state_history)
            
            # Sample along interpolation path
            num_points = 7
            contexts = torch.linspace(0, 1, num_points, device=self.device).reshape(-1, 1)
            
            all_samples = []
            for ctx in contexts:
                ctx_expanded = ctx.expand(30, -1)
                dist = flow(ctx_expanded)
                samples = dist.sample((1,)).squeeze(0)
                all_samples.append(samples)
            
            samples = torch.cat(all_samples, dim=0)
            
            # Calculate redundancy factor
            redundancy = self.antifragility_calc.enhanced_redundancy_factor(samples, omega, k, target_mean)
            
            # Direction indicator
            target_direction = torch.sign(samples - omega.unsqueeze(0)) * torch.sign(target_mean - omega.unsqueeze(0))
            direction = torch.tanh(3.0 * torch.mean(target_direction, dim=1, keepdim=True))
            
            # Check for oscillations
            oscillation_factor = self.antifragility_calc.detect_oscillations(state_history)
            oscillation_tensor = torch.tensor(oscillation_factor, device=self.device)
            
            # Calculate current performance metrics
            current_metrics = self.adaptive_weights.calculate_performance_metrics(
                flow, nominal_dist, target_dist, first_deriv, second_deriv, oscillation_factor
            )
            
            # Update metrics history
            metrics_history.append(current_metrics)
            
            # Update target thresholds adaptively (moving average of best performance)
            if len(metrics_history) > 10:
                for metric in target_thresholds:
                    best_values = [metrics[metric] for metrics in metrics_history[-10:]]
                    target_thresholds[metric] = max(best_values) * 1.05  # Set target slightly above best
            
            # Calculate adaptive weights
            adaptive_weights.update(self.adaptive_weights.calculate_adaptive_weights(current_metrics, target_thresholds))
            
            # Calculate individual antifragile components
            
            # 1. Recovery speed component - penalizes slow recovery
            recovery_factor = torch.exp(-torch.norm(first_deriv))
            antifragile_loss_components['recovery_speed'] = -adaptive_weights['recovery_speed'] * \
                                                        torch.mean(redundancy * recovery_factor)
            
            # 2. Volatility response component - rewards appropriate response to volatility
            volatility_factor = torch.tanh(torch.norm(first_deriv)) * direction
            antifragile_loss_components['volatility_response'] = -adaptive_weights['volatility_response'] * \
                                                                torch.mean(redundancy * volatility_factor)
            
            # 3. Outlier robustness component - improves handling of outliers
            tail_samples = self.adaptive_weights.get_tail_samples(nominal_dist, target_dist, 20)
            if tail_samples is not None:
                outlier_ctx = torch.ones(len(tail_samples), context_dim, device=self.device) * 0.5  # Mid-context
                outlier_dist = flow(outlier_ctx)
                outlier_log_prob = outlier_dist.log_prob(tail_samples)
                antifragile_loss_components['outlier_robustness'] = -adaptive_weights['outlier_robustness'] * \
                                                                torch.mean(outlier_log_prob)
            
            # 4. Interpolation quality component - smoothness of interpolation
            interp_quality = torch.exp(-oscillation_tensor * 2.0) * torch.tanh(torch.norm(second_deriv))
            antifragile_loss_components['interpolation_quality'] = -adaptive_weights['interpolation_quality'] * \
                                                                torch.mean(redundancy * interp_quality)
            
            # Combine all components with phase-in
            antifragile_loss = antifragile_phase_in * sum(antifragile_loss_components.values())
        else:
            antifragile_loss = torch.tensor(0.0, device=self.device)
        
        return antifragile_loss, antifragile_loss_components

# Factory functions for backward compatibility
def train_flow(flow, nominal_data: torch.Tensor, target_data: torch.Tensor, 
               context_dim: int = 1, num_steps: int = 1000, lr: float = 1e-3, 
               subsets: int = 5, use_antifragile: bool = False, 
               device: str = "cpu") -> Tuple[torch.nn.Module, pd.DataFrame]:
    """Factory function for training flows."""
    trainer = FlowTrainer(device)
    return trainer.train_flow(flow, nominal_data, target_data, context_dim, 
                             num_steps, lr, subsets, use_antifragile)

def train_flow_antifragile_centered(flow, nominal_data: torch.Tensor, target_data: torch.Tensor, 
                                   context_dim: int = 1, num_steps: int = 800, lr: float = 1e-3, 
                                   subsets: int = 5, device: str = "cpu") -> Tuple[torch.nn.Module, pd.DataFrame]:
    """Factory function for antifragile-centered training."""
    trainer = FlowTrainer(device)
    return trainer.train_flow_antifragile_centered(flow, nominal_data, target_data, 
                                                  context_dim, num_steps, lr, subsets)