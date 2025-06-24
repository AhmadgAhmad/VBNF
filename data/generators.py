# antifragile_vbnf/data/generators.py
import numpy as np
import pandas as pd
import torch
from typing import Dict, Tuple, List

class DataGenerator:
    """Base class for data generation."""
    
    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def generate(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Generate data. To be implemented by subclasses."""
        raise NotImplementedError

class BananaDataGenerator(DataGenerator):
    """Generator for banana-shaped datasets."""
    
    def generate_semicircle(self, center: List[float], radius: float, 
                          num_points: int, angle_range: List[float], 
                          noise_scale: float = 0.1) -> np.ndarray:
        """Generate points on a semicircle with noise."""
        angles = np.linspace(angle_range[0], angle_range[1], num_points)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        
        # Add noise
        x += np.random.normal(0, noise_scale, num_points)
        y += np.random.normal(0, noise_scale, num_points)
        
        return np.column_stack((x, y))
    
    def generate(self, n_nominal: int = 1000, n_target: int = 50, 
                noise_scale: float = 0.1, horizontal_shift: float = 1.0) -> Dict[str, torch.Tensor]:
        """Generate a banana-shaped dataset with imbalanced classes and horizontal shift."""
        # Nominal data (upper semicircle)
        nominal_center = [0, 0]
        nominal_radius = 1.0
        nominal_angles = [0, np.pi]
        nominal_data = self.generate_semicircle(nominal_center, nominal_radius, n_nominal, 
                                               nominal_angles, noise_scale)
        
        # Target data (lower semicircle) - shifted horizontally
        target_center = [horizontal_shift, 0]  # Shift the center horizontally
        target_radius = 1.0
        target_angles = [np.pi, 2*np.pi]
        target_data = self.generate_semicircle(target_center, target_radius, n_target, 
                                             target_angles, noise_scale)
        
        # Combine data with labels
        nominal_df = pd.DataFrame(nominal_data, columns=['x', 'y'])
        nominal_df['type'] = 'nominal'
        
        target_df = pd.DataFrame(target_data, columns=['x', 'y'])
        target_df['type'] = 'target'
        
        combined_df = pd.concat([nominal_df, target_df], ignore_index=True)
        
        # Convert to torch tensors
        nominal_tensor = torch.tensor(nominal_data, dtype=torch.float32)
        target_tensor = torch.tensor(target_data, dtype=torch.float32)
        
        return {
            'df': combined_df,
            'nominal': nominal_tensor,
            'target': target_tensor
        }

class SyntheticDataGenerator(DataGenerator):
    """Generator for various synthetic datasets for stress testing."""
    
    def generate_shifted_data(self, base_data: torch.Tensor, shift_magnitude: float) -> torch.Tensor:
        """Generate horizontally shifted version of data."""
        shifted_data = base_data.clone()
        shifted_data[:, 0] += shift_magnitude
        return shifted_data
    
    def generate_noisy_data(self, base_data: torch.Tensor, noise_scale: float) -> torch.Tensor:
        """Generate noisier version of data."""
        noise = torch.randn_like(base_data) * noise_scale
        return base_data + noise
    
    def generate_deformed_data(self, base_data: torch.Tensor, deformation_ratio: float) -> torch.Tensor:
        """Generate shape-deformed version of data."""
        deformed_data = base_data.clone()
        deformed_data[:, 1] *= deformation_ratio  # Compress/stretch y-axis
        return deformed_data
    
    def generate_outliers(self, num_outliers: int, magnitude: float, 
                         reference_point: torch.Tensor = None) -> torch.Tensor:
        """Generate extreme outlier points."""
        if reference_point is None:
            reference_point = torch.zeros(2)
        
        outliers = torch.randn(num_outliers, 2) * 0.2 + reference_point + magnitude
        return outliers
    
    def generate_interpolated_data(self, data1: torch.Tensor, data2: torch.Tensor, 
                                 interpolation_factor: float) -> torch.Tensor:
        """Generate interpolated data between two datasets."""
        return data1 * (1 - interpolation_factor) + data2 * interpolation_factor

# Factory function for easy access
def generate_banana_dataset(n_nominal: int = 1000, n_target: int = 50, 
                           noise_scale: float = 0.1, horizontal_shift: float = 1.0,
                           seed: int = None) -> Dict[str, torch.Tensor]:
    """Factory function to generate banana dataset."""
    generator = BananaDataGenerator(seed=seed)
    return generator.generate(n_nominal, n_target, noise_scale, horizontal_shift)

def generate_semicircle(center: List[float], radius: float, num_points: int, 
                       angle_range: List[float], noise_scale: float = 0.1) -> np.ndarray:
    """Factory function to generate semicircle data."""
    generator = BananaDataGenerator()
    return generator.generate_semicircle(center, radius, num_points, angle_range, noise_scale)