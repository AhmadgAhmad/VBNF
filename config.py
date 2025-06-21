# config.py
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json
import torch

@dataclass
class DataConfig:
    """Configuration for data generation."""
    n_nominal: int = 1000
    n_target: int = 50
    noise_scale: float = 0.1
    horizontal_shift: float = 1.0
    
@dataclass
class TrainingConfig:
    """Configuration for training."""
    num_steps: int = 800
    lr: float = 1e-3
    subsets: int = 5
    context_dim: int = 1
    use_antifragile: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
@dataclass
class ModelConfig:
    """Configuration for the normalizing flow model."""
    features: int = 2
    context: int = 1
    transforms: int = 5
    hidden_features: tuple = (64, 64)
    
@dataclass
class EvaluationConfig:
    """Configuration for evaluation and stress testing."""
    batch_size: int = 200
    num_samples: int = 1000
    disruption_levels: int = 10
    alpha: float = 0.05  # For VaR/CVaR calculations
    
@dataclass
class ExperimentConfig:
    """Main configuration class."""
    name: str
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            name=config_dict['name'],
            data=DataConfig(**config_dict['data']),
            training=TrainingConfig(**config_dict['training']),
            model=ModelConfig(**config_dict['model']),
            evaluation=EvaluationConfig(**config_dict['evaluation'])
        )

def get_config(config_name: str) -> ExperimentConfig:
    """Get predefined configurations."""
    configs = {
        'banana': ExperimentConfig(
            name='banana',
            data=DataConfig(n_nominal=1000, n_target=50, horizontal_shift=1.0),
            training=TrainingConfig(num_steps=800, lr=1e-3, use_antifragile=True),
            model=ModelConfig(features=2, context=1, transforms=5),
            evaluation=EvaluationConfig(batch_size=200, num_samples=1000)
        ),
        'quick_test': ExperimentConfig(
            name='quick_test',
            data=DataConfig(n_nominal=500, n_target=25, horizontal_shift=1.0),
            training=TrainingConfig(num_steps=200, lr=1e-3, use_antifragile=True),
            model=ModelConfig(features=2, context=1, transforms=3),
            evaluation=EvaluationConfig(batch_size=100, num_samples=500)
        ),
        'stress_test': ExperimentConfig(
            name='stress_test',
            data=DataConfig(n_nominal=1000, n_target=5, horizontal_shift=1.0),
            training=TrainingConfig(num_steps=800, lr=1e-3, use_antifragile=True),
            model=ModelConfig(features=2, context=1, transforms=5),
            evaluation=EvaluationConfig(batch_size=200, num_samples=1000)
        )
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]