from dataclasses import dataclass, field
from typing import List, Optional, Dict
import os

@dataclass
class ModelConfig:
    """Configuration for the base language model."""
    model_name: str = "meta-llama/Llama-2-7b"
    device: str = "cuda"
    dtype: str = "float16"
    batch_size: int = 4
    max_length: int = 512

    def __post_init__(self):
        assert isinstance(self.model_name, str), "model_name must be a string"
        assert isinstance(self.device, str), "device must be a string"

@dataclass
class ProbeConfig:
    """Configuration for probing classifier."""
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-4
    probe_layers: List[int] = field(default_factory=lambda: [-1, -4, -8])
    activation_threshold: float = 0.7

    def __post_init__(self):
        assert isinstance(self.hidden_size, int), "hidden_size must be an integer"

@dataclass
class DetectorConfig:
    """Configuration for hallucination detection."""
    confidence_threshold: float = 0.8
    min_token_confidence: float = 0.6
    window_size: int = 5
    attention_threshold: float = 0.5
    error_types: List[str] = field(default_factory=lambda: [
        "factual_error",
        "reasoning_error",
        "consistency_error",
        "knowledge_gap"
    ])

@dataclass
class VisualizationConfig:
    """Configuration for visualization tools."""
    plot_attention: bool = True
    plot_activations: bool = True
    plot_confidence: bool = True
    save_plots: bool = True
    output_dir: str = "visualizations"
    color_map: str = "viridis"

@dataclass
class TrainingConfig:
    """Configuration for training settings."""
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    validation_interval: int = 500
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = False
