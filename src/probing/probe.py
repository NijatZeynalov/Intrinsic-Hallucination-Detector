import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from transformers import PreTrainedModel
from ..config.config import ProbeConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class HallucationProbe(nn.Module):
    """
    Probing classifier for hallucination detection.
    """

    def __init__(
            self,
            config: ProbeConfig,
            input_dim: int
    ):
        super().__init__()
        self.config = config

        # Probe architecture
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        ])

        # Metrics tracking
        self.confidence_history = []
        self.detection_history = []

    def forward(
            self,
            hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass of the probe.

        Args:
            hidden_states: Model hidden states [batch_size, sequence_length, hidden_dim]

        Returns:
            Tuple of (confidence scores, detection metrics)
        """
        x = hidden_states

        # Pass through probe layers
        for layer in self.layers:
            x = layer(x)

        # Calculate confidence scores
        confidence_scores = x.squeeze(-1)

        # Calculate detection metrics
        metrics = self._calculate_metrics(confidence_scores)

        return confidence_scores, metrics

    def _calculate_metrics(
            self,
            confidence_scores: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate detection metrics from confidence scores."""
        metrics = {
            "mean_confidence": float(confidence_scores.mean()),
            "min_confidence": float(confidence_scores.min()),
            "max_confidence": float(confidence_scores.max()),
            "std_confidence": float(confidence_scores.std()),
            "detection_rate": float((confidence_scores < self.config.activation_threshold).float().mean())
        }

        return metrics


class ProbeTrainer:
    """
    Trainer for the hallucination probe.
    """

    def __init__(
            self,
            model: PreTrainedModel,
            probe: HallucationProbe,
            config: ProbeConfig
    ):
        self.model = model
        self.probe = probe
        self.config = config
        self.optimizer = torch.optim.Adam(
            probe.parameters(),
            lr=config.learning_rate
        )
        self.criterion = nn.BCELoss()

    def train_step(
            self,
            batch: Dict[str, torch.Tensor],
            labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Input batch
            labels: Ground truth labels

        Returns:
            Dict of training metrics
        """
        self.probe.train()
        self.optimizer.zero_grad()

        # Get model hidden states
        with torch.no_grad():
            outputs = self.model(**batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        # Select layers to probe
        selected_states = torch.stack([
            hidden_states[layer_idx]
            for layer_idx in self.config.probe_layers
        ], dim=1)

        # Forward pass through probe
        confidence_scores, metrics = self.probe(selected_states)

        # Calculate loss
        loss = self.criterion(confidence_scores, labels)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Update metrics
        metrics["loss"] = float(loss)
        metrics["accuracy"] = float(
            ((confidence_scores > 0.5) == labels).float().mean()
        )

        return metrics

    @torch.no_grad()
    def validate(
            self,
            val_dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Validate the probe.

        Args:
            val_dataloader: Validation data loader

        Returns:
            Dict of validation metrics
        """
        self.probe.eval()
        total_metrics = defaultdict(float)
        num_batches = 0

        for batch, labels in val_dataloader:
            # Get model hidden states
            outputs = self.model(**batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            # Select layers to probe
            selected_states = torch.stack([
                hidden_states[layer_idx]
                for layer_idx in self.config.probe_layers
            ], dim=1)

            # Forward pass through probe
            confidence_scores, batch_metrics = self.probe(selected_states)

            # Calculate loss
            loss = self.criterion(confidence_scores, labels)
            batch_metrics["loss"] = float(loss)

            # Update total metrics
            for k, v in batch_metrics.items():
                total_metrics[k] += v
            num_batches += 1

        # Calculate averages
        avg_metrics = {
            k: v / num_batches
            for k, v in total_metrics.items()
        }

        return avg_metrics

    def save_checkpoint(
            self,
            filepath: str,
            metrics: Optional[Dict] = None
    ) -> None:
        """Save probe checkpoint."""
        checkpoint = {
            "probe_state_dict": self.probe.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "metrics": metrics
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str) -> Dict:
        """Load probe checkpoint."""
        checkpoint = torch.load(filepath)

        self.probe.load_state_dict(checkpoint["probe_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint.get("metrics", {})