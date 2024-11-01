import torch
import numpy as np
from typing import Dict, List, Optional, Union
from collections import defaultdict
from datetime import datetime
from .logger import get_logger

logger = get_logger(__name__)


class MetricsTracker:
    """Track and analyze metrics for hallucination detection."""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.current_batch = {}
        self.history = []

    def update(
            self,
            metrics: Dict[str, Union[float, torch.Tensor]],
            category: Optional[str] = None
    ) -> None:
        """Update metrics with new values."""
        try:
            for name, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()

                key = f"{category}/{name}" if category else name
                self.metrics[key].append(value)
                self.current_batch[key] = value

        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")

    def compute_detection_metrics(
            self,
            predictions: torch.Tensor,
            labels: torch.Tensor,
            threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute detection metrics."""
        try:
            binary_preds = (predictions > threshold).float()

            tp = torch.sum((binary_preds == 1) & (labels == 1)).item()
            fp = torch.sum((binary_preds == 1) & (labels == 0)).item()
            fn = torch.sum((binary_preds == 0) & (labels == 1)).item()
            tn = torch.sum((binary_preds == 0) & (labels == 0)).item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            }

            self.update(metrics, 'detection')
            return metrics

        except Exception as e:
            logger.error(f"Error computing detection metrics: {str(e)}")
            return {}

    def compute_attention_metrics(
            self,
            attention_weights: torch.Tensor
    ) -> Dict[str, float]:
        """Compute attention pattern metrics."""
        try:
            entropy = -(attention_weights * torch.log(attention_weights + 1e-10)).sum(-1).mean()
            focus = attention_weights.max(dim=-1)[0].mean()
            coverage = (attention_weights > 0.1).float().mean()

            metrics = {
                'entropy': float(entropy),
                'focus': float(focus),
                'coverage': float(coverage)
            }

            self.update(metrics, 'attention')
            return metrics

        except Exception as e:
            logger.error(f"Error computing attention metrics: {str(e)}")
            return {}

    def get_metrics(self, window_size: Optional[int] = None) -> Dict[str, float]:
        """Get current metrics with optional window size."""
        metrics = {}

        for key, values in self.metrics.items():
            if not values:
                continue

            if window_size:
                values = values[-window_size:]
            metrics[key] = float(np.mean(values))

        return metrics

    def reset(self) -> None:
        """Reset metrics state."""
        self.history.append({
            'metrics': dict(self.metrics),
            'timestamp': datetime.now().isoformat()
        })
        self.metrics.clear()
        self.current_batch.clear()