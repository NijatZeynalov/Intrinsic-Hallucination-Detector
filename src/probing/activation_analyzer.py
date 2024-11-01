import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
from ..config.config import ProbeConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ActivationAnalyzer:
    """
    Analyzes model activations to detect patterns indicative of hallucinations.
    """

    def __init__(self, config: ProbeConfig):
        self.config = config
        self.activation_history = defaultdict(list)
        self.pattern_cache = {}
        self.baseline_stats = None

    def analyze_hidden_states(
            self,
            hidden_states: List[torch.Tensor],
            attention_weights: Optional[List[torch.Tensor]] = None,
            token_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze hidden states for hallucination patterns.

        Args:
            hidden_states: List of hidden states from each layer
            attention_weights: Optional attention weights
            token_ids: Optional token IDs for correlation

        Returns:
            Dict of analyzed features
        """
        try:
            # Extract relevant layers
            selected_states = [
                hidden_states[idx] for idx in self.config.probe_layers
            ]

            # Compute activation statistics
            activation_stats = self._compute_activation_statistics(selected_states)

            # Analyze attention patterns if available
            attention_features = {}
            if attention_weights is not None:
                attention_features = self._analyze_attention_patterns(attention_weights)

            # Detect anomalous patterns
            anomaly_scores = self._detect_anomalous_patterns(
                activation_stats,
                attention_features
            )

            # Compute token-level features if tokens provided
            token_features = {}
            if token_ids is not None:
                token_features = self._compute_token_features(
                    selected_states,
                    token_ids
                )

            return {
                "activation_stats": activation_stats,
                "attention_features": attention_features,
                "anomaly_scores": anomaly_scores,
                "token_features": token_features
            }

        except Exception as e:
            logger.error(f"Error in activation analysis: {str(e)}")
            raise

    def _compute_activation_statistics(
            self,
            hidden_states: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute statistical features from hidden states."""
        stats = {}

        for layer_idx, layer_states in enumerate(hidden_states):
            # Basic statistics
            stats[f"layer_{layer_idx}_mean"] = layer_states.mean(dim=-1)
            stats[f"layer_{layer_idx}_std"] = layer_states.std(dim=-1)

            # Activation magnitudes
            stats[f"layer_{layer_idx}_norm"] = torch.norm(
                layer_states,
                dim=-1
            )

            # Activation changes
            if layer_idx > 0:
                prev_states = hidden_states[layer_idx - 1]
                stats[f"layer_{layer_idx}_delta"] = torch.norm(
                    layer_states - prev_states,
                    dim=-1
                )

            # Distribution statistics
            stats[f"layer_{layer_idx}_skew"] = self._compute_skewness(layer_states)
            stats[f"layer_{layer_idx}_kurtosis"] = self._compute_kurtosis(layer_states)

        return stats

    def _analyze_attention_patterns(
            self,
            attention_weights: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Analyze attention weight patterns."""
        features = {}

        for layer_idx, attn in enumerate(attention_weights):
            # Attention entropy
            entropy = -torch.sum(
                attn * torch.log(attn + 1e-10),
                dim=-1
            )
            features[f"layer_{layer_idx}_attention_entropy"] = entropy

            # Attention concentration
            max_attention = torch.max(attn, dim=-1).values
            features[f"layer_{layer_idx}_attention_concentration"] = max_attention

            # Attention pattern changes
            if layer_idx > 0:
                prev_attn = attention_weights[layer_idx - 1]
                pattern_change = torch.norm(
                    attn - prev_attn,
                    dim=(-2, -1)
                )
                features[f"layer_{layer_idx}_attention_change"] = pattern_change

        return features

    def _detect_anomalous_patterns(
            self,
            activation_stats: Dict[str, torch.Tensor],
            attention_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Detect anomalous activation patterns."""
        anomaly_scores = {}

        # Initialize baseline if not exist
        if self.baseline_stats is None:
            self.baseline_stats = {
                k: v.mean().item()
                for k, v in activation_stats.items()
            }

        # Compute activation anomalies
        for key, value in activation_stats.items():
            baseline = self.baseline_stats[key]
            deviation = torch.abs(value - baseline)
            normalized_deviation = deviation / (baseline + 1e-8)
            anomaly_scores[f"{key}_anomaly"] = normalized_deviation

        # Combine with attention anomalies
        if attention_features:
            for key, value in attention_features.items():
                if key.endswith("_change"):
                    threshold = self.config.activation_threshold
                    anomaly_scores[f"{key}_anomaly"] = (value > threshold).float()

        return anomaly_scores

    def _compute_token_features(
            self,
            hidden_states: List[torch.Tensor],
            token_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute token-level features."""
        features = {}

        # Get embeddings for each token
        for layer_idx, states in enumerate(hidden_states):
            # Token representation consistency
            token_embeddings = states[torch.arange(states.size(0)), token_ids]

            # Compute similarity between consecutive tokens
            if token_ids.size(1) > 1:
                similarities = torch.cosine_similarity(
                    token_embeddings[:, :-1],
                    token_embeddings[:, 1:],
                    dim=-1
                )
                features[f"layer_{layer_idx}_token_similarity"] = similarities

            # Token representation stability
            if layer_idx > 0:
                prev_embeddings = hidden_states[layer_idx - 1][
                    torch.arange(states.size(0)),
                    token_ids
                ]
                stability = torch.cosine_similarity(
                    token_embeddings,
                    prev_embeddings,
                    dim=-1
                )
                features[f"layer_{layer_idx}_token_stability"] = stability

        return features

    @staticmethod
    def _compute_skewness(tensor: torch.Tensor) -> torch.Tensor:
        """Compute skewness of activation distribution."""
        mean = tensor.mean(dim=-1, keepdim=True)
        std = tensor.std(dim=-1, keepdim=True)
        skewness = torch.mean(
            ((tensor - mean) / (std + 1e-8)) ** 3,
            dim=-1
        )
        return skewness

    @staticmethod
    def _compute_kurtosis(tensor: torch.Tensor) -> torch.Tensor:
        """Compute kurtosis of activation distribution."""
        mean = tensor.mean(dim=-1, keepdim=True)
        std = tensor.std(dim=-1, keepdim=True)
        kurtosis = torch.mean(
            ((tensor - mean) / (std + 1e-8)) ** 4,
            dim=-1
        )
        return kurtosis

    def update_baseline_stats(
            self,
            activation_stats: Dict[str, torch.Tensor]
    ) -> None:
        """Update baseline statistics with new observations."""
        if self.baseline_stats is None:
            self.baseline_stats = {
                k: v.mean().item()
                for k, v in activation_stats.items()
            }
        else:
            alpha = 0.1  # Exponential moving average factor
            for k, v in activation_stats.items():
                current_mean = v.mean().item()
                self.baseline_stats[k] = (
                        (1 - alpha) * self.baseline_stats[k] +
                        alpha * current_mean
                )

    def reset_history(self) -> None:
        """Reset activation history and cache."""
        self.activation_history.clear()
        self.pattern_cache.clear()