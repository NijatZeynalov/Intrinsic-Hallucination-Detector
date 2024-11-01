import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
from ..config.config import DetectorConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StateAnalyzer:
    """
    Analyzes model's internal states to detect patterns and anomalies
    that might indicate hallucinations.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.state_history = defaultdict(list)
        self.pattern_cache = {}
        self.statistics = {
            'state_changes': [],
            'pattern_matches': [],
            'anomaly_scores': []
        }

    def analyze_states(
            self,
            hidden_states: torch.Tensor,
            attention_patterns: torch.Tensor,
            layer_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze hidden states and attention patterns.

        Args:
            hidden_states: Model hidden states [batch, seq_len, hidden_dim]
            attention_patterns: Attention weights [batch, heads, seq_len, seq_len]
            layer_idx: Optional layer index

        Returns:
            Dict containing analysis results
        """
        try:
            batch_size, seq_len, hidden_dim = hidden_states.shape

            # Calculate state statistics
            state_stats = self._compute_state_statistics(hidden_states)

            # Analyze state dynamics
            dynamics = self._analyze_state_dynamics(hidden_states)

            # Analyze attention patterns
            attention_analysis = self._analyze_attention_patterns(attention_patterns)

            # Detect anomalies
            anomalies = self._detect_anomalies(
                state_stats,
                dynamics,
                attention_analysis
            )

            # Update history
            if layer_idx is not None:
                self._update_history(
                    layer_idx,
                    hidden_states,
                    state_stats,
                    anomalies
                )

            return {
                'state_statistics': state_stats,
                'dynamics': dynamics,
                'attention_analysis': attention_analysis,
                'anomalies': anomalies
            }

        except Exception as e:
            logger.error(f"Error in state analysis: {str(e)}")
            raise

    def _compute_state_statistics(
            self,
            hidden_states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute statistical measures of hidden states."""
        # Basic statistics
        mean_activation = torch.mean(hidden_states, dim=-1)
        std_activation = torch.std(hidden_states, dim=-1)

        # Compute activation norms
        activation_norms = torch.norm(hidden_states, dim=-1)

        # Compute state gradients
        if hidden_states.size(1) > 1:
            state_gradients = torch.norm(
                hidden_states[:, 1:] - hidden_states[:, :-1],
                dim=-1
            )
        else:
            state_gradients = torch.zeros_like(mean_activation)

        # Compute higher-order statistics
        skewness = self._compute_skewness(hidden_states)
        kurtosis = self._compute_kurtosis(hidden_states)

        return {
            'mean_activation': mean_activation,
            'std_activation': std_activation,
            'activation_norms': activation_norms,
            'state_gradients': state_gradients,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    def _analyze_state_dynamics(
            self,
            hidden_states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Analyze temporal dynamics of hidden states."""
        dynamics = {}

        # Compute state velocities
        if hidden_states.size(1) > 1:
            velocities = hidden_states[:, 1:] - hidden_states[:, :-1]
            dynamics['velocities'] = torch.norm(velocities, dim=-1)

            # Compute accelerations
            if hidden_states.size(1) > 2:
                accelerations = velocities[:, 1:] - velocities[:, :-1]
                dynamics['accelerations'] = torch.norm(accelerations, dim=-1)

        # Compute trajectory smoothness
        if hidden_states.size(1) > 2:
            smoothness = self._compute_trajectory_smoothness(hidden_states)
            dynamics['smoothness'] = smoothness

        # Compute state space coverage
        coverage = self._compute_state_coverage(hidden_states)
        dynamics['coverage'] = coverage

        return dynamics

    def _analyze_attention_patterns(
            self,
            attention_patterns: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Analyze attention pattern characteristics."""
        # Compute attention entropy
        attention_entropy = -(
                attention_patterns * torch.log(attention_patterns + 1e-10)
        ).sum(-1).mean(1)

        # Compute attention focus
        attention_focus = attention_patterns.max(dim=-1)[0].mean(1)

        # Compute attention pattern changes
        if attention_patterns.size(2) > 1:
            pattern_changes = F.cosine_similarity(
                attention_patterns[..., :-1, :],
                attention_patterns[..., 1:, :],
                dim=-1
            ).mean(1)
        else:
            pattern_changes = torch.ones_like(attention_entropy)

        return {
            'entropy': attention_entropy,
            'focus': attention_focus,
            'pattern_changes': pattern_changes
        }

    def _detect_anomalies(
            self,
            state_stats: Dict[str, torch.Tensor],
            dynamics: Dict[str, torch.Tensor],
            attention_analysis: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Detect anomalous patterns in states and attention."""
        anomalies = {}

        # Detect activation anomalies
        activation_anomalies = self._detect_activation_anomalies(
            state_stats['activation_norms'],
            state_stats['std_activation']
        )
        anomalies['activation_anomalies'] = activation_anomalies

        # Detect dynamic anomalies
        if 'velocities' in dynamics:
            dynamic_anomalies = self._detect_dynamic_anomalies(
                dynamics['velocities'],
                dynamics.get('accelerations', None)
            )
            anomalies['dynamic_anomalies'] = dynamic_anomalies

        # Detect attention anomalies
        attention_anomalies = self._detect_attention_anomalies(
            attention_analysis['entropy'],
            attention_analysis['focus']
        )
        anomalies['attention_anomalies'] = attention_anomalies

        # Compute overall anomaly score
        anomalies['overall_score'] = (
                0.4 * activation_anomalies +
                0.3 * anomalies.get('dynamic_anomalies', torch.zeros_like(activation_anomalies)) +
                0.3 * attention_anomalies
        )

        return anomalies

    def _compute_trajectory_smoothness(
            self,
            hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Compute smoothness of state trajectories."""
        velocities = hidden_states[:, 1:] - hidden_states[:, :-1]
        accelerations = velocities[:, 1:] - velocities[:, :-1]
        jerk = accelerations[:, 1:] - accelerations[:, :-1]

        smoothness = 1.0 / (torch.norm(jerk, dim=-1) + 1e-8)
        return smoothness.mean(dim=-1)

    def _compute_state_coverage(
            self,
            hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Compute coverage of state space."""
        # Project states to lower dimension for coverage calculation
        projected_states = hidden_states.mean(dim=1)

        # Compute pairwise distances
        distances = torch.cdist(projected_states, projected_states)

        # Calculate coverage as average minimum distance
        coverage = distances.min(dim=-1)[0].mean(dim=-1)
        return coverage

    def _detect_activation_anomalies(
            self,
            activation_norms: torch.Tensor,
            std_activation: torch.Tensor
    ) -> torch.Tensor:
        """Detect anomalies in activation patterns."""
        # Compute z-scores
        z_scores = (activation_norms - activation_norms.mean()) / (activation_norms.std() + 1e-8)

        # Combine with activation stability
        stability = 1.0 / (std_activation + 1e-8)

        # Compute anomaly scores
        anomaly_scores = torch.sigmoid(abs(z_scores)) * (1 - stability)
        return anomaly_scores

    def _detect_dynamic_anomalies(
            self,
            velocities: torch.Tensor,
            accelerations: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Detect anomalies in state dynamics."""
        # Compute velocity anomalies
        velocity_z_scores = (velocities - velocities.mean()) / (velocities.std() + 1e-8)

        if accelerations is not None:
            # Compute acceleration anomalies
            acceleration_z_scores = (
                                            accelerations - accelerations.mean()
                                    ) / (accelerations.std() + 1e-8)

            # Combine scores
            anomaly_scores = torch.sigmoid(
                0.6 * abs(velocity_z_scores) +
                0.4 * abs(acceleration_z_scores)
            )
        else:
            anomaly_scores = torch.sigmoid(abs(velocity_z_scores))

        return anomaly_scores

    def _detect_attention_anomalies(
            self,
            attention_entropy: torch.Tensor,
            attention_focus: torch.Tensor
    ) -> torch.Tensor:
        """Detect anomalies in attention patterns."""
        # High entropy and low focus indicate potential issues
        entropy_scores = attention_entropy / attention_entropy.max()
        focus_scores = 1 - attention_focus

        # Combine scores
        anomaly_scores = 0.5 * (entropy_scores + focus_scores)
        return anomaly_scores

    def _update_history(
            self,
            layer_idx: int,
            hidden_states: torch.Tensor,
            state_stats: Dict[str, torch.Tensor],
            anomalies: Dict[str, torch.Tensor]
    ) -> None:
        """Update state history for the layer."""
        # Keep limited history per layer
        max_history = 1000

        self.state_history[layer_idx].append({
            'hidden_states': hidden_states.detach().mean(dim=1),
            'stats': {k: v.detach() for k, v in state_stats.items()},
            'anomalies': {k: v.detach() for k, v in anomalies.items()}
        })

        if len(self.state_history[layer_idx]) > max_history:
            self.state_history[layer_idx] = self.state_history[layer_idx][-max_history:]

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

    def get_analysis_statistics(self) -> Dict:
        """Get summary statistics of state analysis."""
        stats = {}

        for layer_idx, history in self.state_history.items():
            layer_stats = {
                'avg_anomaly_score': np.mean([
                    h['anomalies']['overall_score'].mean().item()
                    for h in history
                ]),
                'max_anomaly_score': np.max([
                    h['anomalies']['overall_score'].max().item()
                    for h in history
                ]),
                'avg_activation_norm': np.mean([
                    h['stats']['activation_norms'].mean().item()
                    for h in history
                ])
            }
            stats[f'layer_{layer_idx}'] = layer_stats

        return stats

    def reset(self) -> None:
        """Reset analyzer state."""
        self.state_history.clear()
        self.pattern_cache.clear()
        self.statistics = {
            'state_changes': [],
            'pattern_matches': [],
            'anomaly_scores': []
        }