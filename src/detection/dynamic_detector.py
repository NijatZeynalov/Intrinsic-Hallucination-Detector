import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np
from ..config.config import DetectorConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DynamicHallucationDetector:
    """
    Real-time hallucination detector that analyzes model's internal states
    during generation to identify potential hallucinations.
    """

    def __init__(
            self,
            config: DetectorConfig,
            device: Optional[str] = None
    ):
        self.config = config
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize tracking buffers
        self.state_history = deque(maxlen=config.window_size)
        self.attention_history = deque(maxlen=config.window_size)
        self.confidence_history = deque(maxlen=config.window_size)

        # Thresholds for detection
        self.register_detection_thresholds()

        # Statistics tracking
        self.detection_stats = {
            'hallucinations_detected': 0,
            'total_tokens_processed': 0,
            'confidence_scores': [],
            'attention_patterns': []
        }

    def register_detection_thresholds(self):
        """Initialize adaptive thresholds for detection."""
        self.thresholds = {
            'base_confidence': self.config.confidence_threshold,
            'attention_entropy': 0.5,
            'state_consistency': 0.7,
            'semantic_drift': 0.3,
            'pattern_mismatch': 0.4
        }

        # Adaptive threshold adjustments
        self.threshold_history = {k: [] for k in self.thresholds}

    def detect(
            self,
            hidden_states: torch.Tensor,
            attention_matrix: torch.Tensor,
            token_ids: torch.Tensor,
            prev_states: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Detect hallucinations in the current generation step.

        Args:
            hidden_states: Model's hidden states [batch, seq_len, hidden_dim]
            attention_matrix: Attention weights [batch, heads, seq_len, seq_len]
            token_ids: Current token IDs [batch, seq_len]
            prev_states: Previous states for consistency checking

        Returns:
            Dict containing detection results and confidence scores
        """
        batch_size, seq_len = token_ids.shape

        try:
            # 1. Analyze current state patterns
            state_features = self._extract_state_features(hidden_states)

            # 2. Analyze attention patterns
            attention_features = self._analyze_attention_patterns(attention_matrix)

            # 3. Check semantic consistency
            semantic_scores = self._check_semantic_consistency(
                state_features,
                prev_states
            )

            # 4. Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                state_features,
                attention_features,
                semantic_scores
            )

            # 5. Detect anomalies
            hallucination_probs = self._detect_hallucinations(
                confidence_scores,
                state_features,
                attention_features
            )

            # 6. Update historical tracking
            self._update_history(
                state_features,
                attention_features,
                confidence_scores,
                hallucination_probs
            )

            # 7. Generate detailed results
            results = {
                'hallucination_probs': hallucination_probs,
                'confidence_scores': confidence_scores,
                'attention_scores': attention_features['attention_scores'],
                'semantic_consistency': semantic_scores,
                'state_features': state_features,
                'detection_map': self._generate_detection_map(
                    hallucination_probs,
                    confidence_scores
                )
            }

            # Update statistics
            self._update_statistics(results)

            return results

        except Exception as e:
            logger.error(f"Error in hallucination detection: {str(e)}")
            raise

    def _extract_state_features(
            self,
            hidden_states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract relevant features from hidden states."""
        # Calculate activation statistics
        mean_activation = torch.mean(hidden_states, dim=-1)
        std_activation = torch.std(hidden_states, dim=-1)

        # Calculate state dynamics
        if len(self.state_history) > 0:
            prev_states = self.state_history[-1]
            state_change = F.cosine_similarity(
                hidden_states.view(-1, hidden_states.size(-1)),
                prev_states.view(-1, prev_states.size(-1))
            )
        else:
            state_change = torch.ones(
                hidden_states.size(0),
                device=hidden_states.device
            )

        return {
            'mean_activation': mean_activation,
            'std_activation': std_activation,
            'state_change': state_change,
            'activation_norm': torch.norm(hidden_states, dim=-1),
            'hidden_states': hidden_states.detach()
        }

    def _analyze_attention_patterns(
            self,
            attention_matrix: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Analyze attention patterns for anomalies."""
        # Calculate attention entropy
        attention_probs = attention_matrix.mean(dim=1)  # Average over heads
        attention_entropy = -(
                attention_probs * torch.log(attention_probs + 1e-10)
        ).sum(-1)

        # Calculate attention concentration
        max_attention = attention_probs.max(dim=-1)[0]
        attention_spread = 1.0 - max_attention

        # Detect attention shifts
        if len(self.attention_history) > 0:
            prev_attention = self.attention_history[-1]
            attention_shift = F.cosine_similarity(
                attention_probs.view(-1, attention_probs.size(-1)),
                prev_attention.view(-1, prev_attention.size(-1))
            )
        else:
            attention_shift = torch.ones_like(attention_entropy)

        return {
            'attention_entropy': attention_entropy,
            'attention_spread': attention_spread,
            'attention_shift': attention_shift,
            'attention_scores': attention_probs,
            'max_attention': max_attention
        }

    def _check_semantic_consistency(
            self,
            current_features: Dict[str, torch.Tensor],
            prev_states: Optional[Dict] = None
    ) -> torch.Tensor:
        """Check semantic consistency with previous states."""
        if prev_states is None or len(self.state_history) == 0:
            return torch.ones(
                current_features['mean_activation'].size(0),
                device=self.device
            )

        # Calculate semantic similarity
        current_repr = current_features['hidden_states'].mean(dim=1)
        prev_repr = self.state_history[-1].mean(dim=1)

        semantic_similarity = F.cosine_similarity(
            current_repr,
            prev_repr
        )

        # Adjust for expected drift
        drift_factor = torch.exp(
            -torch.tensor(1 / self.config.window_size).to(self.device)
        )

        return semantic_similarity * drift_factor

    def _calculate_confidence_scores(
            self,
            state_features: Dict[str, torch.Tensor],
            attention_features: Dict[str, torch.Tensor],
            semantic_scores: torch.Tensor
    ) -> torch.Tensor:
        """Calculate confidence scores based on multiple features."""
        # Combine multiple signals
        confidence = (
                0.3 * self._normalize(state_features['activation_norm']) +
                0.2 * (1 - attention_features['attention_entropy']) +
                0.2 * attention_features['max_attention'] +
                0.3 * semantic_scores
        )

        return confidence

    def _detect_hallucinations(
            self,
            confidence_scores: torch.Tensor,
            state_features: Dict[str, torch.Tensor],
            attention_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Detect hallucinations based on combined signals."""
        # Calculate hallucination probability
        hallucination_probs = torch.zeros_like(confidence_scores)

        # Low confidence detection
        low_confidence_mask = confidence_scores < self.thresholds['base_confidence']

        # High entropy detection
        high_entropy_mask = (
                attention_features['attention_entropy'] >
                self.thresholds['attention_entropy']
        )

        # State inconsistency detection
        state_inconsistency = (
                state_features['state_change'] <
                self.thresholds['state_consistency']
        )

        # Combine detection signals
        hallucination_probs = (
                low_confidence_mask.float() * 0.4 +
                high_entropy_mask.float() * 0.3 +
                state_inconsistency.float() * 0.3
        )

        return hallucination_probs

    def _generate_detection_map(
            self,
            hallucination_probs: torch.Tensor,
            confidence_scores: torch.Tensor
    ) -> torch.Tensor:
        """Generate token-level detection map."""
        detection_map = torch.zeros_like(hallucination_probs)

        # Mark tokens as hallucinated based on thresholds
        detection_map[hallucination_probs > 0.5] = 1.0

        # Add confidence information
        detection_map = torch.stack([
            detection_map,
            confidence_scores,
            hallucination_probs
        ], dim=-1)

        return detection_map

    def _update_history(
            self,
            state_features: Dict[str, torch.Tensor],
            attention_features: Dict[str, torch.Tensor],
            confidence_scores: torch.Tensor,
            hallucination_probs: torch.Tensor
    ) -> None:
        """Update historical tracking."""
        self.state_history.append(state_features['hidden_states'])
        self.attention_history.append(attention_features['attention_scores'])
        self.confidence_history.append(confidence_scores)

        # Update adaptive thresholds
        self._update_thresholds(
            confidence_scores,
            hallucination_probs
        )

    def _update_thresholds(
            self,
            confidence_scores: torch.Tensor,
            hallucination_probs: torch.Tensor
    ) -> None:
        """Update adaptive thresholds based on recent history."""
        # Calculate threshold adjustments
        for k in self.thresholds:
            current_scores = {
                'base_confidence': confidence_scores.mean().item(),
                'attention_entropy': self.attention_history[-1].entropy().mean().item(),
                'state_consistency': self.state_history[-1].std().mean().item()
            }.get(k, 0.5)

            self.threshold_history[k].append(current_scores)

            # Adjust thresholds using exponential moving average
            if len(self.threshold_history[k]) > 10:
                alpha = 0.1
                self.thresholds[k] = (
                        (1 - alpha) * self.thresholds[k] +
                        alpha * np.mean(self.threshold_history[k][-10:])
                )

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [0, 1] range."""
        return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

    def get_detection_statistics(self) -> Dict:
        """Get detection statistics."""
        return {
            'total_hallucinations': self.detection_stats['hallucinations_detected'],
            'total_tokens': self.detection_stats['total_tokens_processed'],
            'hallucination_rate': (
                    self.detection_stats['hallucinations_detected'] /
                    max(1, self.detection_stats['total_tokens_processed'])
            ),
            'average_confidence': np.mean(self.detection_stats['confidence_scores']),
            'thresholds': self.thresholds
        }

    def reset(self) -> None:
        """Reset detector state."""
        self.state_history.clear()
        self.attention_history.clear()
        self.confidence_history.clear()
        self.register_detection_thresholds()
        self.detection_stats = {
            'hallucinations_detected': 0,
            'total_tokens_processed': 0,
            'confidence_scores': [],
            'attention_patterns': []
        }