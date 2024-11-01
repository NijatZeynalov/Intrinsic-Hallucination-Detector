import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
from ..config.config import DetectorConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ErrorClassifier:
    """
    Classifies different types of hallucination errors in language model outputs.
    Focuses on identifying specific patterns that indicate different types of errors.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.error_types = config.error_types

        # Initialize pattern detectors
        self.pattern_detectors = {
            'factual_error': FactualErrorDetector(),
            'reasoning_error': ReasoningErrorDetector(),
            'consistency_error': ConsistencyErrorDetector(),
            'knowledge_gap': KnowledgeGapDetector()
        }

        # Error statistics tracking
        self.error_stats = defaultdict(list)
        self.pattern_history = defaultdict(list)

    def classify(
            self,
            hidden_states: torch.Tensor,
            attention_weights: torch.Tensor,
            sequence_info: Optional[Dict] = None
    ) -> Dict[str, Dict]:
        """
        Classify potential hallucination errors.

        Args:
            hidden_states: Model hidden states [batch, seq_len, hidden_dim]
            attention_weights: Attention matrix [batch, heads, seq_len, seq_len]
            sequence_info: Additional sequence information

        Returns:
            Dict containing error classifications and confidence scores
        """
        try:
            batch_size, seq_len, _ = hidden_states.shape

            # Initialize results container
            error_classifications = {
                error_type: {
                    'probability': torch.zeros(batch_size, seq_len),
                    'confidence': torch.zeros(batch_size, seq_len),
                    'features': {}
                }
                for error_type in self.error_types
            }

            # Extract common features
            common_features = self._extract_common_features(
                hidden_states,
                attention_weights
            )

            # Classify each error type
            for error_type in self.error_types:
                detector = self.pattern_detectors[error_type]
                classification = detector.detect(
                    hidden_states,
                    attention_weights,
                    common_features,
                    sequence_info
                )

                error_classifications[error_type].update(classification)

            # Aggregate and normalize probabilities
            total_probs = sum(
                class_info['probability']
                for class_info in error_classifications.values()
            )

            for error_type in error_classifications:
                error_classifications[error_type]['probability'] /= (total_probs + 1e-8)

            # Update error statistics
            self._update_error_statistics(error_classifications)

            return error_classifications

        except Exception as e:
            logger.error(f"Error in error classification: {str(e)}")
            raise

    def _extract_common_features(
            self,
            hidden_states: torch.Tensor,
            attention_weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract common features used across different error types."""
        # Calculate activation patterns
        activation_mean = torch.mean(hidden_states, dim=-1)
        activation_std = torch.std(hidden_states, dim=-1)
        activation_norm = torch.norm(hidden_states, dim=-1)

        # Calculate attention patterns
        attention_entropy = -(
                attention_weights * torch.log(attention_weights + 1e-10)
        ).sum(-1).mean(1)

        max_attention = attention_weights.max(dim=-1)[0].mean(1)
        attention_dispersion = 1 - max_attention

        # Calculate state changes
        if hidden_states.size(1) > 1:
            state_changes = torch.norm(
                hidden_states[:, 1:] - hidden_states[:, :-1],
                dim=-1
            )
        else:
            state_changes = torch.zeros_like(activation_mean)

        return {
            'activation_mean': activation_mean,
            'activation_std': activation_std,
            'activation_norm': activation_norm,
            'attention_entropy': attention_entropy,
            'attention_dispersion': attention_dispersion,
            'state_changes': state_changes,
            'max_attention': max_attention
        }

    def _update_error_statistics(
            self,
            classifications: Dict[str, Dict]
    ) -> None:
        """Update error statistics with new classifications."""
        for error_type, info in classifications.items():
            self.error_stats[error_type].append({
                'mean_prob': float(info['probability'].mean()),
                'max_prob': float(info['probability'].max()),
                'confidence': float(info['confidence'].mean())
            })

            # Keep history bounded
            if len(self.error_stats[error_type]) > 1000:
                self.error_stats[error_type] = self.error_stats[error_type][-1000:]

    def get_error_statistics(self) -> Dict:
        """Get summary statistics for error classifications."""
        stats = {}

        for error_type in self.error_types:
            if not self.error_stats[error_type]:
                continue

            error_data = self.error_stats[error_type]
            stats[error_type] = {
                'mean_probability': np.mean([d['mean_prob'] for d in error_data]),
                'mean_confidence': np.mean([d['confidence'] for d in error_data]),
                'occurrence_rate': np.mean([
                    d['max_prob'] > 0.5 for d in error_data
                ])
            }

        return stats

    def reset(self) -> None:
        """Reset classifier state."""
        self.error_stats.clear()
        self.pattern_history.clear()
        for detector in self.pattern_detectors.values():
            detector.reset()


class BaseErrorDetector:
    """Base class for specific error type detectors."""

    def __init__(self):
        self.history = []
        self.feature_importance = {}

    def detect(
            self,
            hidden_states: torch.Tensor,
            attention_weights: torch.Tensor,
            common_features: Dict[str, torch.Tensor],
            sequence_info: Optional[Dict] = None
    ) -> Dict:
        """
        Detect specific error type.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset detector state."""
        self.history.clear()


class FactualErrorDetector(BaseErrorDetector):
    """Detects factual errors in generated content."""

    def detect(
            self,
            hidden_states: torch.Tensor,
            attention_weights: torch.Tensor,
            common_features: Dict[str, torch.Tensor],
            sequence_info: Optional[Dict] = None
    ) -> Dict:
        # High attention entropy often indicates factual confusion
        attention_scores = common_features['attention_entropy']

        # Unstable activations often indicate factual errors
        activation_stability = 1 - common_features['activation_std']

        # Calculate probability based on combined signals
        error_prob = (
                0.4 * attention_scores +
                0.6 * (1 - activation_stability)
        )

        # Calculate confidence based on feature consistency
        confidence = torch.exp(-common_features['state_changes'])

        return {
            'probability': error_prob,
            'confidence': confidence,
            'features': {
                'attention_entropy': attention_scores,
                'activation_stability': activation_stability
            }
        }


class ReasoningErrorDetector(BaseErrorDetector):
    """Detects errors in logical reasoning and deduction."""

    def detect(
            self,
            hidden_states: torch.Tensor,
            attention_weights: torch.Tensor,
            common_features: Dict[str, torch.Tensor],
            sequence_info: Optional[Dict] = None
    ) -> Dict:
        # Inconsistent attention patterns often indicate reasoning errors
        attention_consistency = 1 - common_features['attention_dispersion']

        # Large state changes can indicate reasoning jumps
        state_coherence = torch.exp(-common_features['state_changes'])

        # Calculate probability
        error_prob = (
                0.5 * (1 - attention_consistency) +
                0.5 * (1 - state_coherence)
        )

        # Confidence based on pattern stability
        confidence = (
                attention_consistency *
                common_features['activation_norm']
        )

        return {
            'probability': error_prob,
            'confidence': confidence,
            'features': {
                'attention_consistency': attention_consistency,
                'state_coherence': state_coherence
            }
        }


class ConsistencyErrorDetector(BaseErrorDetector):
    """Detects internal consistency errors in generated content."""

    def detect(
            self,
            hidden_states: torch.Tensor,
            attention_weights: torch.Tensor,
            common_features: Dict[str, torch.Tensor],
            sequence_info: Optional[Dict] = None
    ) -> Dict:
        # Check for attention to relevant context
        context_attention = common_features['max_attention']

        # Check for state consistency
        state_consistency = torch.exp(
            -torch.abs(
                common_features['activation_mean'].diff(dim=-1)
            )
        )

        # Calculate probability
        error_prob = (
                0.4 * (1 - context_attention) +
                0.6 * (1 - state_consistency)
        )

        # Confidence based on feature stability
        confidence = torch.exp(
            -common_features['activation_std']
        )

        return {
            'probability': error_prob,
            'confidence': confidence,
            'features': {
                'context_attention': context_attention,
                'state_consistency': state_consistency
            }
        }


class KnowledgeGapDetector(BaseErrorDetector):
    """Detects errors stemming from knowledge gaps."""

    def detect(
            self,
            hidden_states: torch.Tensor,
            attention_weights: torch.Tensor,
            common_features: Dict[str, torch.Tensor],
            sequence_info: Optional[Dict] = None
    ) -> Dict:
        # Low activation norms often indicate knowledge gaps
        knowledge_confidence = torch.sigmoid(
            common_features['activation_norm']
        )

        # Scattered attention can indicate searching for information
        attention_focus = 1 - common_features['attention_entropy']

        # Calculate probability
        error_prob = (
                0.7 * (1 - knowledge_confidence) +
                0.3 * (1 - attention_focus)
        )

        # Confidence based on consistency
        confidence = torch.exp(
            -common_features['activation_std']
        )

        return {
            'probability': error_prob,
            'confidence': confidence,
            'features': {
                'knowledge_confidence': knowledge_confidence,
                'attention_focus': attention_focus
            }
        }