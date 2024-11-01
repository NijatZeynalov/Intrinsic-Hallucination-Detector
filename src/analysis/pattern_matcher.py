import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
from ..config.config import DetectorConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PatternMatcher:
    """
    Detects and analyzes recurring patterns in model states and attention
    that might indicate hallucination behaviors.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.pattern_buffer = defaultdict(list)
        self.known_patterns = {}
        self.match_history = []

        # Pattern matching parameters
        self.similarity_threshold = 0.85
        self.min_pattern_length = 3
        self.max_patterns = 1000

    def find_patterns(
            self,
            hidden_states: torch.Tensor,
            attention_weights: torch.Tensor,
            token_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, List[Dict]]:
        """
        Find recurring patterns in model states and attention.

        Args:
            hidden_states: Hidden state sequence [batch, seq_len, hidden_dim]
            attention_weights: Attention weights [batch, heads, seq_len, seq_len]
            token_ids: Optional token IDs for pattern context

        Returns:
            Dict containing detected patterns and their characteristics
        """
        try:
            # Extract state patterns
            state_patterns = self._find_state_patterns(hidden_states)

            # Extract attention patterns
            attention_patterns = self._find_attention_patterns(attention_weights)

            # Analyze pattern correlations
            pattern_correlations = self._analyze_pattern_correlations(
                state_patterns,
                attention_patterns
            )

            # Classify patterns
            classified_patterns = self._classify_patterns(
                state_patterns,
                attention_patterns,
                pattern_correlations
            )

            # Update pattern history
            self._update_pattern_history(classified_patterns, token_ids)

            return classified_patterns

        except Exception as e:
            logger.error(f"Error in pattern matching: {str(e)}")
            raise

    def _find_state_patterns(
            self,
            hidden_states: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """Find patterns in hidden states."""
        patterns = []
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Sliding window pattern detection
        for window_size in range(self.min_pattern_length, min(seq_len // 2, 10)):
            for start_idx in range(seq_len - window_size):
                window = hidden_states[:, start_idx:start_idx + window_size]

                # Check for similar patterns in history
                pattern_info = self._match_state_pattern(
                    window,
                    start_idx,
                    window_size
                )

                if pattern_info is not None:
                    patterns.append(pattern_info)

                if len(patterns) >= self.max_patterns:
                    return patterns

        return patterns

    def _find_attention_patterns(
            self,
            attention_weights: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """Find patterns in attention weights."""
        patterns = []
        batch_size, num_heads, seq_len, _ = attention_weights.shape

        # Analyze per-head attention patterns
        for head in range(num_heads):
            head_weights = attention_weights[:, head]

            # Find recurring attention patterns
            for window_size in range(self.min_pattern_length, min(seq_len // 2, 10)):
                for start_idx in range(seq_len - window_size):
                    window = head_weights[:, start_idx:start_idx + window_size]

                    pattern_info = self._match_attention_pattern(
                        window,
                        head,
                        start_idx,
                        window_size
                    )

                    if pattern_info is not None:
                        patterns.append(pattern_info)

                    if len(patterns) >= self.max_patterns:
                        return patterns

        return patterns

    def _match_state_pattern(
            self,
            window: torch.Tensor,
            start_idx: int,
            window_size: int
    ) -> Optional[Dict]:
        """Match state pattern against known patterns."""
        pattern_key = f"state_{window_size}"

        if pattern_key in self.known_patterns:
            for known_pattern in self.known_patterns[pattern_key]:
                similarity = F.cosine_similarity(
                    window.view(-1),
                    known_pattern['pattern'].view(-1),
                    dim=0
                )

                if similarity > self.similarity_threshold:
                    return {
                        'type': 'state',
                        'pattern': window,
                        'start_idx': start_idx,
                        'size': window_size,
                        'similarity': similarity,
                        'matches': known_pattern.get('matches', 0) + 1
                    }

        # Add new pattern
        new_pattern = {
            'type': 'state',
            'pattern': window,
            'start_idx': start_idx,
            'size': window_size,
            'similarity': 1.0,
            'matches': 1
        }

        if pattern_key not in self.known_patterns:
            self.known_patterns[pattern_key] = []
        self.known_patterns[pattern_key].append(new_pattern)

        return new_pattern

    def _match_attention_pattern(
            self,
            window: torch.Tensor,
            head_idx: int,
            start_idx: int,
            window_size: int
    ) -> Optional[Dict]:
        """Match attention pattern against known patterns."""
        pattern_key = f"attention_{head_idx}_{window_size}"

        if pattern_key in self.known_patterns:
            for known_pattern in self.known_patterns[pattern_key]:
                similarity = F.cosine_similarity(
                    window.view(-1),
                    known_pattern['pattern'].view(-1),
                    dim=0
                )

                if similarity > self.similarity_threshold:
                    return {
                        'type': 'attention',
                        'head_idx': head_idx,
                        'pattern': window,
                        'start_idx': start_idx,
                        'size': window_size,
                        'similarity': similarity,
                        'matches': known_pattern.get('matches', 0) + 1
                    }

        # Add new pattern
        new_pattern = {
            'type': 'attention',
            'head_idx': head_idx,
            'pattern': window,
            'start_idx': start_idx,
            'size': window_size,
            'similarity': 1.0,
            'matches': 1
        }

        if pattern_key not in self.known_patterns:
            self.known_patterns[pattern_key] = []
        self.known_patterns[pattern_key].append(new_pattern)

        return new_pattern

    def _analyze_pattern_correlations(
            self,
            state_patterns: List[Dict],
            attention_patterns: List[Dict]
    ) -> Dict[str, float]:
        """Analyze correlations between state and attention patterns."""
        correlations = {}

        if not state_patterns or not attention_patterns:
            return correlations

        # Analyze temporal correlations
        temporal_correlation = self._compute_temporal_correlation(
            state_patterns,
            attention_patterns
        )
        correlations['temporal'] = temporal_correlation

        # Analyze spatial correlations
        spatial_correlation = self._compute_spatial_correlation(
            state_patterns,
            attention_patterns
        )
        correlations['spatial'] = spatial_correlation

        return correlations

    def _classify_patterns(
            self,
            state_patterns: List[Dict],
            attention_patterns: List[Dict],
            correlations: Dict[str, float]
    ) -> Dict[str, List[Dict]]:
        """Classify detected patterns by type and significance."""
        classified = defaultdict(list)

        # Classify state patterns
        for pattern in state_patterns:
            pattern_type = self._determine_pattern_type(
                pattern,
                'state',
                correlations
            )
            classified[pattern_type].append(pattern)

        # Classify attention patterns
        for pattern in attention_patterns:
            pattern_type = self._determine_pattern_type(
                pattern,
                'attention',
                correlations
            )
            classified[pattern_type].append(pattern)

        return dict(classified)

    def _determine_pattern_type(
            self,
            pattern: Dict,
            pattern_source: str,
            correlations: Dict[str, float]
    ) -> str:
        """Determine the type of a pattern based on its characteristics."""
        # Score different pattern aspects
        recurrence_score = pattern.get('matches', 1) / self.max_patterns
        similarity_score = pattern.get('similarity', 0)
        correlation_score = np.mean(list(correlations.values()))

        # Combine scores
        total_score = (
                0.4 * recurrence_score +
                0.4 * similarity_score +
                0.2 * correlation_score
        )

        # Classify based on score
        if total_score > 0.8:
            return 'significant'
        elif total_score > 0.5:
            return 'moderate'
        else:
            return 'weak'

    def _compute_temporal_correlation(
            self,
            state_patterns: List[Dict],
            attention_patterns: List[Dict]
    ) -> float:
        """Compute temporal correlation between state and attention patterns."""
        if not state_patterns or not attention_patterns:
            return 0.0

        state_times = [p['start_idx'] for p in state_patterns]
        attention_times = [p['start_idx'] for p in attention_patterns]

        correlation = np.corrcoef(state_times, attention_times)[0, 1]
        return float(correlation if not np.isnan(correlation) else 0.0)

    def _compute_spatial_correlation(
            self,
            state_patterns: List[Dict],
            attention_patterns: List[Dict]
    ) -> float:
        """Compute spatial correlation between state and attention patterns."""
        if not state_patterns or not attention_patterns:
            return 0.0

        state_sizes = [p['size'] for p in state_patterns]
        attention_sizes = [p['size'] for p in attention_patterns]

        correlation = np.corrcoef(state_sizes, attention_sizes)[0, 1]
        return float(correlation if not np.isnan(correlation) else 0.0)

    def _update_pattern_history(
            self,
            patterns: Dict[str, List[Dict]],
            token_ids: Optional[torch.Tensor] = None
    ) -> None:
        """Update pattern history with new patterns."""
        # Add token context if available
        if token_ids is not None:
            for pattern_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    pattern['token_context'] = token_ids[
                                               :,
                                               pattern['start_idx']:pattern['start_idx'] + pattern['size']
                                               ]

        # Update match history
        self.match_history.append({
            'patterns': patterns,
            'timestamp': torch.cuda.Event(enable_timing=True)
            if torch.cuda.is_available() else None
        })

        # Keep history bounded
        if len(self.match_history) > 1000:
            self.match_history = self.match_history[-1000:]

    def get_pattern_statistics(self) -> Dict:
        """Get statistics about detected patterns."""
        stats = {
            'total_patterns': len(self.known_patterns),
            'pattern_types': defaultdict(int),
            'average_matches': defaultdict(float),
            'pattern_lengths': defaultdict(list)
        }

        for patterns in self.known_patterns.values():
            for pattern in patterns:
                pattern_type = pattern['type']
                stats['pattern_types'][pattern_type] += 1
                stats['average_matches'][pattern_type] += pattern.get('matches', 1)
                stats['pattern_lengths'][pattern_type].append(pattern['size'])

        # Compute averages
        for pattern_type in stats['average_matches']:
            if stats['pattern_types'][pattern_type] > 0:
                stats['average_matches'][pattern_type] /= stats['pattern_types'][pattern_type]

        return dict(stats)

    def reset(self) -> None:
        """Reset pattern matcher state."""
        self.pattern_buffer.clear()
        self.known_patterns.clear()
        self.match_history.clear()