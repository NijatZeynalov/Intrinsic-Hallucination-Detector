import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
from ..config.config import VisualizationConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StateVisualizer:
    """
    Visualization tools for analyzing and displaying model internal states,
    hallucination patterns, and confidence scores.
    """

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style configurations
        plt.style.use('seaborn')
        self.color_palette = sns.color_palette("husl", 8)
        self.error_colors = {
            'factual_error': '#FF6B6B',
            'reasoning_error': '#4ECDC4',
            'consistency_error': '#45B7D1',
            'knowledge_gap': '#96CEB4'
        }

    def visualize_attention_patterns(
            self,
            attention_weights: torch.Tensor,
            token_labels: List[str],
            hallucination_scores: Optional[torch.Tensor] = None,
            layer_idx: Optional[int] = None,
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize attention patterns with optional hallucination highlighting.

        Args:
            attention_weights: Attention matrix [batch, heads, seq_len, seq_len]
            token_labels: Token text labels
            hallucination_scores: Optional hallucination probabilities
            layer_idx: Optional layer index for multi-layer visualization
            save_path: Optional path to save visualization
        """
        try:
            # Create figure
            fig = plt.figure(figsize=(15, 10))

            # Average attention across heads
            avg_attention = attention_weights.mean(dim=1).cpu().numpy()

            # Create heatmap with custom colormap
            attention_plot = sns.heatmap(
                avg_attention[0],
                cmap='YlOrRd',
                xticklabels=token_labels,
                yticklabels=token_labels,
                cbar_kws={'label': 'Attention Weight'}
            )

            # Highlight hallucinated tokens if provided
            if hallucination_scores is not None:
                scores = hallucination_scores.cpu().numpy()
                for i, score in enumerate(scores[0]):
                    if score > 0.5:  # Highlight high probability hallucinations
                        plt.axvline(x=i, color='red', alpha=0.3)
                        plt.axhline(y=i, color='red', alpha=0.3)

            # Add titles and labels
            layer_text = f" (Layer {layer_idx})" if layer_idx is not None else ""
            plt.title(f"Attention Patterns{layer_text}")
            plt.xlabel("Target Tokens")
            plt.ylabel("Source Tokens")

            # Rotate token labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)

            # Save if path provided
            if save_path and self.config.save_plots:
                plt.savefig(
                    self.output_dir / save_path,
                    bbox_inches='tight',
                    dpi=300
                )

            plt.close()

        except Exception as e:
            logger.error(f"Error in attention visualization: {str(e)}")

    def plot_hallucination_analysis(
            self,
            error_classifications: Dict[str, Dict],
            token_labels: List[str],
            confidence_scores: torch.Tensor,
            save_path: Optional[str] = None
    ) -> None:
        """
        Create comprehensive hallucination analysis visualization.

        Args:
            error_classifications: Error classification results
            token_labels: Token text labels
            confidence_scores: Model confidence scores
            save_path: Optional path to save visualization
        """
        try:
            # Create subplot figure
            fig = make_subplots(
                rows=3,
                cols=1,
                subplot_titles=(
                    'Error Type Probabilities',
                    'Confidence Scores',
                    'Error Distribution'
                ),
                vertical_spacing=0.15
            )

            # Plot error probabilities
            token_positions = np.arange(len(token_labels))
            for error_type, data in error_classifications.items():
                fig.add_trace(
                    go.Scatter(
                        x=token_positions,
                        y=data['probability'].cpu().numpy()[0],
                        name=error_type,
                        line=dict(color=self.error_colors[error_type])
                    ),
                    row=1,
                    col=1
                )

            # Plot confidence scores
            fig.add_trace(
                go.Scatter(
                    x=token_positions,
                    y=confidence_scores.cpu().numpy()[0],
                    name='Confidence',
                    line=dict(color='black')
                ),
                row=2,
                col=1
            )

            # Plot error distribution
            error_dist = {
                error_type: data['probability'].mean().item()
                for error_type, data in error_classifications.items()
            }
            fig.add_trace(
                go.Bar(
                    x=list(error_dist.keys()),
                    y=list(error_dist.values()),
                    marker_color=list(self.error_colors.values())
                ),
                row=3,
                col=1
            )

            # Update layout
            fig.update_layout(
                height=900,
                showlegend=True,
                title_text="Hallucination Analysis"
            )

            # Update axes
            fig.update_xaxes(
                ticktext=token_labels,
                tickvals=token_positions,
                tickangle=45,
                row=1,
                col=1
            )
            fig.update_xaxes(
                ticktext=token_labels,
                tickvals=token_positions,
                tickangle=45,
                row=2,
                col=1
            )

            # Save if path provided
            if save_path and self.config.save_plots:
                fig.write_html(self.output_dir / save_path)

        except Exception as e:
            logger.error(f"Error in hallucination analysis visualization: {str(e)}")

    def visualize_state_dynamics(
            self,
            hidden_states: torch.Tensor,
            attention_patterns: torch.Tensor,
            error_probs: Dict[str, torch.Tensor],
            save_path: Optional[str] = None
    ) -> None:
        """
        Visualize dynamics of model internal states.

        Args:
            hidden_states: Model hidden states
            attention_patterns: Attention patterns
            error_probs: Error probabilities
            save_path: Optional path to save visualization
        """
        try:
            # Create subplot figure
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    'Hidden State Dynamics',
                    'Attention Pattern Evolution',
                    'Error Probability Dynamics',
                    'State-Error Correlation'
                )
            )

            # Plot hidden state dynamics
            state_norms = torch.norm(hidden_states, dim=-1).cpu().numpy()[0]
            fig.add_trace(
                go.Heatmap(
                    z=state_norms,
                    colorscale='Viridis',
                    name='Hidden States'
                ),
                row=1,
                col=1
            )

            # Plot attention pattern evolution
            attention_avg = attention_patterns.mean(dim=1).cpu().numpy()[0]
            fig.add_trace(
                go.Heatmap(
                    z=attention_avg,
                    colorscale='Reds',
                    name='Attention'
                ),
                row=1,
                col=2
            )

            # Plot error probability dynamics
            for error_type, probs in error_probs.items():
                fig.add_trace(
                    go.Scatter(
                        y=probs.cpu().numpy()[0],
                        name=error_type,
                        line=dict(color=self.error_colors[error_type])
                    ),
                    row=2,
                    col=1
                )

            # Plot state-error correlation
            correlations = self._compute_state_error_correlations(
                hidden_states,
                error_probs
            )
            fig.add_trace(
                go.Heatmap(
                    z=correlations,
                    colorscale='RdBu',
                    name='Correlations'
                ),
                row=2,
                col=2
            )

            # Update layout
            fig.update_layout(
                height=800,
                width=1200,
                title_text="Model State Dynamics Analysis"
            )

            # Save if path provided
            if save_path and self.config.save_plots:
                fig.write_html(self.output_dir / save_path)

        except Exception as e:
            logger.error(f"Error in state dynamics visualization: {str(e)}")

    def plot_confidence_evolution(
            self,
            confidence_history: List[torch.Tensor],
            error_history: List[Dict[str, float]],
            window_size: int = 100,
            save_path: Optional[str] = None
    ) -> None:
        """
        Plot evolution of confidence scores and error rates.

        Args:
            confidence_history: History of confidence scores
            error_history: History of error probabilities
            window_size: Window size for moving averages
            save_path: Optional path to save visualization
        """
        try:
            # Create figure
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=(
                    'Confidence Score Evolution',
                    'Error Rate Trends'
                )
            )

            # Plot confidence evolution
            confidence_scores = torch.stack(confidence_history).cpu().numpy()
            window = min(window_size, len(confidence_scores))
            moving_avg = np.convolve(
                confidence_scores.mean(axis=1),
                np.ones(window) / window,
                mode='valid'
            )

            fig.add_trace(
                go.Scatter(
                    y=moving_avg,
                    name='Average Confidence',
                    line=dict(color='black')
                ),
                row=1,
                col=1
            )

            # Plot error rate trends
            for error_type in self.error_colors:
                error_rates = [h.get(error_type, 0) for h in error_history]
                moving_avg = np.convolve(
                    error_rates,
                    np.ones(window) / window,
                    mode='valid'
                )

                fig.add_trace(
                    go.Scatter(
                        y=moving_avg,
                        name=f"{error_type} Rate",
                        line=dict(color=self.error_colors[error_type])
                    ),
                    row=2,
                    col=1
                )

            # Update layout
            fig.update_layout(
                height=600,
                showlegend=True,
                title_text="Confidence and Error Rate Evolution"
            )

            # Save if path provided
            if save_path and self.config.save_plots:
                fig.write_html(self.output_dir / save_path)

        except Exception as e:
            logger.error(f"Error in confidence evolution visualization: {str(e)}")

    def _compute_state_error_correlations(
            self,
            hidden_states: torch.Tensor,
            error_probs: Dict[str, torch.Tensor]
    ) -> np.ndarray:
        """Compute correlations between hidden states and error probabilities."""
        # Flatten hidden states
        states_flat = hidden_states[0].cpu().numpy()

        # Stack error probabilities
        error_stack = np.stack([
            probs.cpu().numpy()[0] for probs in error_probs.values()
        ])

        # Compute correlation matrix
        correlations = np.corrcoef(states_flat.T, error_stack)
        return correlations[:states_flat.shape[1], -len(error_probs):]

    def create_summary_dashboard(
            self,
            analysis_results: Dict,
            save_path: Optional[str] = None
    ) -> None:
        """
        Create comprehensive analysis dashboard.

        Args:
            analysis_results: Complete analysis results
            save_path: Optional path to save dashboard
        """
        try:
            # Create dashboard figure
            fig = make_subplots(
                rows=3,
                cols=2,
                subplot_titles=(
                    'Overall Error Distribution',
                    'Confidence vs Error Rates',
                    'Token-level Analysis',
                    'State Dynamics',
                    'Error Type Correlations',
                    'Temporal Patterns'
                ),
                specs=[
                    [{"type": "pie"}, {"type": "scatter"}],
                    [{"type": "heatmap"}, {"type": "heatmap"}],
                    [{"type": "scatter"}, {"type": "heatmap"}]
                ]
            )

            # Add plots
            self._add_error_distribution(fig, analysis_results, 1, 1)
            self._add_confidence_analysis(fig, analysis_results, 1, 2)
            self._add_token_analysis(fig, analysis_results, 2, 1)
            self._add_state_dynamics(fig, analysis_results, 2, 2)
            self._add_error_correlations(fig, analysis_results, 3, 1)
            self._add_temporal_patterns(fig, analysis_results, 3, 2)

            # Update layout
            fig.update_layout(
                height=1200,
                width=1600,
                showlegend=True,
                title_text="Hallucination Analysis Dashboard"
            )

            # Save if path provided
            if save_path and self.config.save_plots:
                fig.write_html(self.output_dir / save_path)

        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")

    def _add_error_distribution(
            self,
            fig: go.Figure,
            results: Dict,
            row: int,
            col: int
    ) -> None:
        """Add error distribution pie chart."""
        error_dist = results.get('error_distribution', {})
        fig.add_trace(
            go.Pie(
                labels=list(error_dist.keys()),
                values=list(error_dist.values()),
                marker_colors=list(self.error_colors.values())
            ),
            row=row,
            col=col
        )


