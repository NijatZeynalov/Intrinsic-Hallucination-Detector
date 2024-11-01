import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from ..config.config import VisualizationConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ActivationPlotter:
    """
    Visualization tools for model activations and hallucination patterns.
    Provides detailed visualizations of neural network activation patterns,
    attention mechanisms, and hallucination detection results.
    """

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting styles
        plt.style.use('seaborn')
        self.color_scheme = {
            'activation': 'viridis',
            'attention': 'YlOrRd',
            'hallucination': 'RdYlBu_r',
            'confidence': 'PuBu'
        }

        # Initialize plotly template
        self.plotly_template = 'plotly_white'

    def plot_layer_activations(
            self,
            activations: torch.Tensor,
            layer_idx: int,
            token_labels: Optional[List[str]] = None,
            hallucination_scores: Optional[torch.Tensor] = None,
            save_path: Optional[str] = None
    ) -> None:
        """
        Plot neural network layer activations with optional hallucination highlighting.

        Args:
            activations: Layer activation tensor [batch, seq_len, hidden_dim]
            layer_idx: Layer index
            token_labels: Optional token text labels
            hallucination_scores: Optional hallucination probabilities
            save_path: Optional path to save visualization
        """
        try:
            fig = plt.figure(figsize=(15, 8))

            # Plot main activation heatmap
            activation_map = activations[0].detach().cpu().numpy()

            ax = sns.heatmap(
                activation_map,
                cmap=self.color_scheme['activation'],
                center=0,
                cbar_kws={'label': 'Activation Value'}
            )

            # Add token labels if provided
            if token_labels:
                plt.xticks(
                    range(len(token_labels)),
                    token_labels,
                    rotation=45,
                    ha='right'
                )

            # Highlight hallucinations if scores provided
            if hallucination_scores is not None:
                self._add_hallucination_highlights(
                    ax,
                    hallucination_scores[0].cpu().numpy()
                )

            plt.title(f'Layer {layer_idx} Activations')
            plt.xlabel('Token Position')
            plt.ylabel('Hidden Dimension')

            if save_path and self.config.save_plots:
                plt.savefig(
                    self.output_dir / save_path,
                    bbox_inches='tight',
                    dpi=300
                )

            plt.close()

        except Exception as e:
            logger.error(f"Error plotting layer activations: {str(e)}")

    def plot_activation_dynamics(
            self,
            activation_history: List[torch.Tensor],
            tokens: List[str],
            save_path: Optional[str] = None
    ) -> None:
        """
        Plot activation dynamics over time/tokens.

        Args:
            activation_history: List of activation tensors
            tokens: Token labels
            save_path: Optional save path
        """
        try:
            # Create interactive plot
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    'Activation Magnitude Over Time',
                    'Activation Distribution',
                    'Token-wise Activation Pattern',
                    'Activation Gradient'
                )
            )

            # Plot activation magnitude
            magnitudes = [
                torch.norm(act, dim=-1).mean().item()
                for act in activation_history
            ]
            fig.add_trace(
                go.Scatter(
                    y=magnitudes,
                    mode='lines+markers',
                    name='Activation Magnitude'
                ),
                row=1,
                col=1
            )

            # Plot activation distribution
            activations = torch.cat(activation_history, dim=1)
            fig.add_trace(
                go.Histogram(
                    x=activations.flatten().cpu().numpy(),
                    nbinsx=50,
                    name='Activation Distribution'
                ),
                row=1,
                col=2
            )

            # Plot token-wise pattern
            token_activations = activations[0].mean(dim=-1).cpu().numpy()
            fig.add_trace(
                go.Heatmap(
                    z=[token_activations],
                    x=tokens,
                    colorscale=self.color_scheme['activation'],
                    name='Token Activations'
                ),
                row=2,
                col=1
            )

            # Plot activation gradient
            if len(activation_history) > 1:
                gradients = torch.cat([
                    (a[0, 1:] - a[0, :-1]).mean(dim=-1)
                    for a in activation_history
                ]).cpu().numpy()

                fig.add_trace(
                    go.Scatter(
                        y=gradients,
                        mode='lines',
                        name='Activation Gradient'
                    ),
                    row=2,
                    col=2
                )

            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                template=self.plotly_template
            )

            if save_path and self.config.save_plots:
                fig.write_html(self.output_dir / save_path)

        except Exception as e:
            logger.error(f"Error plotting activation dynamics: {str(e)}")

    def plot_comparative_analysis(
            self,
            normal_activations: torch.Tensor,
            hallucinated_activations: torch.Tensor,
            token_labels: List[str],
            save_path: Optional[str] = None
    ) -> None:
        """
        Plot comparative analysis of normal vs hallucinated activations.

        Args:
            normal_activations: Activations from normal generation
            hallucinated_activations: Activations from hallucinated generation
            token_labels: Token labels
            save_path: Optional save path
        """
        try:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    'Normal Activations',
                    'Hallucinated Activations',
                    'Activation Difference',
                    'Statistical Comparison'
                ),
                vertical_spacing=0.15
            )

            # Plot normal activations
            fig.add_trace(
                go.Heatmap(
                    z=normal_activations[0].cpu().numpy(),
                    colorscale=self.color_scheme['activation'],
                    name='Normal'
                ),
                row=1,
                col=1
            )

            # Plot hallucinated activations
            fig.add_trace(
                go.Heatmap(
                    z=hallucinated_activations[0].cpu().numpy(),
                    colorscale=self.color_scheme['activation'],
                    name='Hallucinated'
                ),
                row=1,
                col=2
            )

            # Plot difference
            diff = (
                    hallucinated_activations[0] - normal_activations[0]
            ).cpu().numpy()
            fig.add_trace(
                go.Heatmap(
                    z=diff,
                    colorscale=self.color_scheme['hallucination'],
                    name='Difference'
                ),
                row=2,
                col=1
            )

            # Statistical comparison
            normal_stats = normal_activations[0].mean(dim=-1).cpu().numpy()
            hallucinated_stats = hallucinated_activations[0].mean(dim=-1).cpu().numpy()

            fig.add_trace(
                go.Scatter(
                    x=token_labels,
                    y=normal_stats,
                    name='Normal Mean',
                    line=dict(color='blue')
                ),
                row=2,
                col=2
            )

            fig.add_trace(
                go.Scatter(
                    x=token_labels,
                    y=hallucinated_stats,
                    name='Hallucinated Mean',
                    line=dict(color='red')
                ),
                row=2,
                col=2
            )

            # Update layout
            fig.update_layout(
                height=1000,
                showlegend=True,
                template=self.plotly_template
            )

            if save_path and self.config.save_plots:
                fig.write_html(self.output_dir / save_path)

        except Exception as e:
            logger.error(f"Error plotting comparative analysis: {str(e)}")

    def plot_attention_activations(
            self,
            attention_weights: torch.Tensor,
            activations: torch.Tensor,
            token_labels: List[str],
            save_path: Optional[str] = None
    ) -> None:
        """
        Plot combined attention and activation patterns.

        Args:
            attention_weights: Attention weights
            activations: Hidden state activations
            token_labels: Token labels
            save_path: Optional save path
        """
        try:
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=(
                    'Attention Pattern',
                    'Activation Pattern'
                ),
                vertical_spacing=0.2
            )

            # Plot attention pattern
            attention = attention_weights[0].mean(dim=0).cpu().numpy()
            fig.add_trace(
                go.Heatmap(
                    z=attention,
                    x=token_labels,
                    y=token_labels,
                    colorscale=self.color_scheme['attention'],
                    name='Attention'
                ),
                row=1,
                col=1
            )

            # Plot activation pattern
            activation = activations[0].cpu().numpy()
            fig.add_trace(
                go.Heatmap(
                    z=activation,
                    x=token_labels,
                    colorscale=self.color_scheme['activation'],
                    name='Activation'
                ),
                row=2,
                col=1
            )

            # Update layout
            fig.update_layout(
                height=1000,
                showlegend=True,
                template=self.plotly_template
            )

            if save_path and self.config.save_plots:
                fig.write_html(self.output_dir / save_path)

        except Exception as e:
            logger.error(f"Error plotting attention activations: {str(e)}")

    def _add_hallucination_highlights(
            self,
            ax: plt.Axes,
            hallucination_scores: np.ndarray,
            threshold: float = 0.5
    ) -> None:
        """Add hallucination highlight overlays to plot."""
        for i, score in enumerate(hallucination_scores):
            if score > threshold:
                ax.axvline(
                    x=i,
                    color='red',
                    alpha=0.3,
                    linestyle='--'
                )

    def create_activation_dashboard(
            self,
            data: Dict[str, torch.Tensor],
            save_path: Optional[str] = None
    ) -> None:
        """
        Create comprehensive activation analysis dashboard.

        Args:
            data: Dictionary containing activation data
            save_path: Optional save path
        """
        try:
            fig = make_subplots(
                rows=3,
                cols=2,
                subplot_titles=(
                    'Layer-wise Activation Pattern',
                    'Attention-Activation Correlation',
                    'Token-wise Statistics',
                    'Activation Distribution',
                    'Temporal Dynamics',
                    'Pattern Detection'
                )
            )

            # Add layer-wise activation pattern
            if 'layer_activations' in data:
                self._add_layer_pattern(fig, data['layer_activations'], 1, 1)

            # Add attention-activation correlation
            if 'attention_weights' in data and 'activations' in data:
                self._add_correlation_plot(
                    fig,
                    data['attention_weights'],
                    data['activations'],
                    1,
                    2
                )

            # Add token statistics
            if 'token_stats' in data:
                self._add_token_statistics(fig, data['token_stats'], 2, 1)

            # Add activation distribution
            if 'activations' in data:
                self._add_distribution_plot(fig, data['activations'], 2, 2)

            # Add temporal dynamics
            if 'temporal_data' in data:
                self._add_temporal_plot(fig, data['temporal_data'], 3, 1)

            # Add pattern detection
            if 'patterns' in data:
                self._add_pattern_plot(fig, data['patterns'], 3, 2)

            # Update layout
            fig.update_layout(
                height=1200,
                showlegend=True,
                template=self.plotly_template,
                title_text="Activation Analysis Dashboard"
            )

            if save_path and self.config.save_plots:
                fig.write_html(self.output_dir / save_path)

        except Exception as e:
            logger.error(f"Error creating activation dashboard: {str(e)}")

    def save_animation(
            self,
            activation_sequence: List[torch.Tensor],
            save_path: str,
            fps: int = 5
    ) -> None:
        """
        Create animation of activation patterns over time.

        Args:
            activation_sequence: List of activation tensors
            save_path: Path to save animation
            fps: Frames per second
        """
        try:
            import matplotlib.animation as animation

            fig, ax = plt.subplots(figsize=(10, 6))

            def update(frame):
                ax.clear()
                activations = activation_sequence[frame][0].cpu().numpy()
                sns.heatmap(
                    activations,
                    ax=ax,
                    cmap=self.color_scheme['activation'],
                    center=0
                )
                ax.set_title(f'Frame {frame}')

            anim = animation.FuncAnimation(
                fig,
                update,
                frames=len(activation_sequence),
                interval=1000 / fps
            )

            if save_path and self.config.save_plots:
                anim.save(
                    self.output_dir / save_path,
                    writer='pillow'
                )

            plt.close()

        except Exception as e:
            logger.error(f"Error creating activation animation: {str(e)}")


