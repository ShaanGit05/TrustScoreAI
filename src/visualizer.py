# src/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger("bias_detection.visualizer")

class BiasVisualizer:
    """Generates visualizations for bias detection results."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        self.style = style
        plt.style.use(style)
        
        # Set color palette
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'tertiary': '#2ca02c',
            'quaternary': '#d62728',
            'background': '#f8f9fa',
            'text': '#212529'
        }
    
    def create_ubi_gauge(self, ubi_score: float, save_path: Optional[str] = None) -> str:
        """Create UBI score gauge chart."""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = ubi_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "UBI Score"},
            delta = {'reference': 0.5},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.2], 'color': "lightgreen"},
                    {'range': [0.2, 0.4], 'color': "yellow"},
                    {'range': [0.4, 0.7], 'color': "orange"},
                    {'range': [0.7, 1], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        
        fig.update_layout(
            font={'color': "darkblue", 'family': "Arial"},
            height=400,
            margin={'t': 50, 'b': 50, 'l': 25, 'r': 25}
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"UBI gauge saved to {save_path}")
        
        return fig
    
    def create_component_breakdown(self, components: Dict[str, float], 
                                 weights: Dict[str, float],
                                 save_path: Optional[str] = None) -> str:
        """Create component breakdown chart."""
        categories = list(components.keys())
        scores = list(components.values())
        weight_values = [weights.get(cat.lower(), 0.0) for cat in categories]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Component Scores', 'Component Weights'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Component scores
        fig.add_trace(
            go.Bar(x=categories, y=scores, name='Scores', 
                  marker_color=['#ff9999', '#66b3ff', '#99ff99']),
            row=1, col=1
        )
        
        # Component weights
        fig.add_trace(
            go.Bar(x=categories, y=weight_values, name='Weights',
                  marker_color=['#ff6666', '#3388ff', '#66ff66'],
                  opacity=0.6),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="UBI Component Analysis",
            showlegend=False,
            height=400
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Component breakdown saved to {save_path}")
        
        return fig
    
    def create_bias_timeline(self, results_history: List[Dict[str, Any]], 
                           save_path: Optional[str] = None) -> str:
        """Create bias score timeline."""
        if not results_history:
            logger.warning("No historical data for timeline")
            return None
        
        df = pd.DataFrame(results_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['ubi_score'],
            mode='lines+markers',
            name='UBI Score',
            line=dict(color=self.colors['primary'], width=3)
        ))
        
        fig.update_layout(
            title="UBI Score Timeline",
            xaxis_title="Time",
            yaxis_title="UBI Score",
            height=400
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Bias timeline saved to {save_path}")
        
        return fig
    
    def create_category_comparison(self, category_scores: Dict[str, Dict[str, float]],
                                 save_path: Optional[str] = None) -> str:
        """Create category comparison chart."""
        categories = list(category_scores.keys())
        components = ['bias_magnitude', 'disparity', 'distribution_shift']
        
        fig = go.Figure()
        
        for component in components:
            values = [category_scores[cat].get(component, 0.0) for cat in categories]
            fig.add_trace(go.Bar(
                name=component.replace('_', ' ').title(),
                x=categories,
                y=values
            ))
        
        fig.update_layout(
            title="Bias Components by Category",
            xaxis_title="Category",
            yaxis_title="Score",
            barmode='group',
            height=400
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Category comparison saved to {save_path}")
        
        return fig
    
    def create_heatmap(self, matrix_data: np.ndarray, 
                      row_labels: List[str], col_labels: List[str],
                      title: str = "Bias Heatmap",
                      save_path: Optional[str] = None) -> str:
        """Create bias heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=matrix_data,
            x=col_labels,
            y=row_labels,
            colorscale='RdYlBu_r',
            showscale=True
        ))
        
        fig.update_layout(
            title=title,
            height=400
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Heatmap saved to {save_path}")
        
        return fig
    
    def generate_comprehensive_visualizations(self, results: Dict[str, Any], 
                                            output_dir: str) -> List[str]:
        """Generate all visualizations for results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        try:
            # UBI Gauge
            gauge_path = output_path / "ubi_gauge.html"
            self.create_ubi_gauge(results['ubi_score'], str(gauge_path))
            generated_files.append(str(gauge_path))
            
            # Component Breakdown
            component_path = output_path / "component_breakdown.html"
            self.create_component_breakdown(
                results['components'], 
                results['weights'],
                str(component_path)
            )
            generated_files.append(str(component_path))
            
            # Category Comparison (if available)
            if 'calibrated_scores' in results:
                category_scores = {}
                for category, stats in results['calibrated_scores'].items():
                    category_scores[category] = {
                        'bias_magnitude': stats.get('mean', 0.0),
                        'disparity': 0.0,  # Would need separate calculation
                        'distribution_shift': 0.0  # Would need separate calculation
                    }
                
                category_path = output_path / "category_comparison.html"
                self.create_category_comparison(category_scores, str(category_path))
                generated_files.append(str(category_path))
            
            logger.info(f"Generated {len(generated_files)} visualizations")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        return generated_files
    
    def create_matplotlib_plots(self, results: Dict[str, Any], 
                              output_dir: str) -> List[str]:
        """Create matplotlib-based plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        try:
            # UBI Score Bar Chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            components = list(results['components'].keys())
            scores = list(results['components'].values())
            
            bars = ax.bar(components, scores, color=['#ff9999', '#66b3ff', '#99ff99'])
            ax.set_title('UBI Component Scores', fontsize=16, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            bar_path = output_path / "component_scores.png"
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files.append(str(bar_path))
            
            # Overall UBI Score
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ubi_score = results['ubi_score']
            bias_level = results['bias_level']
            
            # Create gauge-like visualization
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            ax.plot(theta, r, 'k-', linewidth=2)
            ax.fill_between(theta, 0, r, alpha=0.3, color='lightblue')
            
            # Add UBI score indicator
            score_angle = ubi_score * np.pi
            ax.plot([score_angle, score_angle], [0, 1], 'r-', linewidth=4)
            
            ax.set_title(f'UBI Score: {ubi_score:.3f} ({bias_level})', 
                        fontsize=16, fontweight='bold')
            ax.set_ylim(0, 1.2)
            ax.set_xlim(0, np.pi)
            ax.axis('off')
            
            # Add text annotations
            ax.text(np.pi/2, 1.1, f'UBI Score: {ubi_score:.3f}', 
                   ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(np.pi/2, 0.5, bias_level, 
                   ha='center', va='center', fontsize=12)
            
            plt.tight_layout()
            
            gauge_path = output_path / "ubi_score_gauge.png"
            plt.savefig(gauge_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files.append(str(gauge_path))
            
            logger.info(f"Generated {len(generated_files)} matplotlib plots")
            
        except Exception as e:
            logger.error(f"Error generating matplotlib plots: {e}")
        
        return generated_files

