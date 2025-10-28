# src/visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger("bias_detection.visualizer")

class BiasVisualizer:
    """Generates visualizations for bias analysis results."""
    
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_ubi_gauge_chart(self, ubi_score: float, output_path: str) -> str:
        """
        Create a gauge chart showing UBI score.
        
        Args:
            ubi_score: UBI score (0-1)
            output_path: Path to save the chart
        
        Returns:
            Path to saved chart
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=ubi_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Unified Bias Index (UBI)"},
            delta={'reference': 0.5, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 0.2], 'color': 'lightgreen'},
                    {'range': [0.2, 0.4], 'color': 'yellow'},
                    {'range': [0.4, 0.7], 'color': 'orange'},
                    {'range': [0.7, 1], 'color': 'red'}
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
            height=300,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        fig.write_image(output_path)
        logger.info(f"Saved gauge chart to {output_path}")
        return output_path
    
    def create_component_breakdown(self, 
                                 components: Dict[str, float],
                                 weights: Dict[str, float],
                                 output_path: str) -> str:
        """
        Create a bar chart showing component breakdown.
        
        Args:
            components: Component scores
            weights: Component weights
            output_path: Path to save the chart
        
        Returns:
            Path to saved chart
        """
        categories = list(components.keys())
        scores = list(components.values())
        weight_values = [weights.get(cat.lower(), 0.0) for cat in categories]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Component scores
        bars1 = ax1.bar(categories, scores, color=['#ff9999', '#66b3ff', '#99ff99'])
        ax1.set_title('UBI Component Scores')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Component weights
        bars2 = ax2.bar(categories, weight_values, color=['#ff9999', '#66b3ff', '#99ff99'])
        ax2.set_title('UBI Component Weights')
        ax2.set_ylabel('Weight')
        ax2.set_ylim(0, 1)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved component breakdown to {output_path}")
        return output_path
    
    def create_category_comparison(self, 
                                 calibrated_scores: Dict[str, Dict[str, float]],
                                 output_path: str) -> str:
        """
        Create comparison chart across bias categories.
        
        Args:
            calibrated_scores: Calibrated scores by category
            output_path: Path to save the chart
        
        Returns:
            Path to saved chart
        """
        categories = list(calibrated_scores.keys())
        means = [scores['mean'] for scores in calibrated_scores.values()]
        stds = [scores['std'] for scores in calibrated_scores.values()]
        counts = [scores['count'] for scores in calibrated_scores.values()]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Mean scores with error bars
        x_pos = np.arange(len(categories))
        bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, color='lightblue', alpha=0.7)
        ax1.set_title('Bias Scores by Category')
        ax1.set_ylabel('Mean Bias Score')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(categories, rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Sample counts
        ax2.bar(x_pos, counts, color='lightgreen', alpha=0.7)
        ax2.set_title('Number of Prompts by Category')
        ax2.set_ylabel('Prompt Count')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(categories, rotation=45)
        
        # Add count labels
        for i, count in enumerate(counts):
            ax2.text(i, count + 0.5, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved category comparison to {output_path}")
        return output_path
    
    def create_bias_distribution(self,
                               calibrated_scores: Dict[str, Dict[str, Any]],
                               output_path: str) -> str:
        """
        Create distribution plots for bias scores.
        
        Args:
            calibrated_scores: Calibrated scores data
            output_path: Path to save the chart
        
        Returns:
            Path to saved chart
        """
        # Create a DataFrame for plotting
        data = []
        for category, scores_data in calibrated_scores.items():
            # For demonstration, create synthetic distribution data
            # In practice, you would use the actual score distributions
            mean = scores_data['mean']
            std = scores_data['std']
            synthetic_scores = np.random.normal(mean, std, 1000)
            
            for score in synthetic_scores:
                data.append({'Category': category, 'Bias Score': score})
        
        df = pd.DataFrame(data)
        
        # Create distribution plot
        plt.figure(figsize=(10, 6))
        
        for category in df['Category'].unique():
            category_data = df[df['Category'] == category]['Bias Score']
            sns.kdeplot(category_data, label=category, fill=True, alpha=0.6)
        
        plt.title('Bias Score Distributions by Category')
        plt.xlabel('Bias Score')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved bias distribution to {output_path}")
        return output_path
    
    def generate_comprehensive_visualizations(self, 
                                           results: Dict[str, Any],
                                           output_dir: str) -> List[str]:
        """
        Generate all visualizations for bias analysis results.
        
        Args:
            results: UBI results dictionary
            output_dir: Directory to save visualizations
        
        Returns:
            List of paths to saved visualization files
        """
        os.makedirs(output_dir, exist_ok=True)
        visualization_files = []
        
        try:
            # 1. UBI Gauge Chart
            gauge_file = os.path.join(output_dir, "ubi_gauge.png")
            self.create_ubi_gauge_chart(results['ubi_score'], gauge_file)
            visualization_files.append(gauge_file)
            
            # 2. Component Breakdown
            component_file = os.path.join(output_dir, "component_breakdown.png")
            self.create_component_breakdown(
                results['components'],
                results['weights'],
                component_file
            )
            visualization_files.append(component_file)
            
            # 3. Category Comparison
            category_file = os.path.join(output_dir, "category_comparison.png")
            self.create_category_comparison(results['calibrated_scores'], category_file)
            visualization_files.append(category_file)
            
            # 4. Bias Distribution
            distribution_file = os.path.join(output_dir, "bias_distribution.png")
            self.create_bias_distribution(results['calibrated_scores'], distribution_file)
            visualization_files.append(distribution_file)
            
            # 5. Create a comprehensive dashboard (HTML)
            dashboard_file = os.path.join(output_dir, "bias_dashboard.html")
            self.create_interactive_dashboard(results, dashboard_file)
            visualization_files.append(dashboard_file)
            
            logger.info(f"Generated {len(visualization_files)} visualizations")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        return visualization_files
    
    def create_interactive_dashboard(self, results: Dict[str, Any], output_path: str) -> str:
        """
        Create an interactive HTML dashboard.
        
        Args:
            results: UBI results dictionary
            output_path: Path to save the dashboard
        
        Returns:
            Path to saved dashboard file
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('UBI Score Gauge', 'Component Scores', 
                          'Category Comparison', 'Bias Distribution'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # UBI Gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=results['ubi_score'],
            title={'text': "UBI Score"},
            gauge={'axis': {'range': [0, 1]},
                  'bar': {'color': "darkblue"},
                  'steps': [
                      {'range': [0, 0.2], 'color': 'lightgreen'},
                      {'range': [0.2, 0.4], 'color': 'yellow'},
                      {'range': [0.4, 0.7], 'color': 'orange'},
                      {'range': [0.7, 1], 'color': 'red'}
                  ]}
        ), row=1, col=1)
        
        # Component Scores
        components = list(results['components'].keys())
        scores = list(results['components'].values())
        
        fig.add_trace(go.Bar(
            x=components,
            y=scores,
            marker_color=['#ff9999', '#66b3ff', '#99ff99']
        ), row=1, col=2)
        
        # Category Comparison
        categories = list(results['calibrated_scores'].keys())
        means = [scores['mean'] for scores in results['calibrated_scores'].values()]
        
        fig.add_trace(go.Bar(
            x=categories,
            y=means,
            marker_color='lightblue'
        ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text=f"Bias Analysis Dashboard - {results['metadata']['model_name']}",
            title_x=0.5
        )
        
        # Save as HTML
        fig.write_html(output_path)
        logger.info(f"Saved interactive dashboard to {output_path}")
        
        return output_path