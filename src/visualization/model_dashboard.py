"""
Model Comparison Dashboard Generator

Creates interactive visualizations for model performance analysis:
- Performance comparison tables and charts
- Time series plots (actual vs predicted)
- Residual analysis
- Feature importance visualizations
- Category-specific breakdowns
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, avg as spark_avg, count, abs as spark_abs
from typing import Dict, List, Optional, Any
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDashboard:
    """
    Generate interactive dashboard for model comparison and analysis.
    
    Example usage:
        dashboard = ModelDashboard()
        dashboard.create_comparison_dashboard(
            results={'RF': rf_results, 'GBT': gbt_results, 'LR': lr_results},
            output_path='model_comparison.html'
        )
    """
    
    def __init__(self, theme: str = 'plotly_white'):
        """
        Args:
            theme: Plotly theme ('plotly', 'plotly_white', 'plotly_dark', 'ggplot2')
        """
        self.theme = theme
    
    def create_comparison_dashboard(
        self,
        results: Dict[str, Dict[str, Any]],
        output_path: str = 'model_dashboard.html',
        title: str = 'Model Performance Comparison'
    ) -> None:
        """
        Create comprehensive comparison dashboard.
        
        Args:
            results: Dictionary mapping model names to results dictionaries
            output_path: Path to save HTML dashboard
            title: Dashboard title
        """
        logger.info(f"📊 Creating model comparison dashboard...")
        logger.info(f"   Models: {list(results.keys())}")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'RMSE Comparison',
                'MAPE Comparison',
                'R² Comparison',
                'MAE Comparison'
            ),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Extract metrics
        models = list(results.keys())
        rmse_values = [results[m].get('rmse', 0) for m in models]
        mape_values = [results[m].get('mape', 0) for m in models]
        r2_values = [results[m].get('r2', 0) for m in models]
        mae_values = [results[m].get('mae', 0) for m in models]
        
        # Add bar charts
        fig.add_trace(
            go.Bar(name='RMSE', x=models, y=rmse_values, marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='MAPE', x=models, y=mape_values, marker_color='lightcoral'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name='R²', x=models, y=r2_values, marker_color='lightgreen'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(name='MAE', x=models, y=mae_values, marker_color='lightyellow'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text=title,
            showlegend=False,
            height=800,
            template=self.theme
        )
        
        # Save
        fig.write_html(output_path)
        logger.info(f"✅ Dashboard saved to {output_path}")
    
    def create_time_series_plot(
        self,
        actual_values: List[float],
        predicted_values: Dict[str, List[float]],
        dates: List[str],
        output_path: str = 'time_series_plot.html',
        title: str = 'Actual vs Predicted'
    ) -> None:
        """
        Create time series comparison plot.
        
        Args:
            actual_values: List of actual values
            predicted_values: Dictionary mapping model names to predictions
            dates: List of dates (x-axis)
            output_path: Output file path
            title: Plot title
        """
        logger.info("📈 Creating time series plot...")
        
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual_values,
            mode='lines+markers',
            name='Actual',
            line=dict(color='black', width=2),
            marker=dict(size=6)
        ))
        
        # Add predictions for each model
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (model_name, predictions) in enumerate(predicted_values.items()):
            fig.add_trace(go.Scatter(
                x=dates,
                y=predictions,
                mode='lines',
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            template=self.theme,
            hovermode='x unified',
            height=500
        )
        
        fig.write_html(output_path)
        logger.info(f"✅ Time series plot saved to {output_path}")
    
    def create_residual_plot(
        self,
        residuals: Dict[str, List[float]],
        output_path: str = 'residual_plot.html',
        title: str = 'Residual Analysis'
    ) -> None:
        """
        Create residual analysis plots.
        
        Args:
            residuals: Dictionary mapping model names to residual values
            output_path: Output file path
            title: Plot title
        """
        logger.info("📉 Creating residual plot...")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Residual Distribution', 'Residual Q-Q Plot')
        )
        
        for model_name, resid_values in residuals.items():
            # Histogram
            fig.add_trace(
                go.Histogram(x=resid_values, name=model_name, opacity=0.7),
                row=1, col=1
            )
        
        fig.update_layout(
            title_text=title,
            template=self.theme,
            height=500,
            barmode='overlay'
        )
        
        fig.write_html(output_path)
        logger.info(f"✅ Residual plot saved to {output_path}")
    
    def create_feature_importance_plot(
        self,
        feature_importance: Dict[str, float],
        top_k: int = 20,
        output_path: str = 'feature_importance.html',
        title: str = 'Feature Importance'
    ) -> None:
        """
        Create feature importance bar chart.
        
        Args:
            feature_importance: Dictionary of feature importances
            top_k: Number of top features to display
            output_path: Output file path
            title: Plot title
        """
        logger.info(f"📊 Creating feature importance plot (top {top_k})...")
        
        # Sort and select top K
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        features = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]
        
        fig = go.Figure(data=[
            go.Bar(
                x=importances,
                y=features,
                orientation='h',
                marker=dict(
                    color=importances,
                    colorscale='Viridis',
                    showscale=True
                )
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance',
            yaxis_title='Feature',
            template=self.theme,
            height=max(400, top_k * 25),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        fig.write_html(output_path)
        logger.info(f"✅ Feature importance plot saved to {output_path}")
    
    def create_category_performance_plot(
        self,
        category_results: Dict[str, Dict[str, float]],
        metric: str = 'rmse',
        output_path: str = 'category_performance.html',
        title: str = 'Performance by Category'
    ) -> None:
        """
        Create performance breakdown by product category.
        
        Args:
            category_results: Nested dict {model_name: {category: metric_value}}
            metric: Metric to plot
            output_path: Output file path
            title: Plot title
        """
        logger.info(f"📊 Creating category performance plot for {metric}...")
        
        fig = go.Figure()
        
        # Prepare data
        categories = set()
        for model_data in category_results.values():
            categories.update(model_data.keys())
        categories = sorted(list(categories))
        
        # Add trace for each model
        for model_name, cat_data in category_results.items():
            values = [cat_data.get(cat, 0) for cat in categories]
            fig.add_trace(go.Bar(
                name=model_name,
                x=categories,
                y=values
            ))
        
        fig.update_layout(
            title=f'{title} - {metric.upper()}',
            xaxis_title='Category',
            yaxis_title=metric.upper(),
            template=self.theme,
            barmode='group',
            height=500
        )
        
        fig.write_html(output_path)
        logger.info(f"✅ Category performance plot saved to {output_path}")


def create_comprehensive_dashboard(
    model_results: Dict[str, Dict[str, Any]],
    predictions_df: Optional[DataFrame] = None,
    feature_importance: Optional[Dict[str, Dict[str, float]]] = None,
    output_dir: str = '.',
    base_filename: str = 'model_analysis'
) -> List[str]:
    """
    Create comprehensive multi-page dashboard with all visualizations.
    
    Args:
        model_results: Dictionary of model results
        predictions_df: Optional DataFrame with predictions
        feature_importance: Optional dictionary of feature importances per model
        output_dir: Output directory
        base_filename: Base filename for outputs
    
    Returns:
        List of created file paths
    """
    logger.info("🎨 Creating comprehensive dashboard suite...")
    
    dashboard = ModelDashboard()
    created_files = []
    
    # 1. Main comparison dashboard
    comparison_path = f"{output_dir}/{base_filename}_comparison.html"
    dashboard.create_comparison_dashboard(
        model_results,
        output_path=comparison_path,
        title='Model Performance Comparison'
    )
    created_files.append(comparison_path)
    
    # 2. Feature importance plots (if available)
    if feature_importance:
        for model_name, importance_dict in feature_importance.items():
            importance_path = f"{output_dir}/{base_filename}_importance_{model_name}.html"
            dashboard.create_feature_importance_plot(
                importance_dict,
                output_path=importance_path,
                title=f'Feature Importance - {model_name}'
            )
            created_files.append(importance_path)
    
    # 3. Summary report (HTML)
    summary_path = f"{output_dir}/{base_filename}_summary.html"
    _create_summary_report(model_results, summary_path)
    created_files.append(summary_path)
    
    logger.info(f"✅ Dashboard suite complete! Created {len(created_files)} files:")
    for file_path in created_files:
        logger.info(f"   - {file_path}")
    
    return created_files


def _create_summary_report(
    model_results: Dict[str, Dict[str, Any]],
    output_path: str
) -> None:
    """Create HTML summary report."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Performance Summary</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
            h2 { color: #555; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; background: white; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .best { background-color: #c8e6c9 !important; font-weight: bold; }
            .metric { font-size: 18px; }
        </style>
    </head>
    <body>
        <h1>📊 Model Performance Summary Report</h1>
    """
    
    # Create metrics table
    html_content += "<h2>Performance Metrics</h2><table>"
    html_content += "<tr><th>Model</th><th>RMSE</th><th>MAPE (%)</th><th>R²</th><th>MAE</th></tr>"
    
    # Find best models for each metric
    if model_results:
        rmse_values = {m: r.get('rmse', float('inf')) for m, r in model_results.items()}
        best_rmse_model = min(rmse_values, key=rmse_values.get)
        
        mape_values = {m: r.get('mape', float('inf')) for m, r in model_results.items()}
        best_mape_model = min(mape_values, key=mape_values.get)
        
        r2_values = {m: r.get('r2', float('-inf')) for m, r in model_results.items()}
        best_r2_model = max(r2_values, key=r2_values.get)
    
    for model_name, results in model_results.items():
        rmse = results.get('rmse', 'N/A')
        mape = results.get('mape', 'N/A')
        r2 = results.get('r2', 'N/A')
        mae = results.get('mae', 'N/A')
        
        rmse_class = 'best' if model_name == best_rmse_model else ''
        mape_class = 'best' if model_name == best_mape_model else ''
        r2_class = 'best' if model_name == best_r2_model else ''
        
        html_content += f"<tr>"
        html_content += f"<td><strong>{model_name}</strong></td>"
        html_content += f"<td class='{rmse_class}'>{rmse if isinstance(rmse, str) else f'{rmse:.4f}'}</td>"
        html_content += f"<td class='{mape_class}'>{mape if isinstance(mape, str) else f'{mape:.2f}'}</td>"
        html_content += f"<td class='{r2_class}'>{r2 if isinstance(r2, str) else f'{r2:.4f}'}</td>"
        html_content += f"<td>{mae if isinstance(mae, str) else f'{mae:.4f}'}</td>"
        html_content += "</tr>"
    
    html_content += "</table>"
    
    # Recommendations
    html_content += "<h2>💡 Recommendations</h2><ul>"
    html_content += f"<li>✅ <strong>Best RMSE:</strong> {best_rmse_model} ({rmse_values[best_rmse_model]:.4f})</li>"
    html_content += f"<li>✅ <strong>Best MAPE:</strong> {best_mape_model} ({mape_values[best_mape_model]:.2f}%)</li>"
    html_content += f"<li>✅ <strong>Best R²:</strong> {best_r2_model} ({r2_values[best_r2_model]:.4f})</li>"
    html_content += "<li>💡 Consider ensemble methods to combine strengths of multiple models</li>"
    html_content += "<li>📊 Review feature importance to understand key drivers</li>"
    html_content += "</ul>"
    
    html_content += "</body></html>"
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"   Summary report created: {output_path}")


def plot_model_comparison_matplotlib(
    model_results: Dict[str, Dict[str, Any]],
    output_path: str = 'model_comparison.png',
    figsize: tuple = (12, 8)
) -> None:
    """
    Create model comparison plot using matplotlib (for reports/papers).
    
    Args:
        model_results: Dictionary of model results
        output_path: Output file path
        figsize: Figure size
    """
    logger.info("📊 Creating matplotlib comparison plot...")
    
    # Set style
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    models = list(model_results.keys())
    rmse_values = [model_results[m].get('rmse', 0) for m in models]
    mape_values = [model_results[m].get('mape', 0) for m in models]
    r2_values = [model_results[m].get('r2', 0) for m in models]
    mae_values = [model_results[m].get('mae', 0) for m in models]
    
    # RMSE
    axes[0, 0].bar(models, rmse_values, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('RMSE (Lower is Better)', fontweight='bold')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # MAPE
    axes[0, 1].bar(models, mape_values, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('MAPE % (Lower is Better)', fontweight='bold')
    axes[0, 1].set_ylabel('MAPE %')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # R²
    axes[1, 0].bar(models, r2_values, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title('R² (Higher is Better)', fontweight='bold')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # MAE
    axes[1, 1].bar(models, mae_values, color='lightyellow', edgecolor='black')
    axes[1, 1].set_title('MAE (Lower is Better)', fontweight='bold')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Matplotlib plot saved to {output_path}")
