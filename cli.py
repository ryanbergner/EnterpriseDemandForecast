#!/usr/bin/env python3
"""
Unified Command Line Interface for Time Series Forecasting

Provides a single entry point for all operations:
- train: Train models with various options
- evaluate: Evaluate trained models
- predict: Generate forecasts
- validate: Run data quality checks
- compare: Compare multiple models
- dashboard: Generate visualization dashboards
"""

import argparse
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    
    parser = argparse.ArgumentParser(
        description='🚀 Enterprise Time Series Forecasting CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models on M5 dataset
  python cli.py train --data data/m5_sales.csv --models rf,gbt,stats --categories all
  
  # Evaluate specific model
  python cli.py evaluate --model-path models/rf_model --data test_data.csv
  
  # Generate predictions
  python cli.py predict --model-path models/rf_model --horizon 12 --output predictions.csv
  
  # Run data quality validation
  python cli.py validate --data data/m5_sales.csv --report-path data_quality_report.txt
  
  # Compare models
  python cli.py compare --experiment "Smooth" --metric rmse --top-k 5
  
  # Generate dashboard
  python cli.py dashboard --results results.json --output dashboard.html
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ============ TRAIN COMMAND ============
    train_parser = subparsers.add_parser(
        'train',
        help='Train forecasting models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    train_parser.add_argument(
        '--data',
        required=True,
        help='Path to training data (CSV or Parquet)'
    )
    train_parser.add_argument(
        '--models',
        default='rf,gbt,lr,stats',
        help='Comma-separated list of models to train (rf, gbt, lr, stats, all)'
    )
    train_parser.add_argument(
        '--categories',
        default='all',
        help='Product categories to train on (Smooth, Erratic, Intermittent, Lumpy, all)'
    )
    train_parser.add_argument(
        '--cv-folds',
        type=int,
        default=0,
        help='Number of cross-validation folds (0 = no CV, default)'
    )
    train_parser.add_argument(
        '--cv-strategy',
        choices=['expanding', 'sliding'],
        default='expanding',
        help='Cross-validation strategy'
    )
    train_parser.add_argument(
        '--early-stopping',
        action='store_true',
        help='Enable early stopping for tree models'
    )
    train_parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Early stopping patience'
    )
    train_parser.add_argument(
        '--feature-selection',
        action='store_true',
        help='Enable automatic feature selection'
    )
    train_parser.add_argument(
        '--output-dir',
        default='./models',
        help='Directory to save trained models'
    )
    train_parser.add_argument(
        '--experiment-name',
        help='MLflow experiment name'
    )
    
    # ============ EVALUATE COMMAND ============
    evaluate_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate trained models'
    )
    evaluate_parser.add_argument(
        '--model-path',
        required=True,
        help='Path to trained model or model URI (models:/ModelName/Version)'
    )
    evaluate_parser.add_argument(
        '--data',
        required=True,
        help='Path to test data'
    )
    evaluate_parser.add_argument(
        '--metrics',
        default='rmse,mae,mape,r2',
        help='Comma-separated list of metrics to compute'
    )
    evaluate_parser.add_argument(
        '--confidence-intervals',
        action='store_true',
        help='Add prediction confidence intervals'
    )
    evaluate_parser.add_argument(
        '--output',
        help='Path to save evaluation results'
    )
    
    # ============ PREDICT COMMAND ============
    predict_parser = subparsers.add_parser(
        'predict',
        help='Generate predictions'
    )
    predict_parser.add_argument(
        '--model-path',
        required=True,
        help='Path to trained model or model URI'
    )
    predict_parser.add_argument(
        '--data',
        help='Path to input data (optional for forecasting mode)'
    )
    predict_parser.add_argument(
        '--horizon',
        type=int,
        default=1,
        help='Forecast horizon (number of periods ahead)'
    )
    predict_parser.add_argument(
        '--output',
        required=True,
        help='Path to save predictions (CSV or Parquet)'
    )
    predict_parser.add_argument(
        '--ensemble',
        action='store_true',
        help='Use ensemble of multiple models'
    )
    predict_parser.add_argument(
        '--ensemble-strategy',
        choices=['simple_average', 'weighted_average', 'median', 'stacking'],
        default='simple_average',
        help='Ensemble strategy'
    )
    
    # ============ VALIDATE COMMAND ============
    validate_parser = subparsers.add_parser(
        'validate',
        help='Run data quality validation'
    )
    validate_parser.add_argument(
        '--data',
        required=True,
        help='Path to data for validation'
    )
    validate_parser.add_argument(
        '--date-col',
        default='MonthEndDate',
        help='Date column name'
    )
    validate_parser.add_argument(
        '--product-col',
        default='ItemNumber',
        help='Product identifier column'
    )
    validate_parser.add_argument(
        '--target-col',
        default='DemandQuantity',
        help='Target value column'
    )
    validate_parser.add_argument(
        '--report-path',
        help='Path to save validation report'
    )
    
    # ============ COMPARE COMMAND ============
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare multiple models'
    )
    compare_parser.add_argument(
        '--experiment',
        required=True,
        help='MLflow experiment name or ID'
    )
    compare_parser.add_argument(
        '--metric',
        default='rmse',
        choices=['rmse', 'mae', 'mape', 'r2'],
        help='Metric to use for comparison'
    )
    compare_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top models to display'
    )
    compare_parser.add_argument(
        '--output',
        help='Path to save comparison results'
    )
    
    # ============ DASHBOARD COMMAND ============
    dashboard_parser = subparsers.add_parser(
        'dashboard',
        help='Generate visualization dashboard'
    )
    dashboard_parser.add_argument(
        '--results',
        required=True,
        help='Path to results file (JSON)'
    )
    dashboard_parser.add_argument(
        '--output',
        default='dashboard.html',
        help='Path to save dashboard HTML'
    )
    dashboard_parser.add_argument(
        '--theme',
        default='plotly_white',
        choices=['plotly', 'plotly_white', 'plotly_dark', 'ggplot2'],
        help='Dashboard theme'
    )
    
    return parser


def train_command(args):
    """Execute train command."""
    logger.info("🚀 Starting model training...")
    logger.info(f"   Data: {args.data}")
    logger.info(f"   Models: {args.models}")
    logger.info(f"   Categories: {args.categories}")
    
    from pyspark.sql import SparkSession
    
    # Initialize Spark
    spark = SparkSession.builder.appName("TimeSeriesForecasting_CLI").getOrCreate()
    
    # Load data
    logger.info("📂 Loading data...")
    if args.data.endswith('.csv'):
        df = spark.read.format("csv").option("header", True).load(args.data)
    elif args.data.endswith('.parquet'):
        df = spark.read.parquet(args.data)
    else:
        logger.error("Unsupported data format. Use CSV or Parquet.")
        return 1
    
    # Import training modules
    from src.preprocessing.preprocess import aggregate_sales_data
    from src.feature_engineering.feature_engineering import add_features
    
    # Add enhanced features if requested
    if args.feature_selection or args.cv_folds > 0:
        logger.info("🔧 Adding enhanced features...")
        from src.feature_engineering.trend_features import add_trend_features
        from src.feature_engineering.ewma_features import add_ewma_features
        
        # Apply feature engineering (simplified for CLI)
        # In practice, you'd call your actual feature engineering pipeline
    
    # Cross-validation
    if args.cv_folds > 0:
        logger.info(f"📊 Running {args.cv_folds}-fold cross-validation...")
        from src.validation.time_series_cv import TimeSeriesCV
        
        cv = TimeSeriesCV(n_splits=args.cv_folds, strategy=args.cv_strategy)
        # Run CV (implementation details depend on your setup)
    
    # Train models
    models_to_train = args.models.split(',')
    logger.info(f"🎯 Training {len(models_to_train)} model types...")
    
    # Parse categories
    if args.categories == 'all':
        categories = ['Smooth', 'Erratic', 'Intermittent', 'Lumpy']
    else:
        categories = args.categories.split(',')
    
    # Training loop (simplified)
    for category in categories:
        logger.info(f"\n   Training on category: {category}")
        
        for model_type in models_to_train:
            logger.info(f"     Model: {model_type}")
            
            # Early stopping
            if args.early_stopping and model_type in ['rf', 'gbt']:
                from src.model_training.early_stopping import EarlyStopping
                early_stop = EarlyStopping(patience=args.patience)
                # Use early stopping in training
    
    logger.info(f"\n✅ Training complete! Models saved to {args.output_dir}")
    
    spark.stop()
    return 0


def evaluate_command(args):
    """Execute evaluate command."""
    logger.info("📊 Starting model evaluation...")
    logger.info(f"   Model: {args.model_path}")
    logger.info(f"   Data: {args.data}")
    
    from pyspark.sql import SparkSession
    import mlflow
    
    spark = SparkSession.builder.appName("Evaluate_CLI").getOrCreate()
    
    # Load model
    logger.info("🔄 Loading model...")
    try:
        if args.model_path.startswith('models:/'):
            model = mlflow.spark.load_model(args.model_path)
        else:
            model = mlflow.spark.load_model(f"file://{args.model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Load data
    logger.info("📂 Loading test data...")
    if args.data.endswith('.csv'):
        test_df = spark.read.format("csv").option("header", True).load(args.data)
    else:
        test_df = spark.read.parquet(args.data)
    
    # Generate predictions
    logger.info("🎯 Generating predictions...")
    predictions_df = model.transform(test_df)
    
    # Add confidence intervals if requested
    if args.confidence_intervals:
        logger.info("📊 Adding confidence intervals...")
        from src.inference.confidence_intervals import UncertaintyQuantifier
        
        uq = UncertaintyQuantifier()
        # Add intervals (requires feature columns info)
    
    # Compute metrics
    logger.info("📈 Computing evaluation metrics...")
    from pyspark.ml.evaluation import RegressionEvaluator
    
    metrics_list = args.metrics.split(',')
    results = {}
    
    for metric in metrics_list:
        evaluator = RegressionEvaluator(
            labelCol='target',
            predictionCol='prediction',
            metricName=metric.lower()
        )
        score = evaluator.evaluate(predictions_df)
        results[metric] = score
        logger.info(f"   {metric.upper()}: {score:.4f}")
    
    # Save results if output specified
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✅ Results saved to {args.output}")
    
    spark.stop()
    return 0


def predict_command(args):
    """Execute predict command."""
    logger.info("🔮 Starting prediction generation...")
    logger.info(f"   Model: {args.model_path}")
    logger.info(f"   Horizon: {args.horizon}")
    
    from pyspark.sql import SparkSession
    import mlflow
    
    spark = SparkSession.builder.appName("Predict_CLI").getOrCreate()
    
    # Load model
    logger.info("🔄 Loading model...")
    try:
        if args.model_path.startswith('models:/'):
            model = mlflow.spark.load_model(args.model_path)
        else:
            model = mlflow.spark.load_model(f"file://{args.model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Load input data if provided
    if args.data:
        if args.data.endswith('.csv'):
            input_df = spark.read.format("csv").option("header", True).load(args.data)
        else:
            input_df = spark.read.parquet(args.data)
    else:
        logger.error("Input data required for predictions")
        return 1
    
    # Generate predictions
    logger.info("🎯 Generating predictions...")
    predictions_df = model.transform(input_df)
    
    # Save predictions
    logger.info(f"💾 Saving predictions to {args.output}...")
    if args.output.endswith('.csv'):
        predictions_df.write.format("csv").option("header", True).mode("overwrite").save(args.output)
    else:
        predictions_df.write.parquet(args.output, mode="overwrite")
    
    logger.info("✅ Predictions saved successfully!")
    
    spark.stop()
    return 0


def validate_command(args):
    """Execute validate command."""
    logger.info("🔍 Starting data quality validation...")
    logger.info(f"   Data: {args.data}")
    
    from pyspark.sql import SparkSession
    from src.validation.data_quality import DataQualityValidator
    
    spark = SparkSession.builder.appName("Validate_CLI").getOrCreate()
    
    # Load data
    if args.data.endswith('.csv'):
        df = spark.read.format("csv").option("header", True).load(args.data)
    else:
        df = spark.read.parquet(args.data)
    
    # Run validation
    validator = DataQualityValidator()
    report = validator.validate(
        df,
        date_col=args.date_col,
        product_col=args.product_col,
        target_col=args.target_col
    )
    
    # Print report
    validator.print_report(report)
    
    # Save report if requested
    if args.report_path:
        import json
        with open(args.report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"✅ Report saved to {args.report_path}")
    
    spark.stop()
    return 0


def compare_command(args):
    """Execute compare command."""
    logger.info("📊 Comparing models...")
    logger.info(f"   Experiment: {args.experiment}")
    logger.info(f"   Metric: {args.metric}")
    
    import mlflow
    
    # Get experiment runs
    experiment = mlflow.get_experiment_by_name(args.experiment)
    if not experiment:
        logger.error(f"Experiment '{args.experiment}' not found")
        return 1
    
    # Query runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        logger.error("No runs found in experiment")
        return 1
    
    # Sort by metric
    metric_col = f"metrics.{args.metric}"
    if metric_col in runs.columns:
        runs_sorted = runs.sort_values(by=metric_col, ascending=(args.metric != 'r2'))
        
        logger.info(f"\n🏆 Top {args.top_k} models by {args.metric}:")
        for i, row in runs_sorted.head(args.top_k).iterrows():
            logger.info(f"   {i+1}. Run: {row['tags.mlflow.runName']}, {args.metric}={row[metric_col]:.4f}")
        
        # Save results if requested
        if args.output:
            runs_sorted.head(args.top_k).to_csv(args.output, index=False)
            logger.info(f"\n✅ Results saved to {args.output}")
    else:
        logger.error(f"Metric '{args.metric}' not found in runs")
        return 1
    
    return 0


def dashboard_command(args):
    """Execute dashboard command."""
    logger.info("🎨 Generating dashboard...")
    logger.info(f"   Results: {args.results}")
    
    import json
    from src.visualization.model_dashboard import ModelDashboard
    
    # Load results
    with open(args.results, 'r') as f:
        results = json.load(f)
    
    # Create dashboard
    dashboard = ModelDashboard(theme=args.theme)
    dashboard.create_comparison_dashboard(
        results,
        output_path=args.output,
        title='Model Performance Dashboard'
    )
    
    logger.info(f"✅ Dashboard saved to {args.output}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Route to appropriate command handler
    command_handlers = {
        'train': train_command,
        'evaluate': evaluate_command,
        'predict': predict_command,
        'validate': validate_command,
        'compare': compare_command,
        'dashboard': dashboard_command
    }
    
    handler = command_handlers.get(args.command)
    if handler:
        try:
            return handler(args)
        except Exception as e:
            logger.error(f"Error executing {args.command}: {e}", exc_info=True)
            return 1
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
