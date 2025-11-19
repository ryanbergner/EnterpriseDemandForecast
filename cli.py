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
        help='Path to the training dataset (CSV/Parquet or ABFS/DBFS URI)'
    )
    train_parser.add_argument(
        '--format',
        choices=['m5_wide', 'long'],
        default='m5_wide',
        help="Input format: 'm5_wide' for Kaggle layout, 'long' for tidy daily data"
    )
    train_parser.add_argument(
        '--calendar-start',
        default='2011-01-29',
        help="Calendar start date corresponding to d_1 when --format=m5_wide"
    )
    train_parser.add_argument(
        '--date-column',
        default='date',
        help='Date column name when --format=long'
    )
    train_parser.add_argument(
        '--product-column',
        default='item_id',
        help='Product identifier column when --format=long'
    )
    train_parser.add_argument(
        '--quantity-column',
        default='demand',
        help='Quantity column when --format=long'
    )
    train_parser.add_argument(
        '--experiment-name',
        default='M5_CLI_Training',
        help='MLflow experiment name'
    )
    train_parser.add_argument(
        '--mlflow-uri',
        help='Optional MLflow tracking URI'
    )
    train_parser.add_argument(
        '--sample-items',
        type=int,
        default=0,
        help='Optional cap on number of items to train on (0 = use all)'
    )
    train_parser.add_argument(
        '--min-orders',
        type=int,
        default=5,
        help='Minimum number of non-zero order months per item'
    )
    train_parser.add_argument(
        '--train-quantile',
        type=float,
        default=0.8,
        help='Quantile used for the chronological train/test split'
    )
    train_parser.add_argument(
        '--feature-columns',
        help='Comma-separated list of feature columns. Defaults to heuristics if omitted'
    )
    train_parser.add_argument(
        '--requirements-path',
        default='requirements.txt',
        help='Requirements file used when logging Spark models to MLflow'
    )
    train_parser.add_argument(
        '--disable-stats',
        action='store_true',
        help='Disable StatsForecast models'
    )
    train_parser.add_argument(
        '--disable-ml',
        action='store_true',
        help='Disable Spark ML models'
    )
    train_parser.add_argument(
        '--per-category',
        action='store_true',
        help='Train separate experiments per intermittent demand category'
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
    logger.info("🚀 Starting model training for the M5 dataset family...")
    logger.info(f"   Data source: {args.data}")
    logger.info(f"   Format: {args.format}")

    from pyspark.sql import SparkSession, functions as F
    import mlflow

    spark = (
        SparkSession.builder.appName("TimeSeriesForecasting_CLI")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
    )

    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)

    from src.preprocessing.datasets import load_custom_daily_dataset, load_m5_wide_dataset
    from src.model_training.training_workflow import (
        TrainingConfig,
        prepare_feature_frame,
        run_training_workflow,
    )

    if args.format == 'm5_wide':
        sdf = load_m5_wide_dataset(
            spark,
            path=args.data,
            sample_items=args.sample_items if args.sample_items > 0 else None,
            calendar_start=args.calendar_start,
        )
        date_column = "OrderDate"
        product_column = "item_id"
        quantity_column = "Quantity"
    else:
        sdf = load_custom_daily_dataset(
            spark,
            path=args.data,
            date_column=args.date_column,
            product_column=args.product_column,
            quantity_column=args.quantity_column,
        )
        date_column = "OrderDate"
        product_column = "item_id"
        quantity_column = "Quantity"

    feature_columns = None
    if args.feature_columns:
        feature_columns = [col.strip() for col in args.feature_columns.split(",") if col.strip()]

    cfg = TrainingConfig(
        experiment_name=args.experiment_name,
        date_column=date_column,
        product_id_column=product_column,
        quantity_column=quantity_column,
        min_total_orders=args.min_orders,
        sample_products=args.sample_items if args.sample_items > 0 else None,
        feature_columns=feature_columns,
        train_quantile=args.train_quantile,
        requirements_path=args.requirements_path,
        enable_stats=not args.disable_stats,
        enable_ml=not args.disable_ml,
    )

    df_feat = prepare_feature_frame(sdf, cfg)

    if args.per_category and "product_category" in df_feat.columns:
        categories = [row[0] for row in df_feat.select("product_category").distinct().collect()]
        logger.info(f"Training per category: {categories}")
        for category in categories:
            segment_df = df_feat.filter(F.col("product_category") == category)
            if segment_df.count() == 0:
                continue
            run_training_workflow(segment_df, cfg, segment_label=category)
    else:
        run_training_workflow(df_feat, cfg)

    logger.info("✅ Training complete. Check MLflow for run details.")
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
