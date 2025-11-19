"""
Time Series Cross-Validation Module

Implements proper time-series cross-validation strategies:
- Expanding window (train on increasing historical data)
- Sliding window (fixed training window size)
- Prevents data leakage by respecting temporal ordering
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, max as spark_max, min as spark_min
from typing import List, Tuple, Dict, Any, Callable
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesCV:
    """
    Time series cross-validation with expanding or sliding window strategies.
    
    Example usage:
        cv = TimeSeriesCV(n_splits=5, strategy='expanding')
        for fold_num, (train_df, test_df) in enumerate(cv.split(df, date_col='MonthEndDate')):
            print(f"Fold {fold_num}: Train size={train_df.count()}, Test size={test_df.count()}")
            # Train and evaluate model
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        strategy: str = 'expanding',
        test_size: int = 1,
        min_train_size: int = 12,
        gap: int = 0
    ):
        """
        Args:
            n_splits: Number of cross-validation folds
            strategy: 'expanding' (growing train set) or 'sliding' (fixed train size)
            test_size: Number of time periods in test set (default 1 month)
            min_train_size: Minimum number of periods in training set (default 12 months)
            gap: Number of periods to skip between train and test (default 0)
        """
        self.n_splits = n_splits
        self.strategy = strategy
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.gap = gap
        
        if strategy not in ['expanding', 'sliding']:
            raise ValueError("strategy must be 'expanding' or 'sliding'")
    
    def split(
        self,
        df: DataFrame,
        date_col: str = 'MonthEndDate'
    ) -> List[Tuple[DataFrame, DataFrame]]:
        """
        Generate train/test splits respecting temporal ordering.
        
        Args:
            df: Input DataFrame with time series data
            date_col: Name of the date column
        
        Yields:
            Tuples of (train_df, test_df) for each fold
        """
        # Get sorted unique dates
        dates = [row[0] for row in df.select(date_col).distinct().orderBy(date_col).collect()]
        n_periods = len(dates)
        
        logger.info(f"📅 Time Series CV: {n_periods} time periods found")
        logger.info(f"   Strategy: {self.strategy}, Splits: {self.n_splits}, Test size: {self.test_size}")
        
        if n_periods < self.min_train_size + self.test_size:
            raise ValueError(
                f"Insufficient data: {n_periods} periods available, but need at least "
                f"{self.min_train_size + self.test_size} (min_train_size + test_size)"
            )
        
        # Calculate split points
        splits = []
        
        if self.strategy == 'expanding':
            # Expanding window: train size grows with each fold
            # Start with min_train_size, increase incrementally
            available_periods = n_periods - self.min_train_size - self.test_size
            step = max(1, available_periods // self.n_splits)
            
            for i in range(self.n_splits):
                train_end_idx = self.min_train_size + (i * step) + self.gap
                test_start_idx = train_end_idx + self.gap
                test_end_idx = test_start_idx + self.test_size
                
                if test_end_idx > n_periods:
                    break
                
                splits.append((0, train_end_idx, test_start_idx, test_end_idx))
        
        else:  # sliding window
            # Sliding window: train size stays constant
            train_window_size = self.min_train_size
            available_periods = n_periods - train_window_size - self.test_size
            step = max(1, available_periods // self.n_splits)
            
            for i in range(self.n_splits):
                train_start_idx = i * step
                train_end_idx = train_start_idx + train_window_size
                test_start_idx = train_end_idx + self.gap
                test_end_idx = test_start_idx + self.test_size
                
                if test_end_idx > n_periods:
                    break
                
                splits.append((train_start_idx, train_end_idx, test_start_idx, test_end_idx))
        
        logger.info(f"   Generated {len(splits)} valid splits")
        
        # Generate DataFrames for each split
        for fold_num, (train_start, train_end, test_start, test_end) in enumerate(splits):
            train_start_date = dates[train_start]
            train_end_date = dates[train_end - 1]
            test_start_date = dates[test_start]
            test_end_date = dates[test_end - 1]
            
            logger.info(
                f"\n   Fold {fold_num + 1}/{len(splits)}: "
                f"Train=[{train_start_date} to {train_end_date}], "
                f"Test=[{test_start_date} to {test_end_date}]"
            )
            
            train_df = df.filter(
                (col(date_col) >= train_start_date) & 
                (col(date_col) <= train_end_date)
            )
            
            test_df = df.filter(
                (col(date_col) >= test_start_date) & 
                (col(date_col) <= test_end_date)
            )
            
            yield train_df, test_df


def cross_validate_model(
    df: DataFrame,
    train_fn: Callable,
    evaluate_fn: Callable,
    date_col: str = 'MonthEndDate',
    n_splits: int = 5,
    strategy: str = 'expanding',
    **cv_kwargs
) -> Dict[str, Any]:
    """
    Perform cross-validation and aggregate results.
    
    Args:
        df: Input DataFrame
        train_fn: Function that trains model, signature: train_fn(train_df) -> model
        evaluate_fn: Function that evaluates model, signature: evaluate_fn(model, test_df) -> dict of metrics
        date_col: Date column name
        n_splits: Number of CV folds
        strategy: 'expanding' or 'sliding'
        **cv_kwargs: Additional arguments for TimeSeriesCV
    
    Returns:
        Dictionary with averaged metrics and per-fold results
    """
    cv = TimeSeriesCV(n_splits=n_splits, strategy=strategy, **cv_kwargs)
    
    fold_results = []
    
    for fold_num, (train_df, test_df) in enumerate(cv.split(df, date_col)):
        logger.info(f"\n🔄 Training Fold {fold_num + 1}/{n_splits}...")
        
        # Train model
        model = train_fn(train_df)
        
        # Evaluate model
        metrics = evaluate_fn(model, test_df)
        metrics['fold'] = fold_num + 1
        fold_results.append(metrics)
        
        logger.info(f"   Fold {fold_num + 1} Metrics: {metrics}")
    
    # Aggregate results
    if not fold_results:
        return {"error": "No valid folds generated"}
    
    # Calculate mean and std for each metric
    metric_names = [k for k in fold_results[0].keys() if k != 'fold' and isinstance(fold_results[0][k], (int, float))]
    
    aggregated = {
        'n_folds': len(fold_results),
        'fold_results': fold_results
    }
    
    for metric_name in metric_names:
        values = [fold[metric_name] for fold in fold_results if fold[metric_name] is not None]
        if values:
            import statistics
            aggregated[f'{metric_name}_mean'] = statistics.mean(values)
            if len(values) > 1:
                aggregated[f'{metric_name}_std'] = statistics.stdev(values)
            else:
                aggregated[f'{metric_name}_std'] = 0.0
    
    logger.info(f"\n✅ Cross-Validation Complete!")
    logger.info(f"   Averaged Metrics:")
    for metric_name in metric_names:
        mean_key = f'{metric_name}_mean'
        std_key = f'{metric_name}_std'
        if mean_key in aggregated:
            logger.info(f"   {metric_name}: {aggregated[mean_key]:.4f} ± {aggregated.get(std_key, 0):.4f}")
    
    return aggregated


class WalkForwardValidator:
    """
    Walk-forward validation for time series.
    Simulates production scenario: train on all past data, predict next period.
    
    Example:
        validator = WalkForwardValidator(initial_train_size=24, refit_frequency=3)
        results = validator.validate(df, train_fn, predict_fn, date_col='MonthEndDate')
    """
    
    def __init__(
        self,
        initial_train_size: int = 24,
        refit_frequency: int = 1,
        horizon: int = 1
    ):
        """
        Args:
            initial_train_size: Initial training window size (in periods)
            refit_frequency: How often to retrain (1 = every period, 3 = every 3 periods)
            horizon: Forecast horizon (number of periods ahead)
        """
        self.initial_train_size = initial_train_size
        self.refit_frequency = refit_frequency
        self.horizon = horizon
    
    def validate(
        self,
        df: DataFrame,
        train_fn: Callable,
        predict_fn: Callable,
        date_col: str = 'MonthEndDate'
    ) -> List[Dict[str, Any]]:
        """
        Perform walk-forward validation.
        
        Args:
            df: Input DataFrame
            train_fn: Function to train model: train_fn(train_df) -> model
            predict_fn: Function to make predictions: predict_fn(model, test_df) -> predictions_df
            date_col: Date column name
        
        Returns:
            List of dictionaries with predictions and actuals for each step
        """
        dates = [row[0] for row in df.select(date_col).distinct().orderBy(date_col).collect()]
        n_periods = len(dates)
        
        logger.info(f"🚶 Walk-Forward Validation:")
        logger.info(f"   Total periods: {n_periods}")
        logger.info(f"   Initial train: {self.initial_train_size}")
        logger.info(f"   Refit frequency: {self.refit_frequency}")
        
        results = []
        model = None
        
        for i in range(self.initial_train_size, n_periods - self.horizon + 1):
            # Determine if we need to retrain
            should_refit = (i == self.initial_train_size) or \
                          ((i - self.initial_train_size) % self.refit_frequency == 0)
            
            if should_refit:
                # Train on all data up to period i
                train_end_date = dates[i - 1]
                train_df = df.filter(col(date_col) <= train_end_date)
                logger.info(f"   Period {i}: Retraining on data up to {train_end_date}")
                model = train_fn(train_df)
            
            # Predict for period i + horizon
            test_date = dates[i + self.horizon - 1]
            test_df = df.filter(col(date_col) == test_date)
            
            predictions_df = predict_fn(model, test_df)
            
            results.append({
                'test_date': test_date,
                'train_end_date': dates[i - 1],
                'predictions_df': predictions_df,
                'refitted': should_refit
            })
        
        logger.info(f"✅ Walk-forward validation complete: {len(results)} predictions made")
        return results
