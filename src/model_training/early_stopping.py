"""
Early Stopping Module for Tree-Based Models

Implements early stopping to prevent overfitting in iterative models:
- Validation-based stopping for GBT and RandomForest
- Configurable patience and improvement thresholds
- Automatic train/validation splitting
- Monitoring and logging of validation metrics
"""

from pyspark.sql import DataFrame
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from typing import Optional, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping callback for iterative models.
    
    Monitors validation metrics and stops training when improvement plateaus.
    
    Example usage:
        early_stop = EarlyStopping(patience=3, min_delta=0.001)
        model = early_stop.train_with_early_stopping(
            gbt_model,
            train_df,
            feature_cols,
            label_col='target',
            validation_split=0.2
        )
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        metric: str = 'rmse',
        restore_best_weights: bool = True
    ):
        """
        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change in metric to qualify as improvement
            metric: Metric to monitor ('rmse', 'mae', 'r2')
            restore_best_weights: Whether to use best iteration (not last)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.restore_best_weights = restore_best_weights
        self.best_score = float('inf') if metric in ['rmse', 'mae'] else float('-inf')
        self.best_iteration = 0
        self.wait = 0
    
    def train_with_early_stopping(
        self,
        model,
        train_df: DataFrame,
        feature_cols: list,
        label_col: str,
        validation_split: float = 0.2,
        max_iterations: int = 100,
        **model_kwargs
    ) -> Any:
        """
        Train model with early stopping based on validation performance.
        
        Args:
            model: Spark ML model (GBTRegressor or RandomForestRegressor)
            train_df: Training DataFrame
            feature_cols: List of feature column names
            label_col: Target column name
            validation_split: Fraction of training data for validation
            max_iterations: Maximum number of iterations
            **model_kwargs: Additional model parameters
        
        Returns:
            Trained model (at best iteration if restore_best_weights=True)
        """
        logger.info(f"🚀 Training with early stopping (patience={self.patience}, metric={self.metric})")
        
        # Split into train and validation
        train_data, val_data = self._split_data(train_df, validation_split)
        logger.info(f"   Train size: {train_data.count()}, Validation size: {val_data.count()}")
        
        # Setup model with configurable iterations
        if isinstance(model, GBTRegressor):
            return self._train_gbt_with_early_stopping(
                model, train_data, val_data, feature_cols, label_col, max_iterations, **model_kwargs
            )
        elif isinstance(model, RandomForestRegressor):
            return self._train_rf_with_early_stopping(
                model, train_data, val_data, feature_cols, label_col, max_iterations, **model_kwargs
            )
        else:
            logger.warning(f"Early stopping not implemented for {type(model).__name__}, training normally")
            return self._train_normal(model, train_df, feature_cols, label_col)
    
    def _split_data(
        self,
        df: DataFrame,
        validation_split: float
    ) -> Tuple[DataFrame, DataFrame]:
        """Split data chronologically for time series."""
        # For time series, use last validation_split% as validation
        total_count = df.count()
        train_count = int(total_count * (1 - validation_split))
        
        # Use limit and subtract for chronological split
        # This assumes data is already sorted by time
        train_df = df.limit(train_count)
        val_df = df.subtract(train_df)
        
        return train_df, val_df
    
    def _train_gbt_with_early_stopping(
        self,
        model: GBTRegressor,
        train_df: DataFrame,
        val_df: DataFrame,
        feature_cols: list,
        label_col: str,
        max_iterations: int,
        **model_kwargs
    ):
        """Train GBT with early stopping."""
        logger.info("   Training GBT with early stopping...")
        
        # Create assembler
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        
        train_assembled = assembler.transform(train_df)
        val_assembled = assembler.transform(val_df)
        
        # Configure model
        model.setLabelCol(label_col)
        model.setFeaturesCol("features")
        model.setMaxIter(max_iterations)
        
        # Apply additional kwargs
        for key, value in model_kwargs.items():
            if hasattr(model, f'set{key.capitalize()}'):
                getattr(model, f'set{key.capitalize()}')(value)
        
        # Iteratively train and evaluate
        best_model = None
        evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName=self.metric
        )
        
        # Train in increments
        for iteration in range(5, max_iterations + 1, 5):  # Check every 5 iterations
            model.setMaxIter(iteration)
            current_model = model.fit(train_assembled)
            
            # Evaluate on validation set
            val_preds = current_model.transform(val_assembled)
            val_score = evaluator.evaluate(val_preds)
            
            logger.info(f"   Iteration {iteration}: Validation {self.metric}={val_score:.4f}")
            
            # Check for improvement
            improved = self._check_improvement(val_score)
            
            if improved:
                best_model = current_model
                self.best_iteration = iteration
                self.wait = 0
                logger.info(f"   ✓ Improvement detected, best iteration: {iteration}")
            else:
                self.wait += 1
                logger.info(f"   No improvement for {self.wait} checks")
            
            # Early stopping
            if self.wait >= self.patience:
                logger.info(f"   🛑 Early stopping triggered at iteration {iteration}")
                break
        
        if self.restore_best_weights and best_model is not None:
            logger.info(f"   ✅ Returning model from best iteration: {self.best_iteration}")
            return best_model
        else:
            return current_model
    
    def _train_rf_with_early_stopping(
        self,
        model: RandomForestRegressor,
        train_df: DataFrame,
        val_df: DataFrame,
        feature_cols: list,
        label_col: str,
        max_iterations: int,
        **model_kwargs
    ):
        """Train Random Forest with early stopping based on number of trees."""
        logger.info("   Training Random Forest with early stopping...")
        
        # Create assembler
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        
        train_assembled = assembler.transform(train_df)
        val_assembled = assembler.transform(val_df)
        
        # Configure model
        model.setLabelCol(label_col)
        model.setFeaturesCol("features")
        
        # Apply additional kwargs
        for key, value in model_kwargs.items():
            if hasattr(model, f'set{key.capitalize()}'):
                getattr(model, f'set{key.capitalize()}')(value)
        
        best_model = None
        evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName=self.metric
        )
        
        # Train with increasing number of trees
        tree_increments = [10, 20, 30, 50, 75, 100, 150, 200]
        tree_increments = [t for t in tree_increments if t <= max_iterations]
        
        for num_trees in tree_increments:
            model.setNumTrees(num_trees)
            current_model = model.fit(train_assembled)
            
            # Evaluate
            val_preds = current_model.transform(val_assembled)
            val_score = evaluator.evaluate(val_preds)
            
            logger.info(f"   Trees={num_trees}: Validation {self.metric}={val_score:.4f}")
            
            # Check improvement
            improved = self._check_improvement(val_score)
            
            if improved:
                best_model = current_model
                self.best_iteration = num_trees
                self.wait = 0
                logger.info(f"   ✓ Improvement detected, best trees: {num_trees}")
            else:
                self.wait += 1
                logger.info(f"   No improvement for {self.wait} checks")
            
            # Early stopping
            if self.wait >= self.patience:
                logger.info(f"   🛑 Early stopping triggered at {num_trees} trees")
                break
        
        if self.restore_best_weights and best_model is not None:
            logger.info(f"   ✅ Returning model with best trees: {self.best_iteration}")
            return best_model
        else:
            return current_model
    
    def _check_improvement(self, current_score: float) -> bool:
        """Check if current score is an improvement over best score."""
        if self.metric in ['rmse', 'mae']:
            # Lower is better
            improved = current_score < (self.best_score - self.min_delta)
            if improved:
                self.best_score = current_score
        else:  # r2
            # Higher is better
            improved = current_score > (self.best_score + self.min_delta)
            if improved:
                self.best_score = current_score
        
        return improved
    
    def _train_normal(
        self,
        model,
        train_df: DataFrame,
        feature_cols: list,
        label_col: str
    ):
        """Fallback: train model normally without early stopping."""
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        
        pipeline = Pipeline(stages=[assembler, model])
        return pipeline.fit(train_df)


def train_with_validation_curve(
    model,
    train_df: DataFrame,
    feature_cols: list,
    label_col: str,
    validation_split: float = 0.2,
    param_name: str = 'maxIter',
    param_values: list = [10, 20, 50, 100, 200]
) -> Dict[str, Any]:
    """
    Generate validation curve for hyperparameter tuning with early stopping insights.
    
    Args:
        model: Spark ML model
        train_df: Training DataFrame
        feature_cols: Feature column names
        label_col: Target column name
        validation_split: Validation split ratio
        param_name: Parameter to vary
        param_values: List of parameter values to test
    
    Returns:
        Dictionary with validation curve results
    """
    logger.info(f"📈 Generating validation curve for {param_name}")
    
    # Split data
    total_count = train_df.count()
    train_count = int(total_count * (1 - validation_split))
    train_data = train_df.limit(train_count)
    val_data = train_df.subtract(train_data)
    
    # Assembler
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )
    
    train_assembled = assembler.transform(train_data)
    val_assembled = assembler.transform(val_data)
    
    model.setLabelCol(label_col)
    model.setFeaturesCol("features")
    
    evaluator = RegressionEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName='rmse'
    )
    
    results = {
        'param_values': [],
        'train_scores': [],
        'val_scores': []
    }
    
    for param_val in param_values:
        # Set parameter
        if hasattr(model, f'set{param_name}'):
            getattr(model, f'set{param_name}')(param_val)
        
        # Train
        fitted_model = model.fit(train_assembled)
        
        # Evaluate on both train and validation
        train_preds = fitted_model.transform(train_assembled)
        val_preds = fitted_model.transform(val_assembled)
        
        train_score = evaluator.evaluate(train_preds)
        val_score = evaluator.evaluate(val_preds)
        
        results['param_values'].append(param_val)
        results['train_scores'].append(train_score)
        results['val_scores'].append(val_score)
        
        logger.info(f"   {param_name}={param_val}: Train RMSE={train_score:.4f}, Val RMSE={val_score:.4f}")
    
    # Find best parameter
    best_idx = results['val_scores'].index(min(results['val_scores']))
    results['best_param_value'] = results['param_values'][best_idx]
    results['best_val_score'] = results['val_scores'][best_idx]
    
    logger.info(f"✅ Best {param_name}: {results['best_param_value']} (Val RMSE={results['best_val_score']:.4f})")
    
    return results
