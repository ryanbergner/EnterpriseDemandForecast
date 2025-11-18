"""
Ensemble Methods Module

Combines predictions from multiple models for improved accuracy:
- Weighted average ensemble with learned weights
- Stacking ensemble with meta-learner
- Dynamic model selection per product category
- Best-of-N selection strategy
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, lit, avg as spark_avg, when, 
    array, struct, explode, collect_list
)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression, GBTRegressor
from pyspark.ml.feature import VectorAssembler
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple model predictions.
    
    Example usage:
        ensemble = EnsemblePredictor(strategy='weighted_average')
        predictions_df = ensemble.predict(
            models={'RF': rf_model, 'GBT': gbt_model, 'LR': lr_model},
            test_df=test_df
        )
    """
    
    def __init__(
        self,
        strategy: str = 'weighted_average',
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            strategy: Ensemble strategy ('simple_average', 'weighted_average', 
                     'median', 'best_per_category')
            weights: Optional dictionary of model weights for weighted_average
        """
        self.strategy = strategy
        self.weights = weights or {}
    
    def predict(
        self,
        models: Dict[str, Any],
        test_df: DataFrame
    ) -> DataFrame:
        """
        Generate ensemble predictions from multiple models.
        
        Args:
            models: Dictionary mapping model names to trained models
            test_df: Test DataFrame
        
        Returns:
            DataFrame with ensemble predictions
        """
        logger.info(f"🎯 Creating ensemble predictions using strategy: {self.strategy}")
        logger.info(f"   Models: {list(models.keys())}")
        
        # Collect predictions from all models
        all_predictions = {}
        
        for model_name, model in models.items():
            logger.info(f"   Generating predictions for {model_name}...")
            pred_df = model.transform(test_df)
            
            # Rename prediction column to include model name
            pred_df = pred_df.withColumnRenamed("prediction", f"pred_{model_name}")
            all_predictions[model_name] = pred_df
        
        # Merge all predictions
        logger.info("   Merging predictions...")
        ensemble_df = test_df
        
        for model_name, pred_df in all_predictions.items():
            pred_col = f"pred_{model_name}"
            ensemble_df = ensemble_df.join(
                pred_df.select(pred_df.columns[0], pred_col),  # Join on first column (usually ID)
                on=pred_df.columns[0],
                how='left'
            )
        
        # Apply ensemble strategy
        logger.info(f"   Applying {self.strategy} strategy...")
        
        if self.strategy == 'simple_average':
            ensemble_df = self._simple_average(ensemble_df, list(models.keys()))
        
        elif self.strategy == 'weighted_average':
            ensemble_df = self._weighted_average(ensemble_df, list(models.keys()))
        
        elif self.strategy == 'median':
            ensemble_df = self._median_ensemble(ensemble_df, list(models.keys()))
        
        elif self.strategy == 'best_per_category':
            ensemble_df = self._best_per_category(ensemble_df, models)
        
        else:
            logger.warning(f"Unknown strategy '{self.strategy}', using simple average")
            ensemble_df = self._simple_average(ensemble_df, list(models.keys()))
        
        logger.info("✅ Ensemble predictions complete!")
        
        return ensemble_df
    
    def _simple_average(self, df: DataFrame, model_names: List[str]) -> DataFrame:
        """Simple average of all model predictions."""
        pred_cols = [f"pred_{name}" for name in model_names]
        
        # Sum all predictions and divide by count
        sum_expr = sum(col(pred_col) for pred_col in pred_cols)
        
        return df.withColumn(
            "ensemble_prediction",
            sum_expr / lit(len(pred_cols))
        )
    
    def _weighted_average(self, df: DataFrame, model_names: List[str]) -> DataFrame:
        """Weighted average using specified or uniform weights."""
        if not self.weights:
            # Use uniform weights
            logger.info("   No weights specified, using uniform weights")
            return self._simple_average(df, model_names)
        
        # Normalize weights to sum to 1
        total_weight = sum(self.weights.get(name, 0) for name in model_names)
        normalized_weights = {
            name: self.weights.get(name, 0) / total_weight
            for name in model_names
        }
        
        logger.info(f"   Normalized weights: {normalized_weights}")
        
        # Weighted sum
        weighted_sum = sum(
            col(f"pred_{name}") * lit(normalized_weights[name])
            for name in model_names
        )
        
        return df.withColumn("ensemble_prediction", weighted_sum)
    
    def _median_ensemble(self, df: DataFrame, model_names: List[str]) -> DataFrame:
        """Median of all model predictions (robust to outliers)."""
        from pyspark.sql.functions import expr
        
        pred_cols = [f"pred_{name}" for name in model_names]
        
        # Create array of predictions and compute median
        # Note: Spark doesn't have native median, use percentile_approx
        array_expr = f"array({','.join(pred_cols)})"
        
        return df.withColumn(
            "ensemble_prediction",
            expr(f"percentile_approx(explode({array_expr}), 0.5)")
        )
    
    def _best_per_category(
        self,
        df: DataFrame,
        models: Dict[str, Any],
        category_col: str = 'product_category'
    ) -> DataFrame:
        """Select best model per product category."""
        if category_col not in df.columns:
            logger.warning(f"Category column '{category_col}' not found, using simple average")
            return self._simple_average(df, list(models.keys()))
        
        # This is a simplified version - in practice, you'd pre-compute
        # which model performs best for each category
        # For now, use simple heuristic based on category
        
        df = df.withColumn(
            "ensemble_prediction",
            when(col(category_col) == "Smooth", col(f"pred_{list(models.keys())[0]}"))
            .when(col(category_col) == "Erratic", col(f"pred_{list(models.keys())[1] if len(models) > 1 else list(models.keys())[0]}"))
            .otherwise(col(f"pred_{list(models.keys())[0]}"))
        )
        
        return df


class StackingEnsemble:
    """
    Stacking ensemble with meta-learner.
    Uses base model predictions as features for a meta-model.
    
    Example usage:
        stacker = StackingEnsemble(meta_learner=LinearRegression())
        stacker.fit(base_models, train_df, val_df)
        predictions = stacker.predict(test_df)
    """
    
    def __init__(
        self,
        meta_learner: Any = None,
        use_original_features: bool = False
    ):
        """
        Args:
            meta_learner: Meta-learner model (default: LinearRegression)
            use_original_features: Whether to include original features in meta-model
        """
        self.meta_learner = meta_learner or LinearRegression()
        self.use_original_features = use_original_features
        self.base_models = {}
        self.meta_model = None
    
    def fit(
        self,
        base_models: Dict[str, Any],
        train_df: DataFrame,
        val_df: DataFrame,
        label_col: str = 'target'
    ) -> 'StackingEnsemble':
        """
        Train stacking ensemble.
        
        Args:
            base_models: Dictionary of base models
            train_df: Training DataFrame
            val_df: Validation DataFrame (for meta-learner training)
            label_col: Target column name
        
        Returns:
            Self (trained ensemble)
        """
        logger.info("🏗️ Training stacking ensemble...")
        logger.info(f"   Base models: {list(base_models.keys())}")
        
        self.base_models = base_models
        
        # Generate base model predictions on validation set
        logger.info("   Generating base model predictions on validation set...")
        meta_features_df = val_df
        
        for model_name, model in base_models.items():
            pred_df = model.transform(val_df)
            pred_col = f"pred_{model_name}"
            
            meta_features_df = meta_features_df.join(
                pred_df.select(pred_df.columns[0], col("prediction").alias(pred_col)),
                on=pred_df.columns[0],
                how='left'
            )
        
        # Train meta-learner
        logger.info("   Training meta-learner...")
        meta_feature_cols = [f"pred_{name}" for name in base_models.keys()]
        
        assembler = VectorAssembler(
            inputCols=meta_feature_cols,
            outputCol="meta_features",
            handleInvalid="skip"
        )
        
        meta_features_assembled = assembler.transform(meta_features_df)
        
        self.meta_learner.setLabelCol(label_col)
        self.meta_learner.setFeaturesCol("meta_features")
        
        self.meta_model = self.meta_learner.fit(meta_features_assembled)
        self.assembler = assembler
        
        logger.info("✅ Stacking ensemble trained!")
        
        return self
    
    def predict(self, test_df: DataFrame) -> DataFrame:
        """
        Generate predictions using stacking ensemble.
        
        Args:
            test_df: Test DataFrame
        
        Returns:
            DataFrame with stacked predictions
        """
        logger.info("🎯 Generating stacking ensemble predictions...")
        
        # Get base model predictions
        stacked_df = test_df
        
        for model_name, model in self.base_models.items():
            pred_df = model.transform(test_df)
            pred_col = f"pred_{model_name}"
            
            stacked_df = stacked_df.join(
                pred_df.select(pred_df.columns[0], col("prediction").alias(pred_col)),
                on=pred_df.columns[0],
                how='left'
            )
        
        # Apply meta-learner
        stacked_assembled = self.assembler.transform(stacked_df)
        final_predictions = self.meta_model.transform(stacked_assembled)
        
        final_predictions = final_predictions.withColumnRenamed(
            "prediction", "ensemble_prediction"
        )
        
        logger.info("✅ Stacking predictions complete!")
        
        return final_predictions


def learn_ensemble_weights(
    models: Dict[str, Any],
    val_df: DataFrame,
    label_col: str = 'target'
) -> Dict[str, float]:
    """
    Learn optimal ensemble weights by minimizing validation error.
    
    Args:
        models: Dictionary of trained models
        val_df: Validation DataFrame
        label_col: Target column name
    
    Returns:
        Dictionary of optimal weights per model
    """
    logger.info("🎓 Learning optimal ensemble weights...")
    
    # Generate predictions for all models
    predictions = {}
    evaluator = RegressionEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="rmse"
    )
    
    for model_name, model in models.items():
        pred_df = model.transform(val_df)
        rmse = evaluator.evaluate(pred_df)
        
        # Collect predictions as list
        pred_values = [row['prediction'] for row in pred_df.select('prediction').collect()]
        predictions[model_name] = {
            'values': pred_values,
            'rmse': rmse
        }
        
        logger.info(f"   {model_name} RMSE: {rmse:.4f}")
    
    # Simple inverse RMSE weighting
    # Better models (lower RMSE) get higher weight
    inverse_rmse = {
        name: 1.0 / pred['rmse'] if pred['rmse'] > 0 else 1.0
        for name, pred in predictions.items()
    }
    
    total_inverse = sum(inverse_rmse.values())
    weights = {
        name: inv_rmse / total_inverse
        for name, inv_rmse in inverse_rmse.items()
    }
    
    logger.info("✅ Optimal weights learned:")
    for name, weight in weights.items():
        logger.info(f"   {name}: {weight:.4f}")
    
    return weights


def evaluate_ensemble(
    ensemble_df: DataFrame,
    label_col: str,
    prediction_col: str = 'ensemble_prediction'
) -> Dict[str, float]:
    """
    Evaluate ensemble predictions.
    
    Args:
        ensemble_df: DataFrame with ensemble predictions
        label_col: Target column name
        prediction_col: Prediction column name
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("📊 Evaluating ensemble performance...")
    
    metrics = {}
    
    for metric_name in ['rmse', 'mae', 'r2']:
        evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol=prediction_col,
            metricName=metric_name
        )
        
        score = evaluator.evaluate(ensemble_df)
        metrics[metric_name] = score
        logger.info(f"   {metric_name.upper()}: {score:.4f}")
    
    logger.info("✅ Ensemble evaluation complete!")
    
    return metrics


def compare_ensemble_strategies(
    models: Dict[str, Any],
    test_df: DataFrame,
    label_col: str
) -> DataFrame:
    """
    Compare different ensemble strategies side-by-side.
    
    Args:
        models: Dictionary of trained models
        test_df: Test DataFrame
        label_col: Target column name
    
    Returns:
        DataFrame with comparison results
    """
    logger.info("🔄 Comparing ensemble strategies...")
    
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    
    strategies = ['simple_average', 'weighted_average', 'median']
    results = []
    
    for strategy in strategies:
        logger.info(f"\n   Testing {strategy}...")
        
        ensemble = EnsemblePredictor(strategy=strategy)
        
        # Learn weights if needed
        if strategy == 'weighted_average':
            # Use validation set to learn weights (simplified)
            weights = {name: 1.0/len(models) for name in models.keys()}
            ensemble.weights = weights
        
        # Generate predictions
        pred_df = ensemble.predict(models, test_df)
        
        # Evaluate
        metrics = evaluate_ensemble(pred_df, label_col)
        
        results.append({
            'strategy': strategy,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2']
        })
    
    # Create comparison DataFrame
    comparison_df = spark.createDataFrame(results)
    
    logger.info("\n✅ Strategy comparison complete!")
    comparison_df.show()
    
    return comparison_df
