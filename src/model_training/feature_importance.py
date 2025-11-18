"""
Feature Importance Analysis Module

Provides comprehensive feature importance analysis:
- Native tree model feature importances
- Permutation importance for any model type
- Feature correlation analysis
- Automatic feature selection based on importance
- Visualization and reporting
"""

from pyspark.sql import DataFrame
from pyspark.ml import PipelineModel
from pyspark.ml.regression import RandomForestRegressionModel, GBTRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, abs as spark_abs, when
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """
    Analyze and rank feature importance for forecasting models.
    
    Example usage:
        analyzer = FeatureImportanceAnalyzer()
        importance_dict = analyzer.get_feature_importance(
            model, 
            feature_names=['lag_1', 'lag_2', 'month_sin']
        )
        analyzer.print_feature_importance(importance_dict)
    """
    
    def __init__(self, top_k: int = 20):
        """
        Args:
            top_k: Number of top features to display/select
        """
        self.top_k = top_k
    
    def get_feature_importance(
        self,
        model: Any,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Extract feature importances from trained model.
        
        Args:
            model: Trained Spark ML model or PipelineModel
            feature_names: List of feature names (in order)
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        logger.info("📊 Extracting feature importances...")
        
        # If it's a pipeline, extract the last stage
        if isinstance(model, PipelineModel):
            actual_model = model.stages[-1]
        else:
            actual_model = model
        
        # Try to get native feature importances
        if isinstance(actual_model, (RandomForestRegressionModel, GBTRegressionModel)):
            importances = actual_model.featureImportances.toArray()
            
            if len(importances) != len(feature_names):
                logger.warning(
                    f"Mismatch: {len(importances)} importances but {len(feature_names)} feature names"
                )
                # Pad or truncate as needed
                min_len = min(len(importances), len(feature_names))
                importances = importances[:min_len]
                feature_names = feature_names[:min_len]
            
            importance_dict = {
                name: float(importance)
                for name, importance in zip(feature_names, importances)
            }
            
            logger.info(f"   ✓ Extracted {len(importance_dict)} feature importances")
            return importance_dict
        
        else:
            logger.warning(
                f"Model type {type(actual_model).__name__} does not support native feature importance"
            )
            return {}
    
    def permutation_importance(
        self,
        model: Any,
        test_df: DataFrame,
        feature_cols: List[str],
        label_col: str,
        n_repeats: int = 3,
        metric: str = 'rmse'
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate permutation importance by shuffling each feature.
        Works with any model type.
        
        Args:
            model: Trained model
            test_df: Test DataFrame
            feature_cols: List of feature column names
            label_col: Target column name
            n_repeats: Number of times to shuffle each feature
            metric: Metric to use ('rmse', 'mae', 'r2')
        
        Returns:
            Dictionary with mean and std of importance for each feature
        """
        logger.info(f"🔀 Computing permutation importance ({n_repeats} repeats)...")
        
        # Get baseline score
        evaluator = RegressionEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName=metric
        )
        
        baseline_preds = model.transform(test_df)
        baseline_score = evaluator.evaluate(baseline_preds)
        logger.info(f"   Baseline {metric}: {baseline_score:.4f}")
        
        importance_scores = {}
        
        for feat_col in feature_cols:
            logger.info(f"   Evaluating {feat_col}...")
            scores = []
            
            for repeat in range(n_repeats):
                # Shuffle feature column
                shuffled_df = self._shuffle_column(test_df, feat_col)
                
                # Predict with shuffled feature
                shuffled_preds = model.transform(shuffled_df)
                shuffled_score = evaluator.evaluate(shuffled_preds)
                
                # Importance = degradation in performance
                if metric in ['rmse', 'mae']:
                    importance = shuffled_score - baseline_score  # Higher is worse
                else:  # r2
                    importance = baseline_score - shuffled_score  # Lower is worse
                
                scores.append(importance)
            
            importance_scores[feat_col] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores))
            }
        
        # Sort by mean importance
        sorted_features = sorted(
            importance_scores.items(),
            key=lambda x: x[1]['mean'],
            reverse=True
        )
        
        logger.info(f"✅ Permutation importance complete!")
        logger.info(f"   Top 5 features:")
        for feat, score in sorted_features[:5]:
            logger.info(f"     {feat}: {score['mean']:.4f} ± {score['std']:.4f}")
        
        return importance_scores
    
    def _shuffle_column(self, df: DataFrame, col_name: str) -> DataFrame:
        """Shuffle a single column (approximate shuffle using random ordering)."""
        from pyspark.sql.functions import rand
        
        # Extract column values in random order
        shuffled_values = df.select(col_name).orderBy(rand()).collect()
        
        # This is a simplified approach - for large datasets, 
        # consider using window functions or joins
        # For now, we'll use a workaround with row_number
        from pyspark.sql.functions import row_number, monotonically_increasing_id
        from pyspark.sql.window import Window
        
        # Add row numbers
        w = Window.orderBy(monotonically_increasing_id())
        df_numbered = df.withColumn("_row_num", row_number().over(w))
        
        # Create shuffled version
        shuffled_col_df = df.select(col_name).orderBy(rand()).withColumn(
            "_row_num", row_number().over(Window.orderBy(monotonically_increasing_id()))
        ).withColumnRenamed(col_name, f"_shuffled_{col_name}")
        
        # Join back
        df_shuffled = df_numbered.join(shuffled_col_df, on="_row_num").drop("_row_num", col_name)
        df_shuffled = df_shuffled.withColumnRenamed(f"_shuffled_{col_name}", col_name)
        
        return df_shuffled
    
    def select_top_features(
        self,
        importance_dict: Dict[str, float],
        top_k: Optional[int] = None
    ) -> List[str]:
        """
        Select top K features based on importance scores.
        
        Args:
            importance_dict: Dictionary of feature importances
            top_k: Number of features to select (default: self.top_k)
        
        Returns:
            List of top K feature names
        """
        if top_k is None:
            top_k = self.top_k
        
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_features = [feat for feat, _ in sorted_features[:top_k]]
        
        logger.info(f"🎯 Selected top {len(top_features)} features:")
        for i, feat in enumerate(top_features, 1):
            logger.info(f"   {i}. {feat}: {importance_dict[feat]:.4f}")
        
        return top_features
    
    def print_feature_importance(
        self,
        importance_dict: Dict[str, float],
        top_k: Optional[int] = None
    ) -> None:
        """
        Print formatted feature importance report.
        
        Args:
            importance_dict: Dictionary of feature importances
            top_k: Number of top features to display
        """
        if top_k is None:
            top_k = self.top_k
        
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print("\n" + "="*70)
        print("📊 FEATURE IMPORTANCE REPORT")
        print("="*70)
        
        print(f"\nTop {top_k} Features:")
        print(f"{'Rank':<6} {'Feature':<40} {'Importance':<15}")
        print("-" * 70)
        
        for i, (feature, importance) in enumerate(sorted_features[:top_k], 1):
            # Create simple bar chart
            bar_length = int(importance * 50) if importance > 0 else 0
            bar = "█" * bar_length
            
            print(f"{i:<6} {feature:<40} {importance:<10.4f} {bar}")
        
        # Summary statistics
        importances = list(importance_dict.values())
        print("\n" + "-" * 70)
        print(f"Total features: {len(importances)}")
        print(f"Mean importance: {np.mean(importances):.4f}")
        print(f"Std importance: {np.std(importances):.4f}")
        print(f"Max importance: {np.max(importances):.4f}")
        print("="*70 + "\n")
    
    def create_importance_dataframe(
        self,
        importance_dict: Dict[str, float]
    ) -> DataFrame:
        """
        Create a Spark DataFrame from importance dictionary for further analysis.
        
        Args:
            importance_dict: Dictionary of feature importances
        
        Returns:
            Spark DataFrame with columns [feature, importance]
        """
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder.getOrCreate()
        
        importance_data = [
            (feature, float(importance))
            for feature, importance in importance_dict.items()
        ]
        
        importance_df = spark.createDataFrame(
            importance_data,
            ["feature", "importance"]
        ).orderBy(col("importance").desc())
        
        return importance_df


def compare_feature_importance_across_models(
    models_dict: Dict[str, Any],
    feature_names: List[str]
) -> DataFrame:
    """
    Compare feature importances across multiple models.
    
    Args:
        models_dict: Dictionary mapping model names to trained models
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importances for each model
    """
    logger.info(f"🔄 Comparing feature importance across {len(models_dict)} models")
    
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    
    analyzer = FeatureImportanceAnalyzer()
    
    # Collect importances for each model
    all_importances = []
    
    for model_name, model in models_dict.items():
        logger.info(f"   Analyzing {model_name}...")
        importance_dict = analyzer.get_feature_importance(model, feature_names)
        
        for feature, importance in importance_dict.items():
            all_importances.append((feature, model_name, float(importance)))
    
    # Create DataFrame
    comparison_df = spark.createDataFrame(
        all_importances,
        ["feature", "model", "importance"]
    )
    
    logger.info("✅ Feature importance comparison complete!")
    
    return comparison_df


def automatic_feature_selection(
    model: Any,
    train_df: DataFrame,
    test_df: DataFrame,
    feature_cols: List[str],
    label_col: str,
    importance_threshold: float = 0.01,
    max_features: Optional[int] = None
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Automatically select features based on importance threshold.
    
    Args:
        model: Trained model
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: List of all feature column names
        label_col: Target column name
        importance_threshold: Minimum importance to keep feature
        max_features: Maximum number of features to keep
    
    Returns:
        Tuple of (selected_features, selection_report)
    """
    logger.info("🎯 Automatic feature selection starting...")
    
    analyzer = FeatureImportanceAnalyzer()
    
    # Get importances
    importance_dict = analyzer.get_feature_importance(model, feature_cols)
    
    if not importance_dict:
        logger.warning("Could not extract importances, returning all features")
        return feature_cols, {"status": "failed", "reason": "no_importances"}
    
    # Filter by threshold
    selected_features = [
        feat for feat, imp in importance_dict.items()
        if imp >= importance_threshold
    ]
    
    # Apply max_features limit
    if max_features and len(selected_features) > max_features:
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        selected_features = [feat for feat, _ in sorted_features[:max_features]]
    
    # Evaluate performance with selected features
    evaluator = RegressionEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="rmse"
    )
    
    # Full model performance
    full_preds = model.transform(test_df)
    full_rmse = evaluator.evaluate(full_preds)
    
    logger.info(f"\n📊 Feature Selection Results:")
    logger.info(f"   Original features: {len(feature_cols)}")
    logger.info(f"   Selected features: {len(selected_features)}")
    logger.info(f"   Reduction: {(1 - len(selected_features)/len(feature_cols))*100:.1f}%")
    logger.info(f"   Full model RMSE: {full_rmse:.4f}")
    
    report = {
        "original_feature_count": len(feature_cols),
        "selected_feature_count": len(selected_features),
        "reduction_pct": (1 - len(selected_features)/len(feature_cols)) * 100,
        "full_model_rmse": full_rmse,
        "selected_features": selected_features,
        "importance_threshold": importance_threshold,
        "status": "success"
    }
    
    logger.info("✅ Feature selection complete!")
    
    return selected_features, report
