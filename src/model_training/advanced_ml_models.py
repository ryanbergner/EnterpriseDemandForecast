"""
Advanced ML Models Module for Time Series Forecasting

Provides sklearn-compatible models optimized for demand forecasting:
- LightGBM with gradient boosting for fast training and low memory
- XGBoost for high accuracy with regularization
- CatBoost for handling categorical features
- Quantile regressors for uncertainty quantification
- Model training utilities with hyperparameter optimization

These models work with Pandas DataFrames (converted from Spark) for:
- Per-product or per-category model training
- Ensemble model components
- Benchmark comparisons with Spark ML models
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import mlflow

# Suppress warnings during import checks
warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for optional dependencies
_HAS_LIGHTGBM = False
_HAS_XGBOOST = False
_HAS_CATBOOST = False

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    logger.warning("LightGBM not installed. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    logger.warning("XGBoost not installed. Install with: pip install xgboost")

try:
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    logger.warning("CatBoost not installed. Install with: pip install catboost")


@dataclass
class ModelConfig:
    """Configuration for ML model training."""
    model_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    param_grid: Optional[Dict[str, List[Any]]] = None
    cv_folds: int = 5
    early_stopping_rounds: int = 50
    verbose: bool = False


def get_lightgbm_regressor(
    objective: str = 'regression',
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 8,
    num_leaves: int = 31,
    min_child_samples: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 0.1,
    random_state: int = 42,
    **kwargs
) -> Any:
    """
    Create a LightGBM regressor optimized for time series forecasting.
    
    LightGBM advantages for demand forecasting:
    - Fast training on large datasets
    - Low memory usage
    - Handles missing values natively
    - Good for high-cardinality categorical features
    
    Args:
        objective: 'regression' for L2, 'regression_l1' for L1, 
                   'quantile' for quantile regression
        n_estimators: Number of boosting rounds
        learning_rate: Shrinkage rate
        max_depth: Maximum tree depth (-1 for no limit)
        num_leaves: Maximum number of leaves per tree
        min_child_samples: Minimum samples in leaf
        subsample: Row subsampling ratio
        colsample_bytree: Column subsampling ratio
        reg_alpha: L1 regularization
        reg_lambda: L2 regularization
        random_state: Random seed
        **kwargs: Additional LightGBM parameters
    
    Returns:
        LightGBM regressor model
    """
    if not _HAS_LIGHTGBM:
        raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
    
    params = {
        'objective': objective,
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'min_child_samples': min_child_samples,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'random_state': random_state,
        'n_jobs': -1,
        'verbosity': -1,
        **kwargs
    }
    
    return lgb.LGBMRegressor(**params)


def get_lightgbm_quantile_regressor(
    alpha: float = 0.5,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 8,
    random_state: int = 42,
    **kwargs
) -> Any:
    """
    Create a LightGBM quantile regressor for uncertainty quantification.
    
    Useful for generating prediction intervals:
    - alpha=0.5: Median (robust to outliers)
    - alpha=0.1: 10th percentile (lower bound)
    - alpha=0.9: 90th percentile (upper bound)
    
    Args:
        alpha: Quantile to predict (0-1)
        n_estimators: Number of boosting rounds
        learning_rate: Learning rate
        max_depth: Max tree depth
        random_state: Random seed
        **kwargs: Additional parameters
    
    Returns:
        LightGBM quantile regressor
    """
    if not _HAS_LIGHTGBM:
        raise ImportError("LightGBM not installed.")
    
    params = {
        'objective': 'quantile',
        'alpha': alpha,
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'random_state': random_state,
        'n_jobs': -1,
        'verbosity': -1,
        **kwargs
    }
    
    return lgb.LGBMRegressor(**params)


def get_xgboost_regressor(
    objective: str = 'reg:squarederror',
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    max_depth: int = 8,
    min_child_weight: int = 3,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    **kwargs
) -> Any:
    """
    Create an XGBoost regressor for demand forecasting.
    
    XGBoost advantages:
    - Strong regularization (prevents overfitting)
    - Handles missing values
    - Good for medium-sized datasets
    - Robust feature importance
    
    Args:
        objective: 'reg:squarederror' for MSE, 'reg:absoluteerror' for MAE,
                   'reg:quantileerror' for quantile regression
        n_estimators: Number of boosting rounds
        learning_rate: Step size shrinkage
        max_depth: Maximum tree depth
        min_child_weight: Minimum sum of instance weight in child
        subsample: Row subsampling ratio
        colsample_bytree: Column subsampling ratio
        reg_alpha: L1 regularization
        reg_lambda: L2 regularization
        random_state: Random seed
        **kwargs: Additional XGBoost parameters
    
    Returns:
        XGBoost regressor model
    """
    if not _HAS_XGBOOST:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    params = {
        'objective': objective,
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'random_state': random_state,
        'n_jobs': -1,
        'verbosity': 0,
        **kwargs
    }
    
    return xgb.XGBRegressor(**params)


def get_catboost_regressor(
    iterations: int = 500,
    learning_rate: float = 0.05,
    depth: int = 8,
    l2_leaf_reg: float = 3.0,
    random_strength: float = 1.0,
    bagging_temperature: float = 1.0,
    random_seed: int = 42,
    cat_features: Optional[List[str]] = None,
    **kwargs
) -> Any:
    """
    Create a CatBoost regressor for demand forecasting.
    
    CatBoost advantages:
    - Native handling of categorical features
    - Reduces overfitting with ordered boosting
    - Symmetric trees for faster prediction
    - Good for datasets with many categorical columns
    
    Args:
        iterations: Number of boosting iterations
        learning_rate: Learning rate
        depth: Tree depth
        l2_leaf_reg: L2 regularization coefficient
        random_strength: Random strength for scoring splits
        bagging_temperature: Bayesian bootstrap temperature
        random_seed: Random seed
        cat_features: List of categorical feature names
        **kwargs: Additional CatBoost parameters
    
    Returns:
        CatBoost regressor model
    """
    if not _HAS_CATBOOST:
        raise ImportError("CatBoost not installed. Install with: pip install catboost")
    
    params = {
        'iterations': iterations,
        'learning_rate': learning_rate,
        'depth': depth,
        'l2_leaf_reg': l2_leaf_reg,
        'random_strength': random_strength,
        'bagging_temperature': bagging_temperature,
        'random_seed': random_seed,
        'verbose': False,
        'allow_writing_files': False,
        **kwargs
    }
    
    if cat_features:
        params['cat_features'] = cat_features
    
    return CatBoostRegressor(**params)


def get_default_model_configs() -> Dict[str, ModelConfig]:
    """
    Get default configurations for all available ML models.
    
    Returns:
        Dictionary mapping model names to their configurations
    """
    configs = {}
    
    if _HAS_LIGHTGBM:
        configs['LightGBM'] = ModelConfig(
            model_type='lightgbm',
            params={
                'objective': 'regression',
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 8,
                'num_leaves': 31,
            },
            param_grid={
                'max_depth': [4, 6, 8],
                'num_leaves': [15, 31, 63],
                'learning_rate': [0.01, 0.05, 0.1],
            }
        )
        
        configs['LightGBM_Quantile_50'] = ModelConfig(
            model_type='lightgbm_quantile',
            params={'alpha': 0.5}
        )
        configs['LightGBM_Quantile_10'] = ModelConfig(
            model_type='lightgbm_quantile',
            params={'alpha': 0.1}
        )
        configs['LightGBM_Quantile_90'] = ModelConfig(
            model_type='lightgbm_quantile',
            params={'alpha': 0.9}
        )
    
    if _HAS_XGBOOST:
        configs['XGBoost'] = ModelConfig(
            model_type='xgboost',
            params={
                'objective': 'reg:squarederror',
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 8,
            },
            param_grid={
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'min_child_weight': [1, 3, 5],
            }
        )
    
    if _HAS_CATBOOST:
        configs['CatBoost'] = ModelConfig(
            model_type='catboost',
            params={
                'iterations': 500,
                'learning_rate': 0.05,
                'depth': 8,
            },
            param_grid={
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
            }
        )
    
    return configs


def create_model_from_config(config: ModelConfig) -> Any:
    """
    Create a model instance from configuration.
    
    Args:
        config: ModelConfig with model type and parameters
    
    Returns:
        Instantiated model
    """
    model_type = config.model_type.lower()
    
    if model_type == 'lightgbm':
        return get_lightgbm_regressor(**config.params)
    elif model_type == 'lightgbm_quantile':
        return get_lightgbm_quantile_regressor(**config.params)
    elif model_type == 'xgboost':
        return get_xgboost_regressor(**config.params)
    elif model_type == 'catboost':
        return get_catboost_regressor(**config.params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_sklearn_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    early_stopping_rounds: int = 50,
    eval_metric: str = 'rmse',
    verbose: bool = False
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a sklearn-compatible model with optional early stopping.
    
    Args:
        model: Sklearn-compatible model instance
        X_train: Training features
        y_train: Training target
        X_val: Validation features (for early stopping)
        y_val: Validation target (for early stopping)
        early_stopping_rounds: Rounds without improvement to stop
        eval_metric: Metric for early stopping
        verbose: Whether to print progress
    
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    logger.info(f"Training {type(model).__name__}...")
    
    fit_params = {}
    
    # Configure early stopping for gradient boosting models
    if X_val is not None and y_val is not None:
        model_type = type(model).__name__
        
        if 'LGBM' in model_type:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['callbacks'] = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose)
            ]
        elif 'XGB' in model_type:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['early_stopping_rounds'] = early_stopping_rounds
            fit_params['verbose'] = verbose
        elif 'CatBoost' in model_type:
            fit_params['eval_set'] = (X_val, y_val)
            fit_params['early_stopping_rounds'] = early_stopping_rounds
    
    # Fit model
    model.fit(X_train, y_train, **fit_params)
    
    # Calculate training metrics
    y_pred_train = model.predict(X_train)
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'train_r2': r2_score(y_train, y_pred_train),
    }
    
    # Calculate validation metrics if available
    if X_val is not None and y_val is not None:
        y_pred_val = model.predict(X_val)
        metrics['val_rmse'] = np.sqrt(mean_squared_error(y_val, y_pred_val))
        metrics['val_mae'] = mean_absolute_error(y_val, y_pred_val)
        metrics['val_r2'] = r2_score(y_val, y_pred_val)
    
    logger.info(f"Training complete. RMSE: {metrics.get('val_rmse', metrics['train_rmse']):.4f}")
    
    return model, metrics


def evaluate_sklearn_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate a trained sklearn model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name for logging
    
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate MAPE (excluding zeros to avoid division by zero)
    mask = y_test != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    else:
        mape = None
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
    }
    
    logger.info(f"Evaluation for {model_name}:")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAE:  {mae:.4f}")
    logger.info(f"  R2:   {r2:.4f}")
    if mape is not None:
        logger.info(f"  MAPE: {mape:.2f}%")
    
    return metrics


def train_and_evaluate_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    model_configs: Optional[Dict[str, ModelConfig]] = None,
    log_to_mlflow: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate all available ML models.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        model_configs: Optional custom model configurations
        log_to_mlflow: Whether to log results to MLflow
    
    Returns:
        Dictionary mapping model names to their results (model + metrics)
    """
    configs = model_configs or get_default_model_configs()
    results = {}
    
    logger.info(f"Training {len(configs)} models...")
    
    for model_name, config in configs.items():
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name}...")
            
            # Create model
            model = create_model_from_config(config)
            
            # Train model
            trained_model, train_metrics = train_sklearn_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                early_stopping_rounds=config.early_stopping_rounds,
                verbose=config.verbose
            )
            
            # Evaluate on test set
            test_metrics = evaluate_sklearn_model(
                model=trained_model,
                X_test=X_test,
                y_test=y_test,
                model_name=model_name
            )
            
            # Log to MLflow if enabled
            if log_to_mlflow:
                with mlflow.start_run(run_name=f"sklearn::{model_name}", nested=True):
                    mlflow.log_params(config.params)
                    for metric_name, value in test_metrics.items():
                        if value is not None:
                            mlflow.log_metric(metric_name, value)
                    mlflow.sklearn.log_model(trained_model, model_name)
            
            results[model_name] = {
                'model': trained_model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'config': config
            }
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results


def get_feature_importance(
    model: Any,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
    
    Returns:
        DataFrame with feature importances sorted by importance
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return pd.DataFrame()
    
    importance = model.feature_importances_
    
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    df_importance = df_importance.sort_values('importance', ascending=False)
    df_importance['rank'] = range(1, len(df_importance) + 1)
    df_importance['cumulative_importance'] = df_importance['importance'].cumsum()
    df_importance['cumulative_pct'] = df_importance['cumulative_importance'] / df_importance['importance'].sum()
    
    return df_importance


class QuantileEnsemble(BaseEstimator, RegressorMixin):
    """
    Ensemble of quantile regressors for prediction intervals.
    
    Trains models for multiple quantiles (e.g., 0.1, 0.5, 0.9) to generate
    point predictions with confidence intervals.
    
    Example:
        ensemble = QuantileEnsemble(quantiles=[0.1, 0.5, 0.9])
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)  # Returns median
        intervals = ensemble.predict_intervals(X_test)  # Returns intervals
    """
    
    def __init__(
        self,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        base_model_type: str = 'lightgbm',
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 8,
        random_state: int = 42
    ):
        self.quantiles = quantiles
        self.base_model_type = base_model_type
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.models_ = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'QuantileEnsemble':
        """
        Fit quantile models.
        
        Args:
            X: Training features
            y: Training target
        
        Returns:
            Self (fitted ensemble)
        """
        logger.info(f"Training QuantileEnsemble for quantiles: {self.quantiles}")
        
        for q in self.quantiles:
            logger.info(f"  Training quantile={q}...")
            
            if self.base_model_type == 'lightgbm' and _HAS_LIGHTGBM:
                model = get_lightgbm_quantile_regressor(
                    alpha=q,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=self.random_state
                )
            else:
                raise ValueError(f"Unsupported base model type: {self.base_model_type}")
            
            model.fit(X, y)
            self.models_[q] = model
        
        logger.info("QuantileEnsemble training complete!")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using median model (quantile=0.5).
        
        Args:
            X: Features to predict
        
        Returns:
            Predictions array
        """
        if 0.5 in self.models_:
            return self.models_[0.5].predict(X)
        else:
            # Use closest to median
            median_q = min(self.quantiles, key=lambda x: abs(x - 0.5))
            return self.models_[median_q].predict(X)
    
    def predict_intervals(
        self,
        X: pd.DataFrame,
        lower_quantile: float = 0.1,
        upper_quantile: float = 0.9
    ) -> pd.DataFrame:
        """
        Predict with confidence intervals.
        
        Args:
            X: Features to predict
            lower_quantile: Lower bound quantile
            upper_quantile: Upper bound quantile
        
        Returns:
            DataFrame with prediction, lower_bound, upper_bound
        """
        predictions = {}
        
        for q, model in self.models_.items():
            predictions[f'q{int(q*100)}'] = model.predict(X)
        
        result = pd.DataFrame({
            'prediction': self.predict(X),
            'lower_bound': predictions.get(f'q{int(lower_quantile*100)}', predictions[f'q{int(min(self.quantiles)*100)}']),
            'upper_bound': predictions.get(f'q{int(upper_quantile*100)}', predictions[f'q{int(max(self.quantiles)*100)}']),
        })
        
        return result


# Utility functions for model comparison
def compare_models(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'rmse'
) -> pd.DataFrame:
    """
    Compare model results and rank by specified metric.
    
    Args:
        results: Dictionary of model results from train_and_evaluate_all_models
        metric: Metric to compare ('rmse', 'mae', 'r2', 'mape')
    
    Returns:
        DataFrame with model comparison sorted by metric
    """
    comparison = []
    
    for model_name, result in results.items():
        if 'error' in result:
            continue
        
        test_metrics = result.get('test_metrics', {})
        comparison.append({
            'model': model_name,
            'rmse': test_metrics.get('rmse'),
            'mae': test_metrics.get('mae'),
            'r2': test_metrics.get('r2'),
            'mape': test_metrics.get('mape'),
        })
    
    df = pd.DataFrame(comparison)
    
    # Sort by metric (ascending for error metrics, descending for R2)
    ascending = metric != 'r2'
    df = df.sort_values(metric, ascending=ascending)
    df['rank'] = range(1, len(df) + 1)
    
    return df


def get_best_model(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'rmse'
) -> Tuple[str, Any]:
    """
    Get the best performing model based on metric.
    
    Args:
        results: Dictionary of model results
        metric: Metric to optimize
    
    Returns:
        Tuple of (model_name, model_instance)
    """
    comparison = compare_models(results, metric)
    
    if len(comparison) == 0:
        raise ValueError("No successful models found")
    
    best_name = comparison.iloc[0]['model']
    best_model = results[best_name]['model']
    
    logger.info(f"Best model: {best_name} ({metric}={comparison.iloc[0][metric]:.4f})")
    
    return best_name, best_model
