"""
Batch Training Optimizer Module

Implements PySpark optimizations for efficient batch training:
- DataFrame caching strategies
- Broadcast variable management
- Partitioning optimization
- Memory-efficient processing
- Checkpoint management
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, broadcast
from pyspark.ml import Pipeline, PipelineModel
from pyspark.storagelevel import StorageLevel
from typing import Dict, List, Optional, Any, Callable
import logging
from contextlib import contextmanager
from functools import wraps
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchTrainingOptimizer:
    """
    Optimizer for batch training workflows in PySpark.
    
    Provides:
    - Intelligent caching strategies
    - Broadcast variable management
    - Partitioning optimization
    - Memory pressure monitoring
    
    Example:
        optimizer = BatchTrainingOptimizer(spark)
        
        with optimizer.optimized_context(df) as opt_df:
            model = train_model(opt_df)
    """
    
    def __init__(
        self,
        spark: SparkSession,
        default_storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK,
        auto_repartition: bool = True,
        target_partitions: Optional[int] = None
    ):
        """
        Initialize the batch optimizer.
        
        Args:
            spark: SparkSession instance
            default_storage_level: Default storage level for caching
            auto_repartition: Whether to auto-optimize partitions
            target_partitions: Target number of partitions (None = auto)
        """
        self.spark = spark
        self.default_storage_level = default_storage_level
        self.auto_repartition = auto_repartition
        self.target_partitions = target_partitions or spark.conf.get(
            "spark.sql.shuffle.partitions", "200"
        )
        self._cached_dfs: List[DataFrame] = []
        self._broadcast_vars: List[Any] = []
    
    def optimize_partitions(
        self,
        df: DataFrame,
        partition_cols: Optional[List[str]] = None,
        target_partitions: Optional[int] = None
    ) -> DataFrame:
        """
        Optimize DataFrame partitioning for training.
        
        Args:
            df: Input DataFrame
            partition_cols: Columns to partition by (e.g., product_id)
            target_partitions: Target number of partitions
        
        Returns:
            Repartitioned DataFrame
        """
        if not self.auto_repartition:
            return df
        
        target = target_partitions or int(self.target_partitions)
        current_partitions = df.rdd.getNumPartitions()
        
        logger.info(f"Current partitions: {current_partitions}, Target: {target}")
        
        if partition_cols:
            # Partition by specific columns for co-located processing
            logger.info(f"Repartitioning by columns: {partition_cols}")
            return df.repartition(target, *[col(c) for c in partition_cols])
        elif current_partitions > target * 2:
            # Coalesce if too many partitions
            logger.info(f"Coalescing from {current_partitions} to {target}")
            return df.coalesce(target)
        elif current_partitions < target // 2:
            # Repartition if too few
            logger.info(f"Repartitioning from {current_partitions} to {target}")
            return df.repartition(target)
        
        return df
    
    def cache_dataframe(
        self,
        df: DataFrame,
        storage_level: Optional[StorageLevel] = None,
        name: str = "cached_df"
    ) -> DataFrame:
        """
        Cache DataFrame with optimal storage level.
        
        Args:
            df: DataFrame to cache
            storage_level: Storage level (default: MEMORY_AND_DISK)
            name: Name for logging
        
        Returns:
            Cached DataFrame
        """
        level = storage_level or self.default_storage_level
        logger.info(f"Caching '{name}' with storage level: {level}")
        
        cached = df.persist(level)
        self._cached_dfs.append(cached)
        
        # Trigger materialization
        count = cached.count()
        logger.info(f"Cached '{name}': {count} rows")
        
        return cached
    
    def broadcast_small_df(
        self,
        df: DataFrame,
        name: str = "broadcast_df"
    ) -> DataFrame:
        """
        Mark small DataFrame for broadcast join.
        
        Use for lookup tables, feature metadata, etc.
        
        Args:
            df: Small DataFrame to broadcast
            name: Name for logging
        
        Returns:
            Broadcast-marked DataFrame
        """
        logger.info(f"Broadcasting '{name}'")
        return broadcast(df)
    
    def cleanup(self) -> None:
        """
        Clean up cached DataFrames and broadcast variables.
        """
        logger.info(f"Cleaning up {len(self._cached_dfs)} cached DataFrames")
        
        for df in self._cached_dfs:
            try:
                df.unpersist()
            except Exception as e:
                logger.warning(f"Error unpersisting DataFrame: {e}")
        
        self._cached_dfs.clear()
        self._broadcast_vars.clear()
    
    @contextmanager
    def optimized_context(
        self,
        df: DataFrame,
        partition_cols: Optional[List[str]] = None,
        cache: bool = True
    ):
        """
        Context manager for optimized training.
        
        Automatically handles partitioning, caching, and cleanup.
        
        Args:
            df: DataFrame to optimize
            partition_cols: Optional columns for partitioning
            cache: Whether to cache the DataFrame
        
        Yields:
            Optimized DataFrame
        
        Example:
            with optimizer.optimized_context(df, partition_cols=['item_id']) as opt_df:
                model = train_model(opt_df)
        """
        try:
            # Optimize partitions
            opt_df = self.optimize_partitions(df, partition_cols)
            
            # Cache if requested
            if cache:
                opt_df = self.cache_dataframe(opt_df, name="training_data")
            
            yield opt_df
            
        finally:
            self.cleanup()


def optimized_training(
    partition_cols: Optional[List[str]] = None,
    cache: bool = True
):
    """
    Decorator for optimized training functions.
    
    Args:
        partition_cols: Columns to partition by
        cache: Whether to cache input DataFrame
    
    Example:
        @optimized_training(partition_cols=['item_id'])
        def train_model(df, spark):
            # Training logic
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find DataFrame and SparkSession in arguments
            df = None
            spark = None
            
            for arg in args:
                if isinstance(arg, DataFrame):
                    df = arg
                elif isinstance(arg, SparkSession):
                    spark = arg
            
            for key, val in kwargs.items():
                if isinstance(val, DataFrame):
                    df = val
                elif isinstance(val, SparkSession):
                    spark = val
            
            if df is None or spark is None:
                return func(*args, **kwargs)
            
            optimizer = BatchTrainingOptimizer(spark)
            
            with optimizer.optimized_context(df, partition_cols, cache) as opt_df:
                # Replace df in args/kwargs with optimized version
                new_args = tuple(
                    opt_df if isinstance(a, DataFrame) else a 
                    for a in args
                )
                new_kwargs = {
                    k: opt_df if isinstance(v, DataFrame) else v
                    for k, v in kwargs.items()
                }
                
                return func(*new_args, **new_kwargs)
        
        return wrapper
    return decorator


class CheckpointManager:
    """
    Manages checkpointing for long training workflows.
    
    Checkpointing breaks RDD lineage to prevent stack overflow
    and improves recovery from failures.
    """
    
    def __init__(
        self,
        spark: SparkSession,
        checkpoint_dir: str,
        checkpoint_interval: int = 10
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            spark: SparkSession instance
            checkpoint_dir: Directory for checkpoints
            checkpoint_interval: Checkpoint every N iterations
        """
        self.spark = spark
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self._iteration = 0
        
        # Set checkpoint directory
        spark.sparkContext.setCheckpointDir(checkpoint_dir)
        logger.info(f"Checkpoint directory set to: {checkpoint_dir}")
    
    def maybe_checkpoint(
        self,
        df: DataFrame,
        force: bool = False
    ) -> DataFrame:
        """
        Checkpoint DataFrame if needed.
        
        Args:
            df: DataFrame to potentially checkpoint
            force: Force checkpoint regardless of interval
        
        Returns:
            Checkpointed DataFrame (or original if not checkpointed)
        """
        self._iteration += 1
        
        if force or self._iteration % self.checkpoint_interval == 0:
            logger.info(f"Checkpointing at iteration {self._iteration}")
            return df.checkpoint()
        
        return df


def get_optimal_spark_config(
    data_size_gb: float,
    num_executors: int = 4,
    executor_memory_gb: int = 8
) -> Dict[str, str]:
    """
    Generate optimal Spark configuration based on data size.
    
    Args:
        data_size_gb: Estimated data size in GB
        num_executors: Number of executors
        executor_memory_gb: Memory per executor in GB
    
    Returns:
        Dictionary of Spark configuration settings
    """
    # Calculate optimal settings
    shuffle_partitions = max(200, int(data_size_gb * 10))
    broadcast_threshold = min(100 * 1024 * 1024, int(data_size_gb * 10 * 1024 * 1024))
    
    config = {
        # Memory settings
        "spark.executor.memory": f"{executor_memory_gb}g",
        "spark.executor.memoryOverhead": f"{max(1, executor_memory_gb // 4)}g",
        "spark.driver.memory": f"{max(2, executor_memory_gb // 2)}g",
        
        # Parallelism
        "spark.sql.shuffle.partitions": str(shuffle_partitions),
        "spark.default.parallelism": str(num_executors * 4),
        
        # Broadcast
        "spark.sql.autoBroadcastJoinThreshold": str(broadcast_threshold),
        
        # Serialization
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer.max": "1g",
        
        # Adaptive query execution
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        
        # Memory management
        "spark.memory.fraction": "0.8",
        "spark.memory.storageFraction": "0.3",
        
        # Shuffle
        "spark.shuffle.file.buffer": "1m",
        "spark.shuffle.spill.compress": "true",
        
        # Speculation (for stragglers)
        "spark.speculation": "true",
        "spark.speculation.multiplier": "1.5",
    }
    
    logger.info(f"Generated Spark config for {data_size_gb}GB data:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    return config


def estimate_data_size(df: DataFrame) -> float:
    """
    Estimate DataFrame size in GB.
    
    Args:
        df: DataFrame to estimate
    
    Returns:
        Estimated size in GB
    """
    # Sample-based estimation
    sample_size = min(10000, df.count())
    if sample_size == 0:
        return 0.0
    
    sample = df.limit(sample_size)
    
    # Get pandas size for sample
    try:
        sample_pd = sample.toPandas()
        sample_bytes = sample_pd.memory_usage(deep=True).sum()
        total_rows = df.count()
        estimated_bytes = (sample_bytes / sample_size) * total_rows
        return estimated_bytes / (1024 ** 3)
    except Exception:
        # Fallback: rough estimate based on row count and columns
        num_cols = len(df.columns)
        num_rows = df.count()
        # Assume ~100 bytes per cell on average
        return (num_cols * num_rows * 100) / (1024 ** 3)


class MemoryMonitor:
    """
    Monitor memory usage during training.
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        Get current memory status.
        
        Returns:
            Dictionary with memory statistics
        """
        sc = self.spark.sparkContext
        
        # Get storage info
        storage_info = sc._jsc.sc().getRDDStorageInfo()
        
        total_cached = 0
        for rdd in storage_info:
            total_cached += rdd.memSize()
        
        return {
            "cached_memory_mb": total_cached / (1024 * 1024),
            "num_cached_rdds": len(storage_info),
        }
    
    def log_memory_status(self) -> None:
        """Log current memory status."""
        status = self.get_memory_status()
        logger.info(f"Memory Status: {status}")


def timed_operation(operation_name: str):
    """
    Decorator to time operations.
    
    Args:
        operation_name: Name of the operation for logging
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"Starting: {operation_name}")
            
            result = func(*args, **kwargs)
            
            elapsed = time.time() - start_time
            logger.info(f"Completed: {operation_name} in {elapsed:.2f}s")
            
            return result
        return wrapper
    return decorator
