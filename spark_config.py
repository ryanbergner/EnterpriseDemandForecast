"""
Spark Configuration Presets

Provides optimized Spark configurations for different scenarios:
- Local development
- Small cluster (< 10 executors)
- Medium cluster (10-50 executors)
- Large cluster (> 50 executors)
- Memory-constrained environments
- GPU-enabled clusters

These configurations are tuned for demand forecasting workloads
with time series data and ML model training.
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusterSize(Enum):
    """Cluster size categories."""
    LOCAL = "local"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class WorkloadType(Enum):
    """Workload type categories."""
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    INFERENCE = "inference"
    MIXED = "mixed"


@dataclass
class SparkConfig:
    """Spark configuration container."""
    name: str
    description: str
    config: Dict[str, str]


def get_local_config(
    driver_memory: str = "4g",
    cores: int = 4
) -> SparkConfig:
    """
    Get configuration for local development.
    
    Args:
        driver_memory: Driver memory allocation
        cores: Number of local cores to use
    
    Returns:
        SparkConfig for local development
    """
    return SparkConfig(
        name="local_dev",
        description="Optimized configuration for local development and testing",
        config={
            # Master
            "spark.master": f"local[{cores}]",
            
            # Memory
            "spark.driver.memory": driver_memory,
            "spark.driver.maxResultSize": "2g",
            
            # Parallelism
            "spark.sql.shuffle.partitions": str(cores * 2),
            "spark.default.parallelism": str(cores * 2),
            
            # UI
            "spark.ui.enabled": "true",
            "spark.ui.port": "4040",
            
            # SQL optimizations
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            
            # Serialization
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            
            # Cache
            "spark.sql.inMemoryColumnarStorage.compressed": "true",
            "spark.sql.inMemoryColumnarStorage.batchSize": "10000",
            
            # Local disk
            "spark.local.dir": "/tmp/spark-local",
        }
    )


def get_small_cluster_config(
    num_executors: int = 4,
    executor_memory: str = "4g",
    executor_cores: int = 2
) -> SparkConfig:
    """
    Get configuration for small clusters (< 10 executors).
    
    Args:
        num_executors: Number of executors
        executor_memory: Memory per executor
        executor_cores: Cores per executor
    
    Returns:
        SparkConfig for small cluster
    """
    total_cores = num_executors * executor_cores
    
    return SparkConfig(
        name="small_cluster",
        description="Configuration for small Spark clusters (< 10 executors)",
        config={
            # Executors
            "spark.executor.instances": str(num_executors),
            "spark.executor.memory": executor_memory,
            "spark.executor.cores": str(executor_cores),
            "spark.executor.memoryOverhead": "1g",
            
            # Driver
            "spark.driver.memory": "4g",
            "spark.driver.maxResultSize": "2g",
            
            # Parallelism
            "spark.sql.shuffle.partitions": str(total_cores * 4),
            "spark.default.parallelism": str(total_cores * 4),
            
            # Adaptive execution
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.adaptive.localShuffleReader.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            
            # Broadcast
            "spark.sql.autoBroadcastJoinThreshold": "100m",
            
            # Serialization
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.kryoserializer.buffer.max": "512m",
            
            # Shuffle
            "spark.shuffle.compress": "true",
            "spark.shuffle.spill.compress": "true",
            "spark.shuffle.file.buffer": "1m",
            
            # Memory management
            "spark.memory.fraction": "0.75",
            "spark.memory.storageFraction": "0.3",
            
            # Dynamic allocation
            "spark.dynamicAllocation.enabled": "false",
        }
    )


def get_medium_cluster_config(
    num_executors: int = 20,
    executor_memory: str = "8g",
    executor_cores: int = 4
) -> SparkConfig:
    """
    Get configuration for medium clusters (10-50 executors).
    
    Args:
        num_executors: Number of executors
        executor_memory: Memory per executor
        executor_cores: Cores per executor
    
    Returns:
        SparkConfig for medium cluster
    """
    total_cores = num_executors * executor_cores
    
    return SparkConfig(
        name="medium_cluster",
        description="Configuration for medium Spark clusters (10-50 executors)",
        config={
            # Executors
            "spark.executor.instances": str(num_executors),
            "spark.executor.memory": executor_memory,
            "spark.executor.cores": str(executor_cores),
            "spark.executor.memoryOverhead": "2g",
            
            # Driver
            "spark.driver.memory": "8g",
            "spark.driver.maxResultSize": "4g",
            
            # Parallelism
            "spark.sql.shuffle.partitions": str(min(total_cores * 4, 1000)),
            "spark.default.parallelism": str(total_cores * 4),
            
            # Adaptive execution
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.minPartitionNum": str(num_executors),
            "spark.sql.adaptive.localShuffleReader.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes": "256m",
            
            # Broadcast
            "spark.sql.autoBroadcastJoinThreshold": "200m",
            
            # Serialization
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.kryoserializer.buffer.max": "1g",
            
            # Shuffle
            "spark.shuffle.compress": "true",
            "spark.shuffle.spill.compress": "true",
            "spark.shuffle.file.buffer": "2m",
            "spark.reducer.maxSizeInFlight": "96m",
            
            # Memory management
            "spark.memory.fraction": "0.8",
            "spark.memory.storageFraction": "0.3",
            
            # Speculation
            "spark.speculation": "true",
            "spark.speculation.multiplier": "1.5",
            "spark.speculation.quantile": "0.9",
            
            # Network
            "spark.network.timeout": "600s",
            "spark.rpc.askTimeout": "600s",
            
            # Dynamic allocation
            "spark.dynamicAllocation.enabled": "true",
            "spark.dynamicAllocation.minExecutors": str(num_executors // 2),
            "spark.dynamicAllocation.maxExecutors": str(num_executors * 2),
            "spark.dynamicAllocation.initialExecutors": str(num_executors),
        }
    )


def get_large_cluster_config(
    num_executors: int = 100,
    executor_memory: str = "16g",
    executor_cores: int = 4
) -> SparkConfig:
    """
    Get configuration for large clusters (> 50 executors).
    
    Args:
        num_executors: Number of executors
        executor_memory: Memory per executor
        executor_cores: Cores per executor
    
    Returns:
        SparkConfig for large cluster
    """
    total_cores = num_executors * executor_cores
    
    return SparkConfig(
        name="large_cluster",
        description="Configuration for large Spark clusters (50+ executors)",
        config={
            # Executors
            "spark.executor.instances": str(num_executors),
            "spark.executor.memory": executor_memory,
            "spark.executor.cores": str(executor_cores),
            "spark.executor.memoryOverhead": "4g",
            
            # Driver
            "spark.driver.memory": "16g",
            "spark.driver.maxResultSize": "8g",
            
            # Parallelism - more partitions for large clusters
            "spark.sql.shuffle.partitions": str(min(total_cores * 3, 2000)),
            "spark.default.parallelism": str(total_cores * 3),
            
            # Adaptive execution
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.adaptive.coalescePartitions.minPartitionNum": str(num_executors),
            "spark.sql.adaptive.coalescePartitions.initialPartitionNum": str(total_cores * 2),
            "spark.sql.adaptive.localShuffleReader.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            "spark.sql.adaptive.skewJoin.skewedPartitionFactor": "5",
            
            # Broadcast - higher threshold for large data
            "spark.sql.autoBroadcastJoinThreshold": "500m",
            
            # Serialization
            "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
            "spark.kryoserializer.buffer.max": "2g",
            
            # Shuffle - optimized for large scale
            "spark.shuffle.compress": "true",
            "spark.shuffle.spill.compress": "true",
            "spark.shuffle.file.buffer": "4m",
            "spark.reducer.maxSizeInFlight": "128m",
            "spark.shuffle.io.maxRetries": "10",
            "spark.shuffle.io.retryWait": "30s",
            
            # Memory management
            "spark.memory.fraction": "0.8",
            "spark.memory.storageFraction": "0.2",
            "spark.memory.offHeap.enabled": "true",
            "spark.memory.offHeap.size": "4g",
            
            # Speculation
            "spark.speculation": "true",
            "spark.speculation.multiplier": "2.0",
            "spark.speculation.quantile": "0.95",
            
            # Network - longer timeouts for large jobs
            "spark.network.timeout": "900s",
            "spark.rpc.askTimeout": "900s",
            "spark.rpc.message.maxSize": "512",
            
            # Dynamic allocation
            "spark.dynamicAllocation.enabled": "true",
            "spark.dynamicAllocation.minExecutors": str(num_executors // 4),
            "spark.dynamicAllocation.maxExecutors": str(num_executors * 2),
            "spark.dynamicAllocation.initialExecutors": str(num_executors),
            "spark.dynamicAllocation.executorIdleTimeout": "300s",
            
            # SQL
            "spark.sql.files.maxPartitionBytes": "256m",
            "spark.sql.files.openCostInBytes": "8m",
        }
    )


def get_ml_training_config(
    base_config: SparkConfig,
    dataset_size_gb: float = 10.0
) -> SparkConfig:
    """
    Augment base config with ML training optimizations.
    
    Args:
        base_config: Base Spark configuration
        dataset_size_gb: Estimated dataset size in GB
    
    Returns:
        SparkConfig optimized for ML training
    """
    ml_config = base_config.config.copy()
    
    # ML-specific optimizations
    ml_config.update({
        # Higher shuffle partitions for cross-validation
        "spark.sql.shuffle.partitions": str(
            max(int(ml_config.get("spark.sql.shuffle.partitions", "200")), int(dataset_size_gb * 20))
        ),
        
        # Keep intermediate results in memory
        "spark.memory.storageFraction": "0.4",
        
        # Optimize for iterative algorithms
        "spark.rpc.message.maxSize": "256",
        
        # Checkpointing for long jobs
        "spark.cleaner.referenceTracking.cleanCheckpoints": "true",
        
        # ML library specific
        "spark.ml.parallelism": "10",
    })
    
    return SparkConfig(
        name=f"{base_config.name}_ml_training",
        description=f"{base_config.description} with ML training optimizations",
        config=ml_config
    )


def get_config_for_workload(
    cluster_size: ClusterSize,
    workload_type: WorkloadType = WorkloadType.MIXED,
    custom_params: Optional[Dict[str, str]] = None
) -> SparkConfig:
    """
    Get optimized configuration based on cluster size and workload type.
    
    Args:
        cluster_size: Size of the Spark cluster
        workload_type: Type of workload to optimize for
        custom_params: Custom parameters to override
    
    Returns:
        SparkConfig optimized for the scenario
    """
    # Get base config by cluster size
    if cluster_size == ClusterSize.LOCAL:
        config = get_local_config()
    elif cluster_size == ClusterSize.SMALL:
        config = get_small_cluster_config()
    elif cluster_size == ClusterSize.MEDIUM:
        config = get_medium_cluster_config()
    else:
        config = get_large_cluster_config()
    
    # Apply workload-specific optimizations
    if workload_type == WorkloadType.MODEL_TRAINING:
        config = get_ml_training_config(config)
    elif workload_type == WorkloadType.INFERENCE:
        # Lower memory fraction for inference (less caching needed)
        config.config["spark.memory.storageFraction"] = "0.2"
    elif workload_type == WorkloadType.FEATURE_ENGINEERING:
        # Higher shuffle partitions for window functions
        current_partitions = int(config.config.get("spark.sql.shuffle.partitions", "200"))
        config.config["spark.sql.shuffle.partitions"] = str(current_partitions * 2)
    
    # Apply custom overrides
    if custom_params:
        config.config.update(custom_params)
    
    logger.info(f"Generated Spark config: {config.name}")
    return config


def apply_config(
    spark_builder,
    config: SparkConfig
):
    """
    Apply configuration to SparkSession builder.
    
    Args:
        spark_builder: SparkSession.builder instance
        config: SparkConfig to apply
    
    Returns:
        SparkSession builder with applied configuration
    """
    logger.info(f"Applying Spark configuration: {config.name}")
    
    for key, value in config.config.items():
        spark_builder = spark_builder.config(key, value)
        logger.debug(f"  {key}: {value}")
    
    return spark_builder


def print_config(config: SparkConfig) -> None:
    """
    Print configuration in a readable format.
    
    Args:
        config: SparkConfig to print
    """
    print(f"\n{'='*60}")
    print(f"Spark Configuration: {config.name}")
    print(f"Description: {config.description}")
    print(f"{'='*60}")
    
    # Group by category
    categories = {
        "Memory": ["memory", "driver.memory", "executor.memory"],
        "Parallelism": ["parallelism", "shuffle.partitions", "cores"],
        "Adaptive": ["adaptive"],
        "Serialization": ["serializer", "kryo"],
        "Shuffle": ["shuffle"],
        "Network": ["network", "rpc", "timeout"],
        "Dynamic Allocation": ["dynamicAllocation"],
        "Other": []
    }
    
    for category, keywords in categories.items():
        matching = {
            k: v for k, v in config.config.items()
            if any(kw in k for kw in keywords) or (not any(any(kw in k for kw in kws) for kws in categories.values() if kws))
        }
        
        if matching:
            print(f"\n{category}:")
            for k, v in matching.items():
                print(f"  {k}: {v}")
    
    print(f"\n{'='*60}\n")
