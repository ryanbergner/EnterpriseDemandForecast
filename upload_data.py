#!/usr/bin/env python
# upload_data.py
# Script for users to upload their own data matching M5 dataset schema

import argparse
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
from pyspark.sql.types import DoubleType, StringType
import os
import sys

# Expected M5 schema
M5_SCHEMA = {
    "item_id": StringType(),  # Product identifier
    "OrderDate": "date",       # Date column (will be converted)
    "Quantity": DoubleType()   # Quantity/demand column
}

REQUIRED_COLUMNS = ["item_id", "OrderDate", "Quantity"]


def validate_schema(df, spark):
    """
    Validate that the uploaded data matches M5 schema requirements.
    
    Args:
        df: Spark DataFrame
        spark: SparkSession
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check required columns exist
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check data types
    schema = df.schema
    
    # Check item_id is string-like
    item_id_type = dict(schema.fields)["item_id"].dataType
    if not isinstance(item_id_type, StringType):
        return False, f"item_id must be StringType, got {item_id_type}"
    
    # Check Quantity is numeric
    quantity_type = dict(schema.fields)["Quantity"].dataType
    if not isinstance(quantity_type, (DoubleType, type(DoubleType()))):
        try:
            # Try to cast to double
            df.select(col("Quantity").cast(DoubleType())).limit(1).collect()
        except:
            return False, f"Quantity must be numeric (DoubleType), got {quantity_type}"
    
    # Check OrderDate is date-like
    try:
        df.select(to_date(col("OrderDate"))).limit(1).collect()
    except:
        return False, "OrderDate must be convertible to date type"
    
    # Check for null values in required columns
    null_counts = {}
    for col_name in REQUIRED_COLUMNS:
        null_count = df.filter(col(col_name).isNull()).count()
        if null_count > 0:
            null_counts[col_name] = null_count
    
    if null_counts:
        return False, f"Found null values in required columns: {null_counts}"
    
    return True, "Schema validation passed"


def load_and_validate_data(input_path: str, spark: SparkSession):
    """
    Load data from file and validate schema.
    
    Args:
        input_path: Path to input file (CSV, Parquet, etc.)
        spark: SparkSession
        
    Returns:
        DataFrame: Validated and cleaned DataFrame
    """
    print(f"Loading data from: {input_path}")
    
    # Determine file type
    file_ext = os.path.splitext(input_path)[1].lower()
    
    if file_ext == '.csv':
        df = spark.read.format("csv").option("header", True).load(input_path)
    elif file_ext == '.parquet':
        df = spark.read.parquet(input_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported: .csv, .parquet")
    
    print(f"Loaded {df.count()} rows, {len(df.columns)} columns")
    print(f"Columns: {df.columns}")
    
    # Validate schema
    is_valid, message = validate_schema(df, spark)
    if not is_valid:
        raise ValueError(f"Schema validation failed: {message}")
    
    print("✓ Schema validation passed")
    
    # Convert types to match M5 schema
    df = (
        df
        .withColumn("item_id", col("item_id").cast(StringType()))
        .withColumn("OrderDate", to_date(col("OrderDate")))
        .withColumn("Quantity", col("Quantity").cast(DoubleType()))
    )
    
    # Remove any rows with null values after conversion
    initial_count = df.count()
    df = df.dropna()
    final_count = df.count()
    
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} rows with null values")
    
    print(f"Final dataset: {final_count} rows")
    
    return df


def save_for_training(df, output_path: str, spark: SparkSession):
    """
    Save validated data in format ready for training.
    
    Args:
        df: Spark DataFrame
        output_path: Path to save output
        spark: SparkSession
    """
    print(f"Saving validated data to: {output_path}")
    
    # Save as Parquet (recommended) or CSV
    if output_path.endswith('.parquet'):
        df.write.mode("overwrite").parquet(output_path)
    else:
        df.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path)
    
    print("✓ Data saved successfully")
    print(f"\nYou can now use this data for training:")
    print(f"  python main_train.py {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload and validate custom data matching M5 dataset schema"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input data file (CSV or Parquet)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save validated data (default: input_path + '_validated')"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet"],
        default="parquet",
        help="Output format (default: parquet)"
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output_path is None:
        base_name = os.path.splitext(args.input_path)[0]
        args.output_path = f"{base_name}_validated.{args.format}"
    
    # Create Spark session
    spark = SparkSession.builder.appName("DataUploadValidation").getOrCreate()
    
    try:
        # Load and validate data
        df = load_and_validate_data(args.input_path, spark)
        
        # Display sample
        print("\nSample data:")
        df.show(10)
        
        # Display summary statistics
        print("\nSummary statistics:")
        df.select("Quantity").describe().show()
        
        print(f"\nUnique items: {df.select('item_id').distinct().count()}")
        print(f"Date range: {df.agg({'OrderDate': 'min', 'OrderDate': 'max'}).collect()}")
        
        # Save validated data
        save_for_training(df, args.output_path, spark)
        
        print("\n✓ Data upload and validation complete!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
