#!/usr/bin/env python
"""
upload_custom_data.py

Script to upload and validate custom datasets for forecasting

This script allows users to upload their own datasets, validates the schema
against M5 dataset requirements, and prepares the data for training.

Usage:
    python upload_custom_data.py --input-path /path/to/custom_data.csv --output-path data/custom/
    
Requirements:
    - Input CSV must have columns: date, item_id, quantity
    - Date column must be in a parseable date format
    - Quantity column must be numeric
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, expr, count, min as spark_min, max as spark_max
from pyspark.sql.types import DoubleType, DateType, StringType

# Import config for schema validation
from config import (
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS,
    validate_custom_dataset_columns,
    DATE_COLUMN,
    PRODUCT_ID_COLUMN,
    QUANTITY_COLUMN
)


class CustomDataUploader:
    """
    Class to handle custom data upload and validation.
    """
    
    def __init__(self, spark_session=None):
        """
        Initialize the uploader.
        
        Args:
            spark_session: Optional SparkSession. If None, creates a new one.
        """
        if spark_session:
            self.spark = spark_session
        else:
            self.spark = (
                SparkSession.builder
                .appName("CustomDataUpload")
                .config("spark.driver.memory", "2g")
                .getOrCreate()
            )
    
    def load_data(self, input_path, delimiter=",", header=True):
        """
        Load data from file.
        
        Args:
            input_path: Path to input file
            delimiter: CSV delimiter (default: comma)
            header: Whether file has header row
            
        Returns:
            DataFrame: Loaded Spark DataFrame
        """
        print(f"[1/6] Loading data from: {input_path}")
        
        try:
            df = (
                self.spark.read
                .format("csv")
                .option("header", str(header).lower())
                .option("delimiter", delimiter)
                .option("inferSchema", "true")
                .load(input_path)
            )
            
            print(f"✓ Loaded {df.count()} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def validate_schema(self, df):
        """
        Validate that the dataframe has required columns.
        
        Args:
            df: Spark DataFrame to validate
            
        Returns:
            tuple: (is_valid, error_messages)
        """
        print("\n[2/6] Validating schema...")
        
        errors = []
        df_columns = [c.lower() for c in df.columns]
        
        # Check for required columns
        required_lower = [c.lower() for c in REQUIRED_COLUMNS]
        missing_required = [c for c in required_lower if c not in df_columns]
        
        if missing_required:
            errors.append(f"Missing required columns: {missing_required}")
        
        # Check for optional columns
        optional_lower = [c.lower() for c in OPTIONAL_COLUMNS]
        found_optional = [c for c in optional_lower if c in df_columns]
        
        if not errors:
            print("✓ Schema validation passed")
            if found_optional:
                print(f"  Found optional columns: {found_optional}")
            return True, []
        else:
            print("✗ Schema validation failed")
            for error in errors:
                print(f"  - {error}")
            return False, errors
    
    def map_columns(self, df, date_col_name=None, item_id_col_name=None, quantity_col_name=None):
        """
        Map custom column names to standard names.
        
        Args:
            df: Input DataFrame
            date_col_name: Name of date column in input data
            item_id_col_name: Name of item_id column in input data
            quantity_col_name: Name of quantity column in input data
            
        Returns:
            DataFrame: DataFrame with renamed columns
        """
        print("\n[3/6] Mapping column names...")
        
        # If names not provided, try to detect from common patterns
        if not date_col_name:
            date_patterns = ['date', 'orderdate', 'order_date', 'demanddate', 'demand_date', 'timestamp']
            for col_name in df.columns:
                if col_name.lower() in date_patterns:
                    date_col_name = col_name
                    break
        
        if not item_id_col_name:
            item_patterns = ['item_id', 'itemid', 'product_id', 'productid', 'sku', 'itemnumber']
            for col_name in df.columns:
                if col_name.lower() in item_patterns:
                    item_id_col_name = col_name
                    break
        
        if not quantity_col_name:
            qty_patterns = ['quantity', 'qty', 'demand', 'demandquantity', 'demand_quantity', 'sales']
            for col_name in df.columns:
                if col_name.lower() in qty_patterns:
                    quantity_col_name = col_name
                    break
        
        if not all([date_col_name, item_id_col_name, quantity_col_name]):
            print("✗ Could not auto-detect all required columns")
            print(f"  Found: date={date_col_name}, item_id={item_id_col_name}, quantity={quantity_col_name}")
            print(f"  Available columns: {df.columns}")
            return None
        
        # Rename columns to standard names
        df_renamed = df.withColumnRenamed(date_col_name, DATE_COLUMN)
        df_renamed = df_renamed.withColumnRenamed(item_id_col_name, PRODUCT_ID_COLUMN)
        df_renamed = df_renamed.withColumnRenamed(quantity_col_name, QUANTITY_COLUMN)
        
        print(f"✓ Mapped columns:")
        print(f"  {date_col_name} -> {DATE_COLUMN}")
        print(f"  {item_id_col_name} -> {PRODUCT_ID_COLUMN}")
        print(f"  {quantity_col_name} -> {QUANTITY_COLUMN}")
        
        return df_renamed
    
    def validate_data_types(self, df):
        """
        Validate and convert data types.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            DataFrame: DataFrame with corrected types, or None if validation fails
        """
        print("\n[4/6] Validating and converting data types...")
        
        try:
            # Convert date column
            if DATE_COLUMN in df.columns:
                # Try multiple date formats
                df = df.withColumn(
                    DATE_COLUMN,
                    to_date(col(DATE_COLUMN))
                )
                # Check if conversion succeeded
                null_dates = df.filter(col(DATE_COLUMN).isNull()).count()
                total_rows = df.count()
                if null_dates > 0:
                    print(f"⚠ Warning: {null_dates}/{total_rows} rows have invalid dates")
                    if null_dates == total_rows:
                        print("✗ All dates are invalid. Check date format.")
                        return None
            
            # Convert quantity to double
            if QUANTITY_COLUMN in df.columns:
                df = df.withColumn(QUANTITY_COLUMN, col(QUANTITY_COLUMN).cast(DoubleType()))
                null_qty = df.filter(col(QUANTITY_COLUMN).isNull()).count()
                if null_qty > 0:
                    print(f"⚠ Warning: {null_qty} rows have invalid quantities (will be filtered)")
            
            print("✓ Data type validation passed")
            return df
        
        except Exception as e:
            print(f"✗ Error validating data types: {e}")
            return None
    
    def perform_data_quality_checks(self, df):
        """
        Perform data quality checks.
        
        Args:
            df: DataFrame to check
            
        Returns:
            dict: Dictionary with quality metrics
        """
        print("\n[5/6] Performing data quality checks...")
        
        total_rows = df.count()
        
        # Check for nulls
        null_counts = {
            col_name: df.filter(col(col_name).isNull()).count()
            for col_name in [DATE_COLUMN, PRODUCT_ID_COLUMN, QUANTITY_COLUMN]
            if col_name in df.columns
        }
        
        # Date range
        date_stats = df.select(
            spark_min(DATE_COLUMN).alias("min_date"),
            spark_max(DATE_COLUMN).alias("max_date")
        ).collect()[0]
        
        # Quantity statistics
        qty_stats = df.select(QUANTITY_COLUMN).describe().toPandas()
        
        # Distinct items
        distinct_items = df.select(PRODUCT_ID_COLUMN).distinct().count()
        
        # Print summary
        print(f"✓ Data quality summary:")
        print(f"  Total rows: {total_rows:,}")
        print(f"  Distinct items: {distinct_items:,}")
        print(f"  Date range: {date_stats['min_date']} to {date_stats['max_date']}")
        print(f"  Quantity stats:")
        print(f"    Mean: {float(qty_stats[qty_stats['summary'] == 'mean']['Quantity'].values[0]):.2f}")
        print(f"    Min: {float(qty_stats[qty_stats['summary'] == 'min']['Quantity'].values[0]):.2f}")
        print(f"    Max: {float(qty_stats[qty_stats['summary'] == 'max']['Quantity'].values[0]):.2f}")
        
        if any(null_counts.values()):
            print(f"  Null counts:")
            for col_name, null_count in null_counts.items():
                if null_count > 0:
                    print(f"    {col_name}: {null_count}")
        
        return {
            "total_rows": total_rows,
            "distinct_items": distinct_items,
            "date_range": (date_stats['min_date'], date_stats['max_date']),
            "null_counts": null_counts
        }
    
    def save_data(self, df, output_path, format="csv"):
        """
        Save validated data.
        
        Args:
            df: DataFrame to save
            output_path: Path to save the data
            format: Output format (csv, parquet, delta)
            
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\n[6/6] Saving data to: {output_path}")
        
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save based on format
            if format == "csv":
                (
                    df.coalesce(1)
                    .write
                    .mode("overwrite")
                    .option("header", "true")
                    .csv(output_path)
                )
            elif format == "parquet":
                df.write.mode("overwrite").parquet(output_path)
            elif format == "delta":
                df.write.mode("overwrite").format("delta").save(output_path)
            else:
                print(f"✗ Unknown format: {format}")
                return False
            
            print(f"✓ Data saved successfully")
            return True
        
        except Exception as e:
            print(f"✗ Error saving data: {e}")
            return False
    
    def process_custom_data(
        self,
        input_path,
        output_path,
        date_col=None,
        item_id_col=None,
        quantity_col=None,
        output_format="csv"
    ):
        """
        Complete workflow to process custom data.
        
        Args:
            input_path: Path to input file
            output_path: Path to save processed file
            date_col: Name of date column (optional, will auto-detect)
            item_id_col: Name of item_id column (optional, will auto-detect)
            quantity_col: Name of quantity column (optional, will auto-detect)
            output_format: Format to save (csv, parquet, delta)
            
        Returns:
            bool: True if successful, False otherwise
        """
        print("\n" + "="*80)
        print("CUSTOM DATA UPLOAD AND VALIDATION")
        print("="*80 + "\n")
        
        # 1. Load data
        df = self.load_data(input_path)
        if df is None:
            return False
        
        # 2. Map columns
        df_mapped = self.map_columns(df, date_col, item_id_col, quantity_col)
        if df_mapped is None:
            return False
        
        # 3. Validate schema
        is_valid, errors = self.validate_schema(df_mapped)
        if not is_valid:
            return False
        
        # 4. Validate data types
        df_typed = self.validate_data_types(df_mapped)
        if df_typed is None:
            return False
        
        # Remove rows with null values in critical columns
        df_clean = df_typed.dropna(subset=[DATE_COLUMN, PRODUCT_ID_COLUMN, QUANTITY_COLUMN])
        
        # 5. Data quality checks
        quality_metrics = self.perform_data_quality_checks(df_clean)
        
        # 6. Save data
        success = self.save_data(df_clean, output_path, output_format)
        
        if success:
            print("\n" + "="*80)
            print("✓ DATA PROCESSING COMPLETE!")
            print("="*80)
            print(f"\nYour data is ready for training:")
            print(f"  python main_train_m5.py --data-path {output_path}")
            print(f"\nOr use main_train.py with custom column mappings:")
            print(f"  python main_train.py --data-path {output_path} \\")
            print(f"    --date-column {DATE_COLUMN} \\")
            print(f"    --product-id-column {PRODUCT_ID_COLUMN} \\")
            print(f"    --quantity-column {QUANTITY_COLUMN}")
        
        return success


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Upload and validate custom dataset for forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detect columns)
  python upload_custom_data.py --input-path my_data.csv --output-path data/custom/

  # Specify column names
  python upload_custom_data.py \\
    --input-path my_data.csv \\
    --output-path data/custom/ \\
    --date-col OrderDate \\
    --item-id-col SKU \\
    --quantity-col Sales

  # Save as parquet
  python upload_custom_data.py \\
    --input-path my_data.csv \\
    --output-path data/custom/ \\
    --format parquet
        """
    )
    
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save processed data"
    )
    parser.add_argument(
        "--date-col",
        type=str,
        default=None,
        help="Name of date column (will auto-detect if not provided)"
    )
    parser.add_argument(
        "--item-id-col",
        type=str,
        default=None,
        help="Name of item/product ID column (will auto-detect if not provided)"
    )
    parser.add_argument(
        "--quantity-col",
        type=str,
        default=None,
        help="Name of quantity/demand column (will auto-detect if not provided)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "parquet", "delta"],
        help="Output format (default: csv)"
    )
    
    args = parser.parse_args()
    
    # Create uploader and process data
    uploader = CustomDataUploader()
    success = uploader.process_custom_data(
        input_path=args.input_path,
        output_path=args.output_path,
        date_col=args.date_col,
        item_id_col=args.item_id_col,
        quantity_col=args.quantity_col,
        output_format=args.format
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
