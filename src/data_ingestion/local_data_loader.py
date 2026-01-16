"""
Local data ingestion utilities.
"""

from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, to_date

from .m5_transformer import transform_m5_wide_to_long


class DataFormat(str, Enum):
    CSV_LONG = "csv_long"
    PARQUET_LONG = "parquet_long"
    M5_WIDE = "m5_wide"
    UNKNOWN = "unknown"


def _get_spark_session(spark: Optional[SparkSession]) -> SparkSession:
    if spark is not None:
        return spark
    return SparkSession.builder.getOrCreate()


def detect_data_format(
    path: str,
    spark: Optional[SparkSession] = None
) -> Tuple[DataFormat, List[str]]:
    """
    Detect data format based on schema and file type.

    Args:
        path: Path to data file
        spark: Optional SparkSession

    Returns:
        Tuple of (DataFormat, list of column names)
    """
    spark = _get_spark_session(spark)
    data_path = Path(path)

    if data_path.suffix.lower() == ".parquet" or data_path.is_dir():
        df = spark.read.parquet(path)
        columns = df.columns
        if _is_m5_wide(columns):
            return DataFormat.M5_WIDE, columns
        return DataFormat.PARQUET_LONG, columns

    df = (
        spark.read.format("csv")
        .option("header", True)
        .load(path)
    )
    columns = df.columns
    if _is_m5_wide(columns):
        return DataFormat.M5_WIDE, columns

    return DataFormat.CSV_LONG, columns


def _is_m5_wide(columns: List[str]) -> bool:
    day_cols = [c for c in columns if c.startswith("d_")]
    return "item_id" in columns and len(day_cols) > 10


def _standardize_columns(
    df: DataFrame,
    date_col: str,
    product_id_col: str,
    quantity_col: str
) -> DataFrame:
    """
    Rename common column variants to standardized names.
    """
    rename_map: Dict[str, str] = {}

    lower_cols = {c.lower(): c for c in df.columns}

    for candidate in ["orderdate", "date", "ds", "monthenddate"]:
        if candidate in lower_cols and lower_cols[candidate] != date_col:
            rename_map[lower_cols[candidate]] = date_col
            break

    for candidate in ["item_id", "product_id", "sku", "itemnumber", "salesinvoiceproductkey"]:
        if candidate in lower_cols and lower_cols[candidate] != product_id_col:
            rename_map[lower_cols[candidate]] = product_id_col
            break

    for candidate in ["quantity", "demandquantity", "y", "sales", "units"]:
        if candidate in lower_cols and lower_cols[candidate] != quantity_col:
            rename_map[lower_cols[candidate]] = quantity_col
            break

    for old_name, new_name in rename_map.items():
        df = df.withColumnRenamed(old_name, new_name)

    return df


def load_local_data(
    path: str,
    spark: Optional[SparkSession] = None,
    date_col: str = "OrderDate",
    product_id_col: str = "item_id",
    quantity_col: str = "Quantity",
    auto_transform_m5: bool = True,
    m5_start_date: str = "2011-01-29"
) -> DataFrame:
    """
    Load local data from CSV/Parquet or transform M5 wide format to long format.
    """
    spark = _get_spark_session(spark)
    data_format, _ = detect_data_format(path, spark)

    if data_format == DataFormat.M5_WIDE:
        if not auto_transform_m5:
            raise ValueError("M5 wide format detected but auto_transform_m5=False")
        return transform_m5_wide_to_long(
            path=path,
            spark=spark,
            date_col=date_col,
            product_id_col=product_id_col,
            quantity_col=quantity_col,
            start_date=m5_start_date
        )

    if data_format == DataFormat.PARQUET_LONG:
        df = spark.read.parquet(path)
    else:
        df = spark.read.format("csv").option("header", True).load(path)

    df = _standardize_columns(df, date_col, product_id_col, quantity_col)
    df = (
        df.withColumn(date_col, to_date(col(date_col)))
          .withColumn(quantity_col, col(quantity_col).cast("double"))
    )

    return df


class LocalDataLoader:
    """
    Convenience wrapper for loading local data with format detection.
    """

    def __init__(
        self,
        spark: Optional[SparkSession] = None,
        date_col: str = "OrderDate",
        product_id_col: str = "item_id",
        quantity_col: str = "Quantity",
        auto_transform_m5: bool = True,
        m5_start_date: str = "2011-01-29"
    ):
        self.spark = _get_spark_session(spark)
        self.date_col = date_col
        self.product_id_col = product_id_col
        self.quantity_col = quantity_col
        self.auto_transform_m5 = auto_transform_m5
        self.m5_start_date = m5_start_date

    def detect_format(self, path: str) -> Tuple[DataFormat, List[str]]:
        return detect_data_format(path, self.spark)

    def load(self, path: str) -> DataFrame:
        return load_local_data(
            path=path,
            spark=self.spark,
            date_col=self.date_col,
            product_id_col=self.product_id_col,
            quantity_col=self.quantity_col,
            auto_transform_m5=self.auto_transform_m5,
            m5_start_date=self.m5_start_date
        )

    def load_and_validate(self, path: str) -> Tuple[DataFrame, Dict[str, int]]:
        df = self.load(path)
        report = {
            "row_count": df.count(),
            "null_counts": {c: df.filter(col(c).isNull()).count() for c in df.columns}
        }
        return df, report
