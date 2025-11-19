#!/usr/bin/env python3
"""
Unified exploratory data analysis entry point for the M5 dataset.

This script replaces the legacy Databricks exports and focuses on a single,
re-runnable workflow that works in both local and Databricks environments.

Key capabilities
----------------
1. Load the canonical M5 sales dataset (wide "d_1"..."d_1913" format) or a
   user-provided dataset that matches the M5 schema.
2. Convert the data into a tidy daily time-series representation.
3. Produce monthly aggregates that mirror what the training pipeline expects.
4. Save lightweight EDA artefacts (CSV summaries, JSON stats, optional plots).

Usage examples
--------------
  # Standard M5 analysis with plots
  python EDA_M5.py --data-path data/raw/m5/sales_train_validation.csv --save-plots

  # Analyse a custom dataset that follows the M5 column layout
  python EDA_M5.py --data-path /tmp/custom_sales.csv --output-dir reports/eda

  # Inspect pre-aggregated daily data with explicit schema hints
  python EDA_M5.py --data-path data/custom_daily.parquet --format long \
      --date-column date --product-column item_id --quantity-column demand
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

try:  # Optional plotting support
    import matplotlib.pyplot as plt
    _HAVE_PLT = True
except ImportError:  # pragma: no cover - plotting is optional
    plt = None
    _HAVE_PLT = False

# Default configuration for the canonical M5 dataset
M5_FIRST_DAY = "2011-01-29"  # Kaggle calendar: d_1 => 2011-01-29
M5_ID_COLUMNS = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id"
]
DAY_COLUMN_PREFIX = "d_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidated EDA workflow for the M5 or M5-compatible datasets."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the M5 sales data (wide CSV) or a custom dataset."
    )
    parser.add_argument(
        "--format",
        choices=["m5_wide", "long"],
        default="m5_wide",
        help="Input format. 'm5_wide' for the original competition layout, "
             "'long' for pre-unpivoted daily records."
    )
    parser.add_argument(
        "--date-column",
        default="date",
        help="Date column name when --format=long."
    )
    parser.add_argument(
        "--product-column",
        default="item_id",
        help="Product identifier column when --format=long."
    )
    parser.add_argument(
        "--quantity-column",
        default="demand",
        help="Demand/quantity column when --format=long."
    )
    parser.add_argument(
        "--output-dir",
        default="eda_outputs",
        help="Directory where summary artefacts will be stored."
    )
    parser.add_argument(
        "--sample-items",
        type=int,
        default=0,
        help="Optional cap on number of items to analyse (0 = use all items)."
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Generate diagnostic plots (requires matplotlib)."
    )
    parser.add_argument(
        "--calendar-start",
        default=M5_FIRST_DAY,
        help="Base date for d_1 when parsing the wide M5 format."
    )
    return parser.parse_args()


def load_m5_wide_csv(path: Path, sample_items: int, calendar_start: str) -> pd.DataFrame:
    """Load the canonical M5 sales CSV and convert to a tidy daily DataFrame."""
    df = pd.read_csv(path)
    missing_ids = [col for col in M5_ID_COLUMNS if col not in df.columns]
    if missing_ids:
        raise ValueError(
            f"The provided file is missing required identifier columns: {missing_ids}"
        )
    day_columns = sorted(
        [col for col in df.columns if col.startswith(DAY_COLUMN_PREFIX)],
        key=lambda c: int(c.split("_")[1])
    )
    if not day_columns:
        raise ValueError("No day columns (d_1 ... d_N) detected in the dataset.")

    if sample_items and sample_items > 0:
        df = df.sample(n=min(sample_items, len(df)), random_state=42)

    tidy = (
        df.melt(
            id_vars=M5_ID_COLUMNS,
            value_vars=day_columns,
            var_name="day_index",
            value_name="demand"
        )
        .assign(
            day_number=lambda d: d["day_index"].str.split("_").str[1].astype(int),
            date=lambda d: pd.to_datetime(calendar_start) + pd.to_timedelta(d["day_number"] - 1, unit="D")
        )
        .drop(columns=["day_index"])
    )
    tidy.rename(columns={"demand": "Quantity", "item_id": "item_id"}, inplace=True)
    return tidy


def load_long_frame(
    path: Path,
    date_column: str,
    product_column: str,
    quantity_column: str
) -> pd.DataFrame:
    """Load a daily transactional dataset that already follows a tidy schema."""
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    expected_cols = {date_column, product_column, quantity_column}
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"The long-format dataset is missing expected columns: {sorted(missing_cols)}"
        )
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df.rename(
        columns={
            date_column: "date",
            product_column: "item_id",
            quantity_column: "Quantity"
        },
        inplace=True
    )
    return df


def compute_monthly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily records to month-end totals per item."""
    monthly = (
        df.assign(
            month_end=lambda d: d["date"].dt.to_period("M").dt.to_timestamp("M")
        )
        .groupby(["item_id", "month_end"], as_index=False)["Quantity"]
        .sum()
    )
    monthly.rename(columns={"month_end": "MonthEndDate"}, inplace=True)
    return monthly


def summarise_dataset(
    daily_df: pd.DataFrame,
    monthly_df: pd.DataFrame
) -> Dict[str, object]:
    """Generate headline metrics for quick inspection."""
    return {
        "daily_records": int(len(daily_df)),
        "products": int(daily_df["item_id"].nunique()),
        "date_range": {
            "start": daily_df["date"].min().strftime("%Y-%m-%d"),
            "end": daily_df["date"].max().strftime("%Y-%m-%d")
        },
        "monthly_records": int(len(monthly_df)),
        "total_demand": float(daily_df["Quantity"].sum()),
        "average_monthly_demand": float(monthly_df["Quantity"].mean()),
    }


def create_plots(daily_df: pd.DataFrame, monthly_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate simple diagnostic plots if matplotlib is available."""
    if not _HAVE_PLT:  # pragma: no cover - optional output
        print("matplotlib not available; skipping plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Demand distribution
    plt.figure(figsize=(8, 5))
    daily_df["Quantity"].clip(upper=daily_df["Quantity"].quantile(0.99)).hist(bins=50)
    plt.title("Daily demand distribution (clipped at 99th percentile)")
    plt.xlabel("Quantity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "daily_demand_histogram.png")
    plt.close()

    # Monthly totals over time (overall)
    monthly_totals = (
        monthly_df.groupby("MonthEndDate", as_index=False)["Quantity"].sum()
    )
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_totals["MonthEndDate"], monthly_totals["Quantity"])
    plt.title("Total demand per month")
    plt.xlabel("Month end")
    plt.ylabel("Quantity")
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_totals.png")
    plt.close()

    # Top products
    top_items = (
        monthly_df.groupby("item_id")["Quantity"].sum().sort_values(ascending=False).head(5).index
    )
    plt.figure(figsize=(10, 6))
    for item in top_items:
        item_series = monthly_df[monthly_df["item_id"] == item]
        plt.plot(item_series["MonthEndDate"], item_series["Quantity"], label=item)
    plt.title("Top 5 items – monthly demand")
    plt.xlabel("Month end")
    plt.ylabel("Quantity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "top_items_timeseries.png")
    plt.close()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format == "m5_wide":
        daily_df = load_m5_wide_csv(
            path=data_path,
            sample_items=args.sample_items,
            calendar_start=args.calendar_start
        )
    else:
        daily_df = load_long_frame(
            path=data_path,
            date_column=args.date_column,
            product_column=args.product_column,
            quantity_column=args.quantity_column
        )

    monthly_df = compute_monthly_aggregates(daily_df)
    stats = summarise_dataset(daily_df, monthly_df)

    # Persist artefacts
    daily_sample = daily_df.sample(n=min(5000, len(daily_df)), random_state=42)
    daily_sample.to_csv(output_dir / "daily_sample.csv", index=False)
    monthly_df.to_csv(output_dir / "monthly_demand.csv", index=False)
    with open(output_dir / "summary_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    # Save optional plots
    if args.save_plots:
        create_plots(daily_df, monthly_df, output_dir)

    print(json.dumps(stats, indent=2))
    print(f"\nArtefacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
