#!/usr/bin/env python
"""
Deprecated local M5 training wrapper.

Use main_train_local.py for unified local ingestion and training.
"""

import argparse
import sys

from main_train_local import train_local


def main() -> int:
    parser = argparse.ArgumentParser(description="Train forecasting models on M5 dataset (local)")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/m5/sales_train_validation.csv",
        help="Path to M5 sales_train_validation.csv file"
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        default="/M5_Forecasting_Local",
        help="MLflow experiment name prefix"
    )
    parser.add_argument(
        "--m5-start-date",
        type=str,
        default="2011-01-29",
        help="Start date for M5 dataset"
    )
    parser.add_argument(
        "--no-transform-m5",
        action="store_true",
        help="Disable automatic M5 wide-format transformation"
    )
    args = parser.parse_args()

    result = train_local(
        data_path=args.data_path,
        experiment_prefix=args.experiment_prefix,
        m5_start_date=args.m5_start_date,
        auto_transform_m5=not args.no_transform_m5
    )
    print(f"Training result: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
