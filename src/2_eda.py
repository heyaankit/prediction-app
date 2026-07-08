"""
Stage 2 of the DVC pipeline for the prediction-app project.

Performs Exploratory Data Analysis on the ingested dataset and produces
visual artifacts (PNG plots) + a summary statistics CSV. All outputs
are tracked by DVC so they're versioned alongside the data.

Input:  data/processed/ingested.csv   (produced by data_ingestion.py)
Outputs:
  - reports/eda/missing_values.png
  - reports/eda/demand_distribution.png
  - reports/eda/demand_by_region.png
  - reports/eda/demand_by_weather.png
  - reports/eda/demand_by_category.png
  - reports/eda/promotion_effect.png
  - reports/eda/demand_over_time.png
  - reports/eda/correlation_heatmap.png
  - reports/eda/summary_stats.csv

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eda")

DEFAULT_PARAMS_PATH = "params.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_params(path: str = DEFAULT_PARAMS_PATH) -> dict:
    p = Path(path)
    if not p.exists():
        log.warning("%s not found — using built-in defaults.", path)
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_config(args: argparse.Namespace) -> dict:
    params = load_params(args.params)
    section = params.get("eda", {}) if isinstance(params, dict) else {}
    return {
        "input_path": args.input_path
        or section.get("input_path", "data/processed/ingested.csv"),
        "output_dir": args.output_dir
        or section.get("output_dir", "reports/eda"),
        "target_column": section.get("target_column", "Demand"),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def setup_seaborn() -> None:
    """Consistent visual style for all EDA plots."""
    sns.set_style("whitegrid")
    sns.set_context("notebook", rc={"figure.figsize": (10, 5)})


def plot_missing_values(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of missing value counts per column (notebook cell 25)."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        log.info("No missing values found — skipping missing_values plot.")
        # Still write a placeholder so DVC out exists.
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No missing values in any column",
                ha="center", va="center", fontsize=14)
        ax.set_axis_off()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    missing.plot.bar(ax=ax, color="coral")
    ax.set_title("Missing Values per Column")
    ax.set_ylabel("Count of missing rows")
    ax.set_xlabel("Column")
    plt.xticks(rotation=45, ha="right")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def plot_demand_distribution(df: pd.DataFrame, target: str, out_path: Path) -> None:
    """Histogram + KDE of the target variable."""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[target], bins=50, kde=True, color="steelblue", ax=ax)
    ax.set_title(f"Distribution of {target}")
    ax.set_xlabel(target)
    ax.set_ylabel("Frequency")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def plot_demand_by_categorical(
    df: pd.DataFrame, target: str, col: str, out_path: Path
) -> None:
    """Boxplot of target grouped by a categorical column."""
    n_unique = df[col].nunique()
    if n_unique == 0:
        log.warning("Column '%s' has no values — skipping plot.", col)
        return
    fig, ax = plt.subplots(figsize=(max(10, n_unique * 0.5), 5))
    order = df.groupby(col)[target].median().sort_values().index
    sns.boxplot(data=df, x=col, y=target, order=order, ax=ax, palette="Set2")
    ax.set_title(f"{target} by {col}")
    ax.set_xlabel(col)
    ax.set_ylabel(target)
    plt.xticks(rotation=45, ha="right")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s (n_unique=%d)", out_path, n_unique)


def plot_promotion_effect(df: pd.DataFrame, target: str, out_path: Path) -> None:
    """Bar chart of mean demand by promotion flag (notebook cell 15)."""
    promo_effect = df.groupby("Promotion")[target].mean()
    log.info("Promotion effect:\n%s", promo_effect)
    fig, ax = plt.subplots(figsize=(6, 5))
    promo_effect.plot.bar(ax=ax, color=["steelblue", "coral"])
    ax.set_title(f"Mean {target} by Promotion Flag")
    ax.set_xlabel("Promotion (0 = No, 1 = Yes)")
    ax.set_ylabel(f"Mean {target}")
    plt.xticks(rotation=0)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def plot_demand_over_time(df: pd.DataFrame, target: str, out_path: Path) -> None:
    """Time series of target averaged per date (line plot)."""
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    daily = df.groupby("Date")[target].mean().sort_index()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily.index, daily.values, color="steelblue", linewidth=0.8, alpha=0.8)
    ax.set_title(f"{target} Over Time (daily mean)")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Mean {target}")
    fig.autofmt_xdate()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Correlation heatmap of numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        log.warning("Fewer than 2 numeric columns — skipping correlation heatmap.")
        return
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title("Correlation Heatmap (numeric features)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


def save_summary_stats(df: pd.DataFrame, out_path: Path) -> None:
    """Save df.describe() to CSV (notebook cell 26)."""
    stats = df.describe(include="all")
    stats.to_csv(out_path)
    log.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DVC stage: EDA")
    p.add_argument("--params", default=DEFAULT_PARAMS_PATH)
    p.add_argument("--input-path", default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config(args)

    in_path = Path(cfg["input_path"])
    if not in_path.exists():
        raise FileNotFoundError(
            f"Input not found: {in_path}. Run data_ingestion.py first."
        )

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Reading ingested data: %s", in_path)
    df = pd.read_csv(in_path)
    log.info("Loaded %d rows x %d columns.", *df.shape)
    log.info("Columns: %s", list(df.columns))

    # Print key insights to stdout (visible in `dvc repro` output)
    log.info("=" * 60)
    log.info("KEY INSIGHTS")
    log.info("=" * 60)
    log.info("Shape: %s", df.shape)
    log.info("Date range: %s → %s", df["Date"].min(), df["Date"].max())
    for col in ["Region", "Weather Condition", "Category"]:
        if col in df.columns:
            log.info("%s unique values: %s", col, df[col].unique().tolist())
    log.info("Promotion effect (mean %s):\n%s",
             cfg["target_column"],
             df.groupby("Promotion")[cfg["target_column"]].mean())
    log.info("Missing values:\n%s", df.isnull().sum()[df.isnull().sum() > 0])
    log.info("=" * 60)

    # Generate all plots
    setup_seaborn()
    plot_missing_values(df, out_dir / "missing_values.png")
    plot_demand_distribution(df, cfg["target_column"], out_dir / "demand_distribution.png")
    if "Region" in df.columns:
        plot_demand_by_categorical(df, cfg["target_column"], "Region",
                                   out_dir / "demand_by_region.png")
    if "Weather Condition" in df.columns:
        plot_demand_by_categorical(df, cfg["target_column"], "Weather Condition",
                                   out_dir / "demand_by_weather.png")
    if "Category" in df.columns:
        plot_demand_by_categorical(df, cfg["target_column"], "Category",
                                   out_dir / "demand_by_category.png")
    if "Promotion" in df.columns:
        plot_promotion_effect(df, cfg["target_column"], out_dir / "promotion_effect.png")
    if "Date" in df.columns:
        plot_demand_over_time(df, cfg["target_column"], out_dir / "demand_over_time.png")
    plot_correlation_heatmap(df, out_dir / "correlation_heatmap.png")
    save_summary_stats(df, out_dir / "summary_stats.csv")

    log.info("EDA complete. All artifacts in %s", out_dir)


if __name__ == "__main__":
    main()
