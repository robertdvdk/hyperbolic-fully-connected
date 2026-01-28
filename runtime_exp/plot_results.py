from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_COLUMNS = [
	"Fwd Mean (ms)",
	"Fwd Std (ms)",
]


def _normalize_compiled_column(df: pd.DataFrame) -> pd.DataFrame:
	if "Compiled" not in df.columns:
		return df
	compiled_str = df["Compiled"].astype(str).str.strip().str.lower()
	df["Compiled"] = compiled_str.isin({"true", "1", "yes"})
	return df


def _coerce_metrics(df: pd.DataFrame) -> pd.DataFrame:
	for col in METRIC_COLUMNS:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")
	return df


def load_results(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	df = _normalize_compiled_column(df)
	df = _coerce_metrics(df)
	return df


def plot_fwd_mean_with_std(
	df: pd.DataFrame,
	x_col: str,
	output_path: Path,
):
	output_path.parent.mkdir(parents=True, exist_ok=True)

	plt.figure(figsize=(10, 6))
	for model_name in sorted(df["Model"].unique()):
		subset = df[df["Model"] == model_name].copy()
		subset = subset.sort_values(x_col)
		x_values = subset[x_col].to_numpy(dtype=float)
		mean_values = subset["Fwd Mean (ms)"].to_numpy(dtype=float)
		std_values = subset["Fwd Std (ms)"].to_numpy(dtype=float)
		mask = np.isfinite(x_values) & np.isfinite(mean_values) & np.isfinite(std_values)
		if mask.sum() == 0:
			continue
		x_values = x_values[mask]
		mean_values = mean_values[mask]
		std_values = std_values[mask]

		plt.plot(
			x_values,
			mean_values,
			marker="o",
			linewidth=1.8,
			markersize=4,
			label=model_name,
		)
		plt.fill_between(
			x_values,
			mean_values - std_values,
			mean_values + std_values,
			alpha=0.2,
		)

	plt.title("Forward runtime vs dimension")
	plt.xlabel("Dimension")
	plt.ylabel("Fwd Mean (ms)")
	plt.xscale("log", base=2)
	plt.grid(True, alpha=0.3)
	plt.legend(fontsize=8, ncol=2)
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()


def main():
	parser = argparse.ArgumentParser(description="Plot runtime benchmark results.")
	parser.add_argument(
		"--csv",
		type=Path,
		default=Path(__file__).with_name("runtime_results_mine.csv"),
		help="Path to the CSV file.",
	)
	parser.add_argument(
		"--outdir",
		type=Path,
		default=Path(__file__).with_name("plots"),
		help="Directory to save plots.",
	)
	args = parser.parse_args()

	df = load_results(args.csv)

	if df.empty:
		raise SystemExit("No data available after filtering.")

	max_batch = df["Batch"].max()
	sweep_df = df[df["Batch"] == max_batch]
	output_path = args.outdir / "runtime_fwd_mean_with_std.png"
	plot_fwd_mean_with_std(sweep_df, x_col="In", output_path=output_path)


if __name__ == "__main__":
	main()
