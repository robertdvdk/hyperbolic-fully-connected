import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def _load_results(path: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    parsed: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for model, dists in raw.items():
        parsed[model] = {}
        for dist_str, stats in dists.items():
            parsed[model][int(dist_str)] = stats
    return parsed


def _style() -> None:
    try:
        plt.style.use("classic")
    except Exception:
        plt.style.use("default")

    # ICML-style typography and sizing
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.8,
            "axes.linewidth": 0.8,
        }
    )


def _get_palette() -> Dict[str, str]:
    # Colorblind-friendly palette
    return {
        "ours": "#009E73",
        "chen": "#0072B2",
        "poincare": "#D55E00",
    }


def _get_linestyles() -> Dict[str, Tuple[int, Tuple[int, ...]]]:
    # Distinct line styles to separate overlapping curves
    return {
        "ours": (0, ()),          # solid
        "chen": (0, (3, 2)),      # dashed
        "poincare": (0, (1, 2)),  # dotted
    }


def _get_labels() -> Dict[str, str]:
    return {
        "ours": "FGG-LNN",
        "chen": "Chen",
        "poincare": "Poincaré",
    }


def _prepare_series(
    data: Dict[int, Dict[str, Any]]
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    distances = sorted(data.keys())
    means = np.array([data[d]["mean"] for d in distances], dtype=float)
    stds = np.array([data[d]["std"] for d in distances], dtype=float)
    return distances, means, stds


def plot_from_json(
    input_json: Path,
    output_path: Path,
    max_iterations: int = 50000,
    title: str = "Hyperplane fitting convergence vs target hyperbolic distance",
) -> None:
    results = _load_results(input_json)
    palette = _get_palette()
    linestyles = _get_linestyles()
    labels = _get_labels()

    _style()
    # ICML single-column friendly size
    fig, ax = plt.subplots(figsize=(6.75, 3.1))

    for model_name, data in results.items():
        if not data:
            continue
        distances, means, _ = _prepare_series(data)
        color = palette.get(model_name, "#444444")
        linestyle = linestyles.get(model_name, (0, ()))
        label = labels.get(model_name, model_name)

        (line,) = ax.plot(
            distances,
            means,
            label=label,
            color=color,
            linewidth=2.4,
            linestyle=linestyle,
            zorder=3,
        )
        line.set_path_effects(
            [pe.Stroke(linewidth=4.4, foreground="white"), pe.Normal()]
        )

    ax.axhline(
        y=max_iterations,
        color="#666666",
        linestyle=(0, (4, 4)),
        linewidth=1.2,
        alpha=0.7,
        label="Max iterations",
    )

    ax.set_yscale("log")

    ax.set_xlabel("Target hyperbolic distance", fontsize=11)
    ax.set_ylabel("Iterations to convergence", fontsize=11)
    ax.set_ylim(top=20000)
    ax.set_title(title, fontsize=13, pad=10)

    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.grid(True, which="major", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot hyperplane convergence results from JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./hyperplane_convergence_results.json"),
        help="Path to hyperplane_convergence_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./hyperplane_convergence_plot_pretty.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10000,
        help="Max iteration line (for reference)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Convergence vs Hyperbolic Distance",
        help="Plot title",
    )
    args = parser.parse_args()

    plot_from_json(
        input_json=args.input,
        output_path=args.output,
        max_iterations=args.max_iterations,
        title=args.title,
    )
    print(f"✅ Saved plot to '{args.output}'")


if __name__ == "__main__":
    main()
