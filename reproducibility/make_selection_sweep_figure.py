#!/usr/bin/env python3
"""
make_selection_sweep_figure.py - Build fig:selection (read) and fig:write
(write) for paper_gpu_implementation.tex from the real GPFS benchmark data,
sized to sit side by side as two subfigures within one figure* block.

One subplot per layout (2-D/3-D x chunked/contiguous), each plotting
wall-clock time (ms) vs. selection coverage (%) for naive-full,
naive-partial, and ours.

Usage
-----
    python make_selection_sweep_figure.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_LAYOUT_TITLES = {
    "2d_chunked": "2-D Chunked",
    "2d_contiguous": "2-D Contiguous",
    "3d_chunked": "3-D Chunked",
    "3d_contiguous": "3-D Contiguous",
}
_LAYOUT_ORDER = ["2d_chunked", "2d_contiguous", "3d_chunked", "3d_contiguous"]

_METHOD_STYLE = {
    "naive_full":    dict(label="naive-full",    color="#b0413e", marker="s", linestyle="--"),
    "naive_partial": dict(label="naive-partial", color="#4c72b0", marker="o", linestyle="-"),
    "ours":          dict(label="ours",          color="#2a9d5c", marker="^", linestyle="-"),
}
_METHOD_ORDER = ["naive_full", "naive_partial", "ours"]

# Native size tuned for a subfigure occupying ~0.48\textwidth side by side
# with its counterpart, not a scaled-down copy of a wider image -- keeps
# absolute font size (and hence legibility) the same as a standalone figure.
_FIGSIZE = (2.7, 2.3)


def _make_figure(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    means = df.groupby(["layout", "coverage_pct", "method"])["time_s"].mean().reset_index()
    means["time_ms"] = means["time_s"] * 1000.0

    y_min = means["time_ms"].min() * 0.85
    y_max = means["time_ms"].max() * 1.15

    fig, axes = plt.subplots(2, 2, figsize=_FIGSIZE, sharex=True, sharey=True)

    for ax, layout in zip(axes.flat, _LAYOUT_ORDER):
        sub = means[means["layout"] == layout]
        for method in _METHOD_ORDER:
            m = sub[sub["method"] == method].sort_values("coverage_pct")
            style = _METHOD_STYLE[method]
            ax.plot(m["coverage_pct"], m["time_ms"], markersize=4,
                    linewidth=1.2, **style)
        ax.set_title(_LAYOUT_TITLES[layout], fontsize=6.3)
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.tick_params(labelsize=5.5)

    for ax in axes[-1, :]:
        ax.set_xlabel("Coverage (%)", fontsize=5.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Time (ms, log)", fontsize=5.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=5.5,
              bbox_to_anchor=(0.5, 1.04), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main() -> None:
    _make_figure("results_bench_gpfs_14150027/selection_sweep.csv",
                "fig_selection_sweep_new.pdf")
    _make_figure("results_bench_gpfs_14166199/write_selection_sweep.csv",
                "fig_write_selection_sweep.pdf")


if __name__ == "__main__":
    main()
