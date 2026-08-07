#!/usr/bin/env python3
"""
make_fused_compute_figure.py - Build fig:fused for
paper_gpu_implementation.tex from the real GPFS benchmark data in
results_bench_gpfs_14178829/fused_compute_{3d_chunked,2d_contiguous}.csv.

2x2 grid: rows are {transform, reduce}, columns are {3-D chunked (512
small chunks), 2-D contiguous (a handful of large row-bands)}. Each panel
plots naive vs. fused wall-clock time (ms) against the synthetic per-piece
compute cost n_ops. Both naive and fused now route through
read_double_buffered/reduce_double_buffered uniformly for both layouts
(see benchmark_fused_compute.py's docstring), so unlike the earlier
chunk-wise-vs-row-band run, fused is never slower than naive here, on
either layout or case -- the figure backs sec:eval:fused's "hidden behind
I/O until compute itself dominates" claim, not the old "regresses on
chunked" one.

Usage
-----
    python make_fused_compute_figure.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_FILES = {
    "3d_chunked": "results_bench_gpfs_14178829/fused_compute_3d_chunked.csv",
    "2d_contiguous": "results_bench_gpfs_14178829/fused_compute_2d_contiguous.csv",
}
_LAYOUT_TITLES = {
    "3d_chunked": "3-D Chunked",
    "2d_contiguous": "2-D Contiguous",
}
_LAYOUT_ORDER = ["3d_chunked", "2d_contiguous"]
_CASE_ORDER = ["transform", "reduce"]
_CASE_TITLES = {"transform": "Transform", "reduce": "Reduce"}

_STYLE = {
    "naive": dict(label="naive", color="#b0413e", marker="s", linestyle="--"),
    "fused": dict(label="fused", color="#2a9d5c", marker="^", linestyle="-"),
}

OUT = "fig_fused_compute.pdf"


def main() -> None:
    # 2x2 grid: rows are {transform, reduce}, columns are the two layouts --
    # the paper's text discusses both cases with real numbers, so the figure
    # needs to actually show both rather than only the reduce row.
    data = {name: pd.read_csv(path) for name, path in _FILES.items()}

    n_ops_vals = sorted(data["3d_chunked"]["n_ops"].unique())
    x_pos = list(range(len(n_ops_vals)))
    pos_of = {n: i for i, n in enumerate(n_ops_vals)}

    fig, axes = plt.subplots(2, 2, figsize=(3.8, 3.4), sharey=False)

    for row, case in enumerate(_CASE_ORDER):
        for col, layout in enumerate(_LAYOUT_ORDER):
            ax = axes[row, col]
            df = data[layout]
            sub = df[df["case"] == case].sort_values("n_ops")
            xs = [pos_of[n] for n in sub["n_ops"]]
            ax.plot(xs, sub["naive_s"] * 1000, markersize=3, linewidth=1.0,
                    **_STYLE["naive"])
            ax.plot(xs, sub["fused_s"] * 1000, markersize=3, linewidth=1.0,
                    **_STYLE["fused"])
            ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
            ax.tick_params(labelsize=7.2)
            ax.set_xticks(x_pos)
            if row == len(_CASE_ORDER) - 1:
                ax.set_xticklabels([str(n) for n in n_ops_vals], rotation=90,
                                    fontsize=6.5)
            else:
                ax.set_xticklabels([])
            if row == 0:
                ax.set_title(_LAYOUT_TITLES[layout], fontsize=7.5, pad=3)
        axes[row, 0].set_ylabel(f"{_CASE_TITLES[case]}\nTime (ms)", fontsize=7.2)

    fig.text(0.5, 0.01, "$n$ (synthetic per-piece compute cost)",
             ha="center", fontsize=7.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7.5,
              bbox_to_anchor=(0.5, 1.0), frameon=False)

    # Explicit margins instead of tight_layout: tight_layout's own spacing
    # algorithm fights with the wide subplot gap this side-by-side layout
    # needs (long titles on each panel collide without it).
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.18, top=0.88,
                        wspace=0.5, hspace=0.35)
    fig.savefig(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
