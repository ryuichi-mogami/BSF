import os
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.abspath("../pymoo"))

from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions

def asf(F, ref_point, w):
    F = np.asarray(F, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float)
    w = np.asarray(w, dtype=float)

    return np.max(w * (F - ref_point), axis=1)

def load_result_csvs(result_dir, run_id, n_ref):
    data = []

    for i in range(1, n_ref + 1):
        filename = None
        for f in os.listdir(result_dir):
            if f.startswith(f"pop_{run_id}th_run_") and f.endswith(f"fevals_ref{i}.csv"):
                filename = f
                break

        if filename is None:
            raise FileNotFoundError(f"CSV for ref{i} was not found: {result_dir}")

        path = os.path.join(result_dir, filename)
        F = np.loadtxt(path, delimiter=",")
        F = np.atleast_2d(F)
        data.append((i, F))

    return data


def main(n_obj, problem_name, alg, roi_type, ref_points, roi_radius, run_id):

    n_ref = len(ref_points)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(n_ref)]
    result_dir = os.path.join(
        f"./results_ref{n_ref}",
        roi_type,
        alg,
        problem_name.upper(),
        f"m{n_obj}"
    )

    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    output_dir = os.path.join(
        "./plot",
        f"results_ref{n_ref}",
        roi_type,
        alg,
        problem_name.upper(),
        f"m{n_obj}"
    )
    os.makedirs(output_dir, exist_ok=True)

    problem = get_problem(problem_name, n_obj=n_obj)
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=3000)

    PF = problem.pareto_front(ref_dirs)

    result_data = load_result_csvs(result_dir, run_id, n_ref)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    ax.scatter(PF[:, 0], PF[:, 1], s=5, alpha=0.2, color="black")

    for i, F in result_data:
        ax.scatter(F[:, 0], F[:, 1], s=40, label=f"ref{i}")

    for i, z in enumerate(ref_points):
        z = np.asarray(z, dtype=float)
        ax.scatter(z[0], z[1], marker="^", s=140, color=colors[i])
    legend_handles = []
    for i, (rp, rr) in enumerate(zip(ref_points, roi_radius)):
        color = colors[i]
        label = (
            f"ref{i+1}: "
            f"point=({rp[0]:.2f}, {rp[1]:.2f}), "
            f"radius=({rr[0]:.2f}, {rr[1]:.2f})"
        )
        legend_handles.append(
            Line2D(
                [0], [0],
                marker='s',
                linestyle='None',
                markerfacecolor=color,
                markeredgecolor=color,
                markersize=6,
                label=label
            )
        )

    ax.legend(handles=legend_handles, loc="lower left")

    if roi_type == "roi-c" or roi_type == "roi-a":
        pf_min = PF.min(axis=0)
        pf_max = PF.max(axis=0)
        span = pf_max - pf_min
        span = np.where(np.abs(span) < 1e-12, 1.0, span)

        for z, r in zip(ref_points, roi_radius):
            z = np.asarray(z, dtype=float)
            r = np.asarray(r, dtype=float)

            if roi_type == "roi-c":
                idx = np.argmin(np.linalg.norm(PF - z, axis=1))
            else:  # roi-a
                w = np.ones_like(PF)
                idx = np.argmin(asf(PF, z, w))

            pivot = PF[idx]

            e = Ellipse(
                xy=(pivot[0], pivot[1]),
                width=2 * span[0] * r[0],
                height=2 * span[1] * r[1],
                fill=False,
                linestyle="--",
                linewidth=1.5,
                edgecolor="black"
            )
            ax.add_patch(e)
    elif roi_type == "roi-p":
        for z in ref_points:
            z = np.asarray(z, dtype=float)
            ax.axvline(z[0], linestyle="--", linewidth=1.0, color="black")
            ax.axhline(z[1], linestyle="--", linewidth=1.0, color="black")

    ax.set_xlabel(r"$f_1$")
    ax.set_ylabel(r"$f_2$")

    plt.tight_layout()

    base_name = f"{problem_name.upper()}_{alg}_{roi_type}_run{run_id}"
    png_path = os.path.join(output_dir, base_name + ".png")
    pdf_path = os.path.join(output_dir, base_name + ".pdf")

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_obj", type=int, required=True)
    parser.add_argument("--problem_name", type=str, required=True)
    parser.add_argument("--alg", type=str, required=True)
    parser.add_argument("--roi_type", type=str, required=True)
    parser.add_argument("--ref_points", type=str, required=True)
    parser.add_argument("--roi_radius", type=str, required=True)
    parser.add_argument("--run_id", type=int, required=True)
    args = parser.parse_args()

    main(
        n_obj=args.n_obj,
        problem_name=args.problem_name,
        alg=args.alg,
        roi_type=args.roi_type,
        ref_points=json.loads(args.ref_points),
        roi_radius=json.loads(args.roi_radius),
        run_id=args.run_id,
    )