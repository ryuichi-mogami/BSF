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


EPS = 1e-12


def asf(F, ref_point, w):
    F = np.asarray(F, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float)
    w = np.asarray(w, dtype=float)

    return np.max(w * (F - ref_point), axis=1)


def normalize_objectives(F, ideal, nadir):
    F = np.asarray(F, dtype=float)
    ideal = np.asarray(ideal, dtype=float)
    nadir = np.asarray(nadir, dtype=float)

    denom = nadir - ideal
    denom = np.where(np.abs(denom) < EPS, 1.0, denom)

    return (F - ideal) / denom


def normalize_reference_points(ref_points, ideal, nadir):
    """
    ref_points are always normalized for plotting,
    regardless of the value of --space.
    """
    ref_points = np.asarray(ref_points, dtype=float)
    return normalize_objectives(ref_points, ideal, nadir)


def convert_roi_radius(roi_radius, space, ideal, nadir):
    """
    Convert ROI radii to normalized objective space.

    normalized_space:
        roi_radius is already normalized, so use it as is.

    original_space:
        roi_radius is given in the original objective space,
        so divide by the objective-wise PF range.
    """
    roi_radius = np.asarray(roi_radius, dtype=float)
    ideal = np.asarray(ideal, dtype=float)
    nadir = np.asarray(nadir, dtype=float)

    if space == "normalized_space":
        return roi_radius.copy()

    if space == "original_space":
        denom = nadir - ideal
        denom = np.where(np.abs(denom) < EPS, 1.0, denom)
        return roi_radius / denom

    raise ValueError(
        f"Unknown space: {space}. "
        "Choose 'normalized_space' or 'original_space'."
    )


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


def main(n_obj, problem_name, alg, roi_type, space, ref_points, roi_radius, run_id):

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

    # PF in original objective space
    PF = problem.pareto_front(ref_dirs)
    PF = np.asarray(PF, dtype=float)

    # Estimate ideal and nadir from PF
    ideal = PF.min(axis=0)
    nadir = PF.max(axis=0)

    # Normalize PF for plotting
    PF_norm = normalize_objectives(PF, ideal, nadir)
    # ref_points are ALWAYS normalized for plotting
    ref_points_norm = normalize_reference_points(ref_points, ideal, nadir)

    # roi_radius depends on --space
    roi_radius_norm = convert_roi_radius(roi_radius, space, ideal, nadir)

    result_data = load_result_csvs(result_dir, run_id, n_ref)

    # Normalize populations for plotting
    result_data_norm = []
    for i, F in result_data:
        F_norm = normalize_objectives(F, ideal, nadir)
        result_data_norm.append((i, F_norm))

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Plot normalized PF
    ax.scatter(PF_norm[:, 0], PF_norm[:, 1], s=5, alpha=0.2, color="black")

    # Plot normalized populations
    for i, F_norm in result_data_norm:
        ax.scatter(F_norm[:, 0], F_norm[:, 1], s=40, label=f"ref{i}", color=colors[i - 1])

    # Plot normalized reference points
    for i, z in enumerate(ref_points_norm):
        ax.scatter(z[0], z[1], marker="^", s=140, color=colors[i])

    legend_handles = []
    for i, (rp, rr) in enumerate(zip(ref_points_norm, roi_radius_norm)):
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
        for z, r in zip(ref_points_norm, roi_radius_norm):
            z = np.asarray(z, dtype=float)
            r = np.asarray(r, dtype=float)

            if roi_type == "roi-c":
                idx = np.argmin(np.linalg.norm(PF_norm - z, axis=1))
            else:  # roi-a
                w = np.ones(PF_norm.shape[1], dtype=float)
                idx = np.argmin(asf(PF_norm, z, w))

            pivot = PF_norm[idx]

            e = Ellipse(
                xy=(pivot[0], pivot[1]),
                width=2 * r[0],
                height=2 * r[1],
                fill=False,
                linestyle="--",
                linewidth=1.5,
                edgecolor="black"
            )
            ax.add_patch(e)

    elif roi_type == "roi-p":
        for z in ref_points_norm:
            z = np.asarray(z, dtype=float)
            ax.axvline(z[0], linestyle="--", linewidth=1.0, color="black")
            ax.axhline(z[1], linestyle="--", linewidth=1.0, color="black")

    ax.set_xlabel(r"$f_1$")
    ax.set_ylabel(r"$f_2$")

    plt.tight_layout()

    base_name = f"{problem_name.upper()}_{alg}_{roi_type}_{space}_run{run_id}"
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
    parser.add_argument("--space", type=str, required=True,
                        choices=["normalized_space", "original_space"])
    parser.add_argument("--ref_points", type=str, required=True)
    parser.add_argument("--roi_radius", type=str, required=True)
    parser.add_argument("--run_id", type=int, required=True)
    args = parser.parse_args()

    main(
        n_obj=args.n_obj,
        problem_name=args.problem_name,
        alg=args.alg,
        roi_type=args.roi_type,
        space=args.space,
        ref_points=json.loads(args.ref_points),
        roi_radius=json.loads(args.roi_radius),
        run_id=args.run_id,
    )