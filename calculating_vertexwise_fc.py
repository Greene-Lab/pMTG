#!/usr/bin/env python3
"""Compute vertex-wise pMTG-to-network functional connectivity.

This is the vertex-wise counterpart to ``calculating_fc.py``.  For every
eligible subject, it correlates the time series of each vertex in the left and
right pMTG parcels with each group-average network time series.  Unlike the
original script, it does not average FC values across pMTG vertices.

Output columns use the following convention:

    <network>_<Python dense index>_<hemisphere>_fz

For example, ``DMN_full_12345_L_fz`` contains Fisher-z transformed FC between
left-pMTG vertex 12345 (its zero-based Python/CIFTI dense index) and the
generated ``DMN_full`` network summary.
"""

import os
import subprocess
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


# Update these paths if the input data or Workbench installation moves.
WORKBENCH = "/labs/greene-lab/Shared_Tools/workbench/bin_linux64/wb_command"
SURFACE_FILES = {
    "Left": (
        "/labs/greene-lab/lab_members/emily/mtg/surfaces/HCP1200/"
        "S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
    ),
    "Right": (
        "/labs/greene-lab/lab_members/emily/mtg/surfaces/HCP1200/"
        "S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii"
    ),
}

DENSITY_PATH = (
    "/labs/greene-lab/lab_members/emily/mtg/maps/"
    "Variants_Density_cPFM_not06not11not13_185avrg.dtseries.nii"
)
AVG_PATH = (
    "/labs/greene-lab/lab_members/emily/mtg/maps/"
    "abcd_template_matching_combined_clusters_thresh0.61.dlabel.nii"
)
MOTION_PATH = "/labs/greene-lab/lab_members/emily/mtg/data/motion_data_summary.csv"
BASE_DIR = "/labs/greene-lab/ABCD/derivatives/GL_Smoothing"

CLUSTER_OUTPUT_PATH = (
    "/labs/greene-lab/lab_members/emily/mtg/maps/"
    "Variants_Density_cPFM_not06not11not13_185avrg_thresholded_clusters.dtseries.nii"
)
PMTG_OUTPUT_PATH = (
    "/labs/greene-lab/lab_members/emily/mtg/maps/pMTG_regions.dtseries.nii"
)
OUTPUT_PATH = (
    "/labs/greene-lab/lab_members/emily/mtg/data/"
    "pMTG_FC_profiles_midb61_vertexwiseFC.csv"
)

# Cluster labels identifying pMTG in the Workbench cluster output.
LEFT_PMTG_CLUSTER = 2
RIGHT_PMTG_CLUSTER = 7

NETWORK_LABELS = {
    "DMN_left": [1, 16, 18, 28, 30, 73, 74],
    "DMN_right": [33, 49, 58, 60, 62, 81, 84],
    "SMl_left": [2, 79, 96, 104],
    "SMl_right": [34, 90, 100, 106],
    "Vis_left": [3, 77, 91, 103],
    "Vis_right": [35, 89, 92, 105],
    "FP_left": [4, 21, 27, 29, 71, 76],
    "FP_right": [36, 37, 53, 59, 61, 82, 86, 87],
    "Aud_left": [5, 98],
    "Aud_right": [38, 102],
    "Tpole_left": [6, 64],
    "Tpole_right": [39, 65],
    "PMN_left": [7, 24],
    "PMN_right": [40, 55],
    "PON_left": [8, 26],
    "PON_right": [41, 57],
    "MTL_left": [9, 93],
    "MTL_right": [42, 94],
    "CO_left": [10, 17, 20, 32, 69, 75, 97],
    "CO_right": [43, 48, 51, 63, 101, 107],
    "Sal_left": [11, 23, 31, 66, 95],
    "Sal_right": [44, 54, 67, 99],
    "SMd_left": [12, 13, 68, 78],
    "SMd_right": [45, 80, 88],
    "DAN_left": [14, 15, 25, 70, 72],
    "DAN_right": [46, 47, 56, 83, 85],
    "VAN_left": [19, 22],
    "VAN_right": [50, 52],
    "DMN_full": [1, 16, 18, 28, 30, 33, 49, 58, 60, 62, 73, 74, 81, 84],
    "SMl_full": [2, 34, 79, 90, 96, 100, 104, 106],
    "Vis_full": [3, 35, 77, 89, 91, 92, 103, 105],
    "FP_full": [4, 21, 27, 29, 36, 37, 53, 59, 61, 71, 76, 82, 86, 87],
    "Aud_full": [5, 38, 98, 102],
    "Tpole_full": [6, 39, 64, 65],
    "PMN_full": [7, 24, 40, 55],
    "PON_full": [8, 26, 41, 57],
    "MTL_full": [9, 42, 93, 94],
    "CO_full": [10, 17, 20, 32, 43, 48, 51, 63, 69, 75, 97, 101, 107],
    "Sal_full": [11, 23, 31, 44, 54, 66, 67, 95, 99],
    "SMd_full": [12, 13, 45, 68, 78, 80, 88],
    "DAN_full": [14, 15, 25, 46, 47, 56, 70, 72, 83, 85],
    "VAN_full": [19, 22, 50, 52],
}


def find_pmtg_vertices():
    """Create the cluster map and return zero-based dense indices for each pMTG."""
    subprocess.run(
        [
            WORKBENCH,
            "-cifti-find-clusters",
            DENSITY_PATH,
            "4",
            "40",
            "0",
            "0",
            "COLUMN",
            CLUSTER_OUTPUT_PATH,
            "-left-surface",
            SURFACE_FILES["Left"],
            "-right-surface",
            SURFACE_FILES["Right"],
        ],
        check=True,
    )

    clusters_img = nib.load(CLUSTER_OUTPUT_PATH)
    clusters = np.asarray(clusters_img.get_fdata()).squeeze()
    if clusters.ndim != 1:
        raise ValueError(
            f"Expected a one-map cluster CIFTI; got data shape {clusters.shape}."
        )

    pmtg_vertices = {
        "L": np.flatnonzero(clusters == LEFT_PMTG_CLUSTER),
        "R": np.flatnonzero(clusters == RIGHT_PMTG_CLUSTER),
    }
    for hemisphere, indices in pmtg_vertices.items():
        if indices.size == 0:
            raise ValueError(f"No {hemisphere} pMTG vertices were found.")
        print(f"{hemisphere} pMTG Python indices ({indices.size}): {indices}")

    return clusters_img, pmtg_vertices


def save_pmtg_mask(clusters_img, pmtg_vertices):
    """Save a two-valued pMTG mask (left=1, right=2) for visual inspection."""
    n_grayordinates = clusters_img.shape[-1]
    mask = np.zeros((1, n_grayordinates), dtype=np.float32)
    mask[0, pmtg_vertices["L"]] = 1
    mask[0, pmtg_vertices["R"]] = 2
    nib.save(nib.Cifti2Image(mask, header=clusters_img.header), PMTG_OUTPUT_PATH)
    print(f"Saved pMTG mask to {PMTG_OUTPUT_PATH}")


def get_network_indices(pmtg_vertices):
    """Map network parcel labels to dense indices, excluding both pMTG parcels."""
    avg_data = np.asarray(nib.load(AVG_PATH).get_fdata()).squeeze()
    if avg_data.ndim != 1:
        raise ValueError(f"Expected a one-map network CIFTI; got {avg_data.shape}.")

    all_pmtg = np.concatenate([pmtg_vertices["L"], pmtg_vertices["R"]])
    network_indices = {}
    for network, labels in NETWORK_LABELS.items():
        indices = np.flatnonzero(np.isin(avg_data, labels))
        indices = np.setdiff1d(indices, all_pmtg, assume_unique=True)
        if indices.size == 0:
            raise ValueError(f"Network {network} contains no vertices after exclusion.")
        network_indices[network] = indices
        print(f"{network}: {indices.size} non-pMTG vertices")

    return network_indices


def fisher_z_correlations(vertex_timecourses, network_timecourse):
    """Correlate columns with one time course and return Fisher-z values."""
    vertices = np.asarray(vertex_timecourses, dtype=np.float64)
    network = np.asarray(network_timecourse, dtype=np.float64)
    if vertices.ndim != 2 or network.ndim != 1:
        raise ValueError("Vertex time courses must be 2-D and network time course 1-D.")
    if vertices.shape[0] != network.shape[0]:
        raise ValueError("Vertex and network time courses have different lengths.")

    vertices = vertices - vertices.mean(axis=0)
    network = network - network.mean()
    denominator = np.linalg.norm(vertices, axis=0) * np.linalg.norm(network)

    correlations = np.full(vertices.shape[1], np.nan, dtype=np.float64)
    valid = denominator > 0
    correlations[valid] = (network @ vertices[:, valid]) / denominator[valid]

    # Floating-point rounding can put a mathematically valid correlation just
    # outside [-1, 1]. Clipping also prevents infinite arctanh values.
    epsilon = np.finfo(np.float64).eps
    correlations[valid] = np.clip(
        correlations[valid], -1.0 + epsilon, 1.0 - epsilon
    )
    return np.arctanh(correlations)


def calculate_subject_fc(dt, pmtg_vertices, network_indices):
    """Calculate all network-by-pMTG-vertex FC values for one subject."""
    dt = np.asarray(dt)
    if dt.ndim != 2:
        raise ValueError(f"Expected time-by-grayordinate data; got shape {dt.shape}.")

    subject_result = {}
    for network, indices in network_indices.items():
        network_timecourse = np.mean(dt[:, indices], axis=1)
        for hemisphere in ("L", "R"):
            vertex_indices = pmtg_vertices[hemisphere]
            fz_values = fisher_z_correlations(
                dt[:, vertex_indices], network_timecourse
            )
            for python_index, fz_value in zip(vertex_indices, fz_values):
                column = f"{network}_{int(python_index)}_{hemisphere}_fz"
                subject_result[column] = float(fz_value)

    return subject_result


def find_subject_dtseries(subject_path):
    """Return the first expected smoothed CIFTI for a subject, if present."""
    smoothed_dir = os.path.join(
        subject_path, "ses-2YearFollowUpYArm1", "Smoothed"
    )
    if not os.path.isdir(smoothed_dir):
        return None

    matches = sorted(
        glob(os.path.join(smoothed_dir, "*_CENSORED_6.0mm_SMOOTHED.dtseries.nii"))
    )
    return matches[0] if matches else None


def main():
    """Run vertex-wise FC for every motion-eligible subject."""
    clusters_img, pmtg_vertices = find_pmtg_vertices()
    save_pmtg_mask(clusters_img, pmtg_vertices)
    network_indices = get_network_indices(pmtg_vertices)

    motion = pd.read_csv(MOTION_PATH)
    eligible_subjects = set(
        motion.loc[motion["good_frames"] >= 600, "src_subject_id"].astype(str)
    )
    print(f"Eligible subjects: {len(eligible_subjects)}")

    results = []
    for subject_path in sorted(glob(os.path.join(BASE_DIR, "sub-*"))):
        subject_id = Path(subject_path).name
        if subject_id not in eligible_subjects:
            continue

        dt_path = find_subject_dtseries(subject_path)
        if dt_path is None:
            print(f"Skipping {subject_id}: smoothed dtseries not found")
            continue

        try:
            dt = nib.load(dt_path).get_fdata()
            subject_result = {"subject_id": subject_id}
            subject_result.update(
                calculate_subject_fc(dt, pmtg_vertices, network_indices)
            )
            results.append(subject_result)
            print(f"Processed {subject_id}")
        except Exception as error:
            print(f"Error processing {subject_id}: {error}")

    if not results:
        raise RuntimeError("No eligible subjects were successfully processed.")

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Saved {len(results_df)} subjects and "
        f"{len(results_df.columns) - 1} FC columns to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
