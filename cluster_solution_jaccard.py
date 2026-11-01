#!/usr/bin/env python3
"""Compare full-network consensus solutions with and without INR residualization.

This script computes Jaccard indices between the two-subtype consensus
assignments saved by `clustering.ipynb` using the full network template for the
run without INR residualization and the run with INR residualization. K-Means
(`kmeans_2_consensus`) and Louvain (`louvain_consensus`) are compared to
themselves across the two conditions. Only participants assigned a subtype in both runs 
are compared (i.e., subjects missing INR data are dropped).
"""

import argparse
import os
import re
from glob import glob
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import jaccard_score

# Paths and labels
CLUSTERS_DIR = '/Users/emk/Documents/Documents - Ron Weasley V/Research/Verbal-Ability/Final'
NETWORK_TEMPLATE_LABEL = 'fullnetworks'
WITHOUT_INR_LABEL = 'without_inr_residualization'
WITH_INR_LABEL = 'with_inr_residualization'

CONSENSUS_COLUMNS = {
    'K-Means': 'kmeans_2_consensus',
    'Louvain': 'louvain_consensus',
}

def find_latest_clustering_file(clusters_dir, output_label):
    """Return the most recent clustering CSV for a residualization condition."""
    pattern = os.path.join(
        clusters_dir,
        f'{NETWORK_TEMPLATE_LABEL}_meanFC_clusters_{output_label}_*.csv',
    )
    candidates = glob(pattern)
    if not candidates:
        raise FileNotFoundError(f'No clustering output matched: {pattern}')

    def timestamp(path):
        match = re.search(r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})\.csv$', path)
        return match.group(1) if match else ''

    return max(candidates, key=timestamp)


def standardize_subject_id(subject_ids):
    return (
        subject_ids.astype(str)
        .str.replace('_', '', regex=False)
        .str.replace('^sub-', '', regex=True)
    )


def load_consensus_labels(path):
    columns = ['src_subject_id', *CONSENSUS_COLUMNS.values()]
    subjects = pd.read_csv(path, usecols=columns)
    subjects['src_subject_id'] = standardize_subject_id(subjects['src_subject_id'])
    return subjects.drop_duplicates(subset='src_subject_id')


def match_participants(without_inr_run, with_inr_run, subtype_column):
    subjects_in_both_runs = without_inr_run[['src_subject_id', subtype_column]].merge(
        with_inr_run[['src_subject_id', subtype_column]],
        on='src_subject_id',
        how='inner',
        suffixes=('_without_inr', '_with_inr'),
    )
    has_subtype_in_both_runs = subjects_in_both_runs[
        [f'{subtype_column}_without_inr', f'{subtype_column}_with_inr']
    ].notna().all(axis=1)
    subjects_with_subtype_in_both_runs = subjects_in_both_runs.loc[has_subtype_in_both_runs]

    return (
        subjects_with_subtype_in_both_runs[f'{subtype_column}_without_inr'].to_numpy(dtype=int),
        subjects_with_subtype_in_both_runs[f'{subtype_column}_with_inr'].to_numpy(dtype=int),
        int(len(subjects_in_both_runs)),
    )


def jaccard_matrix(without_inr_subtype_per_subject, with_inr_subtype_per_subject):
    without_inr_unique_subtypes = np.unique(without_inr_subtype_per_subject)
    with_inr_unique_subtypes = np.unique(with_inr_subtype_per_subject)

    matrix = np.array([
        [
            jaccard_score(
                without_inr_subtype_per_subject == without_inr_subtype,
                with_inr_subtype_per_subject == with_inr_subtype,
            )
            for with_inr_subtype in with_inr_unique_subtypes
        ]
        for without_inr_subtype in without_inr_unique_subtypes
    ])

    return matrix, without_inr_unique_subtypes, with_inr_unique_subtypes


def match_subtypes(without_inr_subtype_per_subject, with_inr_subtype_per_subject):
    """Match subtypes across conditions by maximum total Jaccard index."""
    matrix, without_inr_unique_subtypes, with_inr_unique_subtypes = jaccard_matrix(
        without_inr_subtype_per_subject, with_inr_subtype_per_subject
    )
    matched_rows, matched_columns = linear_sum_assignment(-matrix) # aligns subtype labels in case numbering differed between INR and no INR runs

    subtype_matches = []
    for row, column in zip(matched_rows, matched_columns):
        without_inr_subtype = without_inr_unique_subtypes[row]
        with_inr_subtype = with_inr_unique_subtypes[column]
        in_without_inr_subtype = without_inr_subtype_per_subject == without_inr_subtype
        in_with_inr_subtype = with_inr_subtype_per_subject == with_inr_subtype
        subtype_matches.append({
            'without_inr_subtype': int(without_inr_subtype) + 1,
            'with_inr_subtype': int(with_inr_subtype) + 1,
            'n_without_inr': int(in_without_inr_subtype.sum()),
            'n_with_inr': int(in_with_inr_subtype.sum()),
            'n_intersection': int(np.logical_and(in_without_inr_subtype, in_with_inr_subtype).sum()),
            'n_union': int(np.logical_or(in_without_inr_subtype, in_with_inr_subtype).sum()),
            'jaccard': matrix[row, column],
        })
    return subtype_matches


def pairwise_jaccard(without_inr_subtype_per_subject, with_inr_subtype_per_subject):
    together_without_inr = [
        subtype_1 == subtype_2
        for subtype_1, subtype_2 in combinations(without_inr_subtype_per_subject, 2)
    ]
    together_with_inr = [
        subtype_1 == subtype_2
        for subtype_1, subtype_2 in combinations(with_inr_subtype_per_subject, 2)
    ]
    return jaccard_score(together_without_inr, together_with_inr)


def compare_conditions(algorithm, subtype_column, without_inr_run, with_inr_run):
    without_inr_subtype_per_subject, with_inr_subtype_per_subject, n_subjects_in_both_runs = (
        match_participants(without_inr_run, with_inr_run, subtype_column)
    )
    n_subjects_compared = len(without_inr_subtype_per_subject)

    records = []
    for subtype_match in match_subtypes(
        without_inr_subtype_per_subject, with_inr_subtype_per_subject
    ):
        records.append({
            'algorithm': algorithm,
            'comparison': (
                f'Subtype {subtype_match["without_inr_subtype"]} without INR versus '
                f'subtype {subtype_match["with_inr_subtype"]} with INR'
            ),
            'jaccard_type': 'per-subtype',
            'jaccard': subtype_match['jaccard'],
            'n_without_inr': subtype_match['n_without_inr'],
            'n_with_inr': subtype_match['n_with_inr'],
            'n_intersection': subtype_match['n_intersection'],
            'n_union': subtype_match['n_union'],
            'n_participants': n_subjects_compared,
        })

    records.append({
        'algorithm': algorithm,
        'comparison': 'Without INR versus with INR (all participant pairs)',
        'jaccard_type': 'pair-counting',
        'jaccard': pairwise_jaccard(
            without_inr_subtype_per_subject, with_inr_subtype_per_subject
        ),
        'n_without_inr': np.nan,
        'n_with_inr': np.nan,
        'n_intersection': np.nan,
        'n_union': np.nan,
        'n_participants': n_subjects_compared,
    })

    return records


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--clusters-dir',
        default=CLUSTERS_DIR,
        help='Directory containing the clustering outputs from clustering.ipynb.',
    )
    parser.add_argument(
        '--without-inr-file',
        default=None,
        help='Clustering CSV without INR residualization (default: most recent).',
    )
    parser.add_argument(
        '--with-inr-file',
        default=None,
        help='Clustering CSV with INR residualization (default: most recent).',
    )
    parser.add_argument(
        '--algorithms',
        nargs='+',
        choices=sorted(CONSENSUS_COLUMNS),
        default=sorted(CONSENSUS_COLUMNS),
        help='Consensus solutions to compare across conditions.',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Path for the results CSV (default: <clusters-dir>/'
             'clustering_inr_residualization_jaccard_fullnetworks.csv).',
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()

    without_inr_path = arguments.without_inr_file or find_latest_clustering_file(
        arguments.clusters_dir, WITHOUT_INR_LABEL
    )
    with_inr_path = arguments.with_inr_file or find_latest_clustering_file(
        arguments.clusters_dir, WITH_INR_LABEL
    )
    print('Without INR residualization:', os.path.basename(without_inr_path))
    print('With INR residualization:', os.path.basename(with_inr_path))

    without_inr_run = load_consensus_labels(without_inr_path)
    with_inr_run = load_consensus_labels(with_inr_path)

    records = []
    for algorithm in arguments.algorithms:
        records.extend(compare_conditions(
            algorithm, CONSENSUS_COLUMNS[algorithm], without_inr_run, with_inr_run
        ))

    results = pd.DataFrame(records)
    results['without_inr_file'] = os.path.basename(without_inr_path)
    results['with_inr_file'] = os.path.basename(with_inr_path)

    print()
    print(
        results[[
            'algorithm', 'comparison', 'jaccard_type', 'jaccard',
            'n_intersection', 'n_union', 'n_participants',
        ]].to_string(index=False, float_format=lambda value: f'{value:.4f}')
    )

    output_path = arguments.output or os.path.join(
        arguments.clusters_dir,
        'clustering_inr_residualization_jaccard_fullnetworks.csv',
    )
    results.to_csv(output_path, index=False)
    print(f'\nSaved Jaccard results to: {output_path}')


if __name__ == '__main__':
    main()
