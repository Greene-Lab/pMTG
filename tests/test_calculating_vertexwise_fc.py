import numpy as np

from calculating_vertexwise_fc import calculate_subject_fc, fisher_z_correlations


def test_fisher_z_correlations_matches_numpy_corrcoef():
    vertices = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [4.0, 3.0],
            [8.0, 7.0],
        ]
    )
    network = np.array([1.0, 3.0, 2.0, 6.0])

    observed = fisher_z_correlations(vertices, network)
    expected = np.array(
        [np.arctanh(np.corrcoef(vertices[:, i], network)[0, 1]) for i in range(2)]
    )

    assert np.allclose(observed, expected)


def test_subject_fc_keeps_each_vertex_and_labels_python_index_and_hemisphere():
    dt = np.array(
        [
            [1.0, 2.0, 4.0, 3.0, 8.0],
            [2.0, 1.0, 5.0, 4.0, 7.0],
            [3.0, 4.0, 2.0, 6.0, 5.0],
            [5.0, 3.0, 1.0, 8.0, 2.0],
        ]
    )
    pmtg_vertices = {"L": np.array([1, 3]), "R": np.array([4])}
    network_indices = {"DMN_full": np.array([0, 2])}

    result = calculate_subject_fc(dt, pmtg_vertices, network_indices)

    assert list(result) == [
        "DMN_full_1_L_fz",
        "DMN_full_3_L_fz",
        "DMN_full_4_R_fz",
    ]
    network_timecourse = dt[:, [0, 2]].mean(axis=1)
    assert np.isclose(
        result["DMN_full_3_L_fz"],
        np.arctanh(np.corrcoef(dt[:, 3], network_timecourse)[0, 1]),
    )
