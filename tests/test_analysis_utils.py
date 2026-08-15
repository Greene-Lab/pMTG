import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_utils as au


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_standard_fc_covariates_include_motion_qa_mean_fd():
    assert "mean_fd_0.20" in au.STANDARD_FC_COVARIATES


def test_default_categorical_covariates_do_not_include_analysis_exclusions():
    assert au.DEFAULT_CATEGORICAL_COVARIATES == {
        "demo_sex_v2",
        "site_id_l",
        "ehi1b",
    }
    assert "interview_age" in au.STANDARD_FC_COVARIATES
    assert "site_id_l" in au.STANDARD_FC_COVARIATES
    assert "ehi1b" in au.STANDARD_FC_COVARIATES
    assert "mri_info_softwareversion" not in au.STANDARD_FC_COVARIATES
    assert "rel_family_id" not in au.DEFAULT_CATEGORICAL_COVARIATES
    assert "demo_comb_income_v2" not in au.DEFAULT_CATEGORICAL_COVARIATES


def test_merge_motion_qa_standardizes_subject_ids_and_keeps_left_rows(tmp_path):
    motion_path = tmp_path / "motion_QA_results.csv"
    pd.DataFrame(
        {
            "src_subject_id": ["sub-NDAR_001", "sub-NDAR_002"],
            "mean_fd_0.20": [0.12, 0.34],
        }
    ).to_csv(motion_path, index=False)

    df = pd.DataFrame({"src_subject_id": ["NDAR001", "NDAR003"], "value": [1, 2]})
    merged = au.merge_motion_qa(df, motion_path, how="left")

    assert len(merged) == 2
    assert merged.loc[merged["src_subject_id"] == "NDAR001", "mean_fd_0.20"].iloc[0] == 0.12
    assert np.isnan(
        merged.loc[merged["src_subject_id"] == "NDAR003", "mean_fd_0.20"].iloc[0]
    )


def test_preprocess_ravlt_data_drops_rows_with_incomplete_immediate_trials():
    ravlt = pd.DataFrame(
        {
            "src_subject_id": ["s1", "s2"],
            "pea_ravlt_sd_trial_i_tc": [1, 1],
            "pea_ravlt_sd_trial_ii_tc": [2, np.nan],
            "pea_ravlt_sd_trial_iii_tc": [3, 3],
            "pea_ravlt_sd_trial_iv_tc": [4, 4],
            "pea_ravlt_sd_trial_v_tc": [5, 5],
            "pea_ravlt_sd_trial_vi_tc": [6, 6],
            "pea_ravlt_ld_trial_vii_tc": [7, 7],
        }
    )

    cleaned = au.preprocess_ravlt_data(ravlt)

    assert cleaned["src_subject_id"].tolist() == ["s1"]
    assert cleaned["ravlt_immediate"].iloc[0] == 15
    assert cleaned["ravlt_short_delay"].iloc[0] == 6
    assert cleaned["ravlt_long_delay"].iloc[0] == 7


def test_calculate_income_to_needs_handles_large_households():
    df = pd.DataFrame({"demo_comb_income_v2": [8.0], "demo_roster_v2": [10.0]})

    result = au.calculate_income_to_needs(df)

    assert result["poverty_line_2017"].iloc[0] == 41320 + (2 * 4180)
    assert np.isclose(result["inr"].iloc[0], 87500 / (41320 + (2 * 4180)))
    assert not bool(result["inr_missing"].iloc[0])
    assert result.columns.tolist() == [
        "demo_comb_income_v2",
        "demo_roster_v2",
        "income_median",
        "poverty_line_2017",
        "inr",
        "inr_missing",
    ]


def test_calculate_income_to_needs_marks_missing_inputs():
    df = pd.DataFrame(
        {
            "demo_comb_income_v2": [8.0, np.nan, 8.0, 777],
            "demo_roster_v2": [4.0, 4.0, 999.0, 4.0],
        }
    )

    result = au.calculate_income_to_needs(df)

    observed_inr = 87500 / 24600
    assert np.isclose(result["inr"].iloc[0], observed_inr)
    assert result["inr"].iloc[1:].isna().all()
    assert result["inr_missing"].tolist() == [False, True, True, True]


def test_residualize_fc_profiles_removes_linear_mean_fd_signal():
    mean_fd = np.arange(8, dtype=float)
    noise = np.array([1, -1, 1, -1, -1, 1, -1, 1], dtype=float)
    df = pd.DataFrame(
        {
            "src_subject_id": [f"s{i}" for i in range(8)],
            "demo_sex_v2": [1, 2, 1, 2, 1, 2, 1, 2],
            "interview_age": [120, 120, 121, 121, 122, 122, 123, 123],
            "site_id_l": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "ehi1b": [1, 1, 2, 2, 1, 1, 2, 2],
            "mean_fd_0.20": mean_fd,
            "DMN_left_L_fz": 10 + (3 * mean_fd) + noise,
        }
    )

    result = au.residualize_fc_profiles(df, ["DMN_left_L_fz"])

    assert "DMN_left_L_fz_resid" in result.columns
    assert abs(result["DMN_left_L_fz_resid"].mean()) < 1e-10
    assert abs(np.corrcoef(result["DMN_left_L_fz_resid"], result["mean_fd_0.20"])[0, 1]) < 1e-10


def test_residualize_fc_profiles_leaves_missing_covariate_rows_nan():
    mean_fd = np.arange(8, dtype=float)
    df = pd.DataFrame(
        {
            "src_subject_id": [f"s{i}" for i in range(8)],
            "demo_sex_v2": [1, 2, 1, 2, 1, 2, 1, 2],
            "interview_age": [120, 120, 121, 121, 122, 122, 123, 123],
            "site_id_l": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "ehi1b": [1, 1, 2, 2, 1, 1, 2, 2],
            "mean_fd_0.20": mean_fd,
            "inr": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            "DMN_left_L_fz": 10 + (3 * mean_fd),
        }
    )

    result = au.residualize_fc_profiles(
        df,
        ["DMN_left_L_fz"],
        covariates=[*au.STANDARD_FC_COVARIATES, "inr"],
    )

    assert np.isnan(result.loc[2, "DMN_left_L_fz_resid"])
    assert result.loc[result.index != 2, "DMN_left_L_fz_resid"].notna().all()


def test_vertexwise_fc_columns_parse_python_index_and_hemisphere():
    df = pd.DataFrame(
        columns=[
            "subject_id",
            "DMN_left_12345_L_fz",
            "VAN_right_67890_R_fz",
            "DMN_full_22222_L_fz",
            "DMN_left_L_fz",
            "DMN_left_12345_L_fz_resid",
        ]
    )

    assert au.parse_vertexwise_fc_column("DMN_left_12345_L_fz") == (
        "DMN_left",
        12345,
        "L",
    )
    assert au.parse_vertexwise_fc_column("DMN_left_L_fz") is None
    assert au.get_vertexwise_fc_columns(df) == [
        "DMN_left_12345_L_fz",
        "VAN_right_67890_R_fz",
    ]


def test_region_fc_helpers_keep_only_raw_nonfull_fisher_z_columns():
    df = pd.DataFrame(
        {
            "src_subject_id": ["s1"],
            "DMN_left_L_fz": [0.1],
            "DMN_full_L_fz": [0.2],
            "DMN_left_L": [0.3],
            "DMN_left_L_fz_resid": [0.4],
        }
    )

    dropped, dropped_columns = au.drop_full_fc_columns(df)

    assert dropped_columns == ["DMN_full_L_fz"]
    assert "DMN_full_L_fz" not in dropped.columns
    assert au.get_region_fc_columns(dropped) == ["DMN_left_L_fz"]


def test_residualize_measure_replaces_existing_output_column_on_rerun():
    df = pd.DataFrame(
        {
            "src_subject_id": ["s1", "s2", "s3", "s4"],
            "measure": [1.0, 2.0, 3.0, 4.0],
            "age": [10.0, 11.0, 12.0, 13.0],
            "sex": [1, 2, 1, 2],
            "measure_resid": [99, 99, 99, 99],
        }
    )

    result = au.residualize_measure(df, "measure", ["age", "sex"])

    assert result.columns.tolist().count("measure_resid") == 1
    assert not (result["measure_resid"] == 99).all()


def test_compute_correlations_and_corrections_handle_constant_inputs():
    df = pd.DataFrame({"fc": [1.0, 1.0, 1.0], "measure": [2.0, 3.0, 4.0]})

    results = au.compute_correlations(df, ["measure"], ["fc"])
    corrected = au.apply_multiple_comparison_corrections(results)

    assert np.isnan(corrected["r"].iloc[0])
    assert np.isnan(corrected["p"].iloc[0])
    assert not bool(corrected["sig_bonf"].iloc[0])


def test_compute_correlations_can_require_nonmissing_inr():
    df = pd.DataFrame(
        {
            "fc": [1.0, 2.0, 3.0, 100.0],
            "measure": [1.0, 2.0, 3.0, -100.0],
            "inr": [1.0, 2.0, 3.0, np.nan],
        }
    )

    results = au.compute_correlations(
        df,
        ["measure"],
        ["fc"],
        required_nonmissing=["inr"],
    )

    assert results["n"].iloc[0] == 3
    assert np.isclose(results["r"].iloc[0], 1.0)


def test_fdr_correction_can_use_separate_cognitive_and_inr_families():
    results = pd.DataFrame(
        {
            "measure": ["cognitive_a", "cognitive_b", "inr"],
            "col": ["fc", "fc", "fc"],
            "r": [0.1, 0.1, 0.1],
            "p": [0.03, 0.04, 0.9],
            "n": [100, 100, 100],
            "fdr_family": ["cognitive", "cognitive", "inr"],
        }
    )

    corrected = au.apply_multiple_comparison_corrections(
        results,
        fdr_group_col="fdr_family",
    )

    cognitive = corrected[corrected["fdr_family"] == "cognitive"]
    inr = corrected[corrected["fdr_family"] == "inr"]
    assert np.allclose(cognitive["p_fdr"], [0.04, 0.04])
    assert cognitive["sig_fdr"].all()
    assert np.isclose(inr["p_fdr"].iloc[0], 0.9)
    assert not bool(inr["sig_fdr"].iloc[0])


def test_pairwise_rand_scores_returns_unique_pair_scores():
    df = pd.DataFrame(
        {
            "a": [0, 0, 1, 1],
            "b": [0, 0, 1, 1],
            "c": [1, 1, 0, 0],
        }
    )

    scores = au.pairwise_rand_scores(df, ["a", "b", "c"])

    assert len(scores) == 3
    assert scores == [1.0, 1.0, 1.0]


def test_mean_pairwise_rand_score_matches_explicit_scores():
    df = pd.DataFrame(
        {
            "a": [0, 0, 1, 1],
            "b": [0, 0, 1, 1],
            "c": [0, 1, 0, 1],
        }
    )

    scores = au.pairwise_rand_scores(df, ["a", "b", "c"])
    mean_score = au.mean_pairwise_rand_score(df, ["a", "b", "c"])

    assert np.isclose(mean_score, np.mean(scores))


def test_collapse_bootstrap_assignments_uses_modal_label_and_sorted_indices():
    subject_indices, labels = au.collapse_bootstrap_assignments(
        [3, 1, 3, 2, 1, 1],
        [8, 4, 8, 6, 5, 4],
    )

    assert subject_indices.tolist() == [1, 2, 3]
    assert labels.tolist() == [4, 6, 8]


def test_cluster_jaccard_scores_matches_labels_without_requiring_same_keys():
    scores = au.cluster_jaccard_scores(
        [0, 0, 1, 1, 2, 2],
        [9, 9, 7, 7, 7, 8],
    )

    assert np.isclose(scores[0], 1.0)
    assert np.isclose(scores[1], 2 / 3)
    assert np.isclose(scores[2], 0.5)


def test_clustering_uses_exported_fc_pca_and_jaccard_bootstrap():
    pca_text = (PROJECT_ROOT / "PCA_FC.ipynb").read_text()
    assert "pca_fc_scores_" in pca_text
    assert "pca_fc_variance_" in pca_text

    for notebook_name in [
        "clustering_without_inr_residualization.ipynb",
        "clustering_with_inr_residualization.ipynb",
    ]:
        notebook = json.loads((PROJECT_ROOT / notebook_name).read_text())
        pca_cell = next(
            cell for cell in notebook["cells"]
            if cell.get("id") == "load-existing-fc-pca"
        )
        stability_cell = next(
            cell for cell in notebook["cells"]
            if cell.get("id") == "bootstrap-cluster-stability"
        )
        pca_source = "".join(pca_cell["source"])
        stability_source = "".join(stability_cell["source"])

        assert "pca_fc_scores_" in pca_source
        assert "PCA(" not in pca_source
        assert "N_CLUSTER_BOOTSTRAPS = 5000" in stability_source
        assert "bootstrap_rng.integers" in stability_source
        assert "cluster_jaccard_scores" in stability_source
        assert "mean_pairwise_rand_score" in stability_source


def test_modified_notebooks_have_valid_python_code_cells():
    notebook_paths = [
        PROJECT_ROOT / "data_wrangling.ipynb",
        PROJECT_ROOT / "brain_behavior_without_inr_residualization.ipynb",
        PROJECT_ROOT / "brain_behavior_with_inr_residualization.ipynb",
        PROJECT_ROOT / "vertexwise_brain_behavior_without_ses_residualization.ipynb",
        PROJECT_ROOT / "vertexwise_brain_behavior_with_ses_residualization.ipynb",
        PROJECT_ROOT / "clustering.ipynb",
        PROJECT_ROOT / "clustering_without_inr_residualization.ipynb",
        PROJECT_ROOT / "clustering_with_inr_residualization.ipynb",
        PROJECT_ROOT / "compare_ses_residualization_brain_behavior.ipynb",
        PROJECT_ROOT / "PCA_tasks.ipynb",
        PROJECT_ROOT / "PCA_FC.ipynb",
    ]

    for notebook_path in notebook_paths:
        notebook = json.loads(notebook_path.read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                ast.parse(source, filename=f"{notebook_path.name}:cell-{index}")


def test_data_wrangling_uses_separate_meanfc_source_and_wrangled_output_paths():
    notebook_text = (PROJECT_ROOT / "data_wrangling.ipynb").read_text()

    assert "MEANFC_SOURCE_PATH = OUTPUT_DIR / 'pMTG_FC_profiles_midb61_meanFC.csv'" in notebook_text
    assert "OUTPUT_PATH = OUTPUT_DIR / 'wrangled_pMTG_FC_data_midb61_meanFC.csv'" in notebook_text
    assert "meanfc_source = load_data(MEANFC_SOURCE_PATH)" in notebook_text
    assert "FC_PATH =" not in notebook_text


def test_data_wrangling_filters_qa_and_families_before_fc_residualization():
    notebook_text = (PROJECT_ROOT / "data_wrangling.ipynb").read_text()

    assert "FAMILY_SELECTION_SEED = 42" in notebook_text
    assert "def select_one_per_family(" in notebook_text
    assert "merged_data = merged_data[merged_data['imgincl_rsfmri_include'].eq(1)].copy()" in notebook_text
    assert "merged_data = select_one_per_family(" in notebook_text
    assert "Families with more than one retained participant" in notebook_text
    assert (
        notebook_text.index("merged_data = merged_data[merged_data['imgincl_rsfmri_include'].eq(1)].copy()")
        < notebook_text.index("merged_data = select_one_per_family(")
        < notebook_text.index("merged_data = residualize_fc_profiles(")
    )


def test_active_notebooks_use_current_wrangled_meanfc_filename():
    notebook_paths = [
        PROJECT_ROOT / "PCA_tasks.ipynb",
        PROJECT_ROOT / "PCA_FC.ipynb",
        PROJECT_ROOT / "clustering.ipynb",
        PROJECT_ROOT / "clustering_without_inr_residualization.ipynb",
        PROJECT_ROOT / "clustering_with_inr_residualization.ipynb",
        PROJECT_ROOT / "brain_behavior_without_inr_residualization.ipynb",
        PROJECT_ROOT / "brain_behavior_with_inr_residualization.ipynb",
        PROJECT_ROOT / "vertexwise_brain_behavior_without_ses_residualization.ipynb",
        PROJECT_ROOT / "vertexwise_brain_behavior_with_ses_residualization.ipynb",
        PROJECT_ROOT / "compare_ses_residualization_brain_behavior.ipynb",
    ]

    for notebook_path in notebook_paths:
        notebook_text = notebook_path.read_text()
        assert "wrangled_pMTG_FC_data_midb61_meanFC.csv" in notebook_text
        assert "abcd_pMTG_FC_data_midb61_meanFC.csv" not in notebook_text


def test_brain_behavior_variant_flags_are_explicit():
    without_inr = (PROJECT_ROOT / "brain_behavior_without_inr_residualization.ipynb").read_text()
    with_inr = (PROJECT_ROOT / "brain_behavior_with_inr_residualization.ipynb").read_text()

    assert "RESIDUALIZE_FOR_INR = False" in without_inr
    assert "RESIDUALIZE_FOR_INR = True" in with_inr
    assert "INR_RESIDUALIZATION_COVARIATE = 'inr'" in without_inr
    assert "INR_RESIDUALIZATION_COVARIATE = 'inr'" in with_inr
    assert "required_nonmissing=[INR_VARIABLE] if RESIDUALIZE_FOR_INR else None" in without_inr
    assert "required_nonmissing=[INR_VARIABLE] if RESIDUALIZE_FOR_INR else None" in with_inr


def test_brain_behavior_notebooks_use_matched_efa_scores():
    notebook_paths = [
        PROJECT_ROOT / "brain_behavior_without_inr_residualization.ipynb",
        PROJECT_ROOT / "brain_behavior_with_inr_residualization.ipynb",
    ]

    for notebook_path in notebook_paths:
        notebook_text = notebook_path.read_text()
        assert "from pathlib import Path" in notebook_text
        assert (
            "EFA_SCORE_MODEL_NAME = 'with_ses_residualization' if RESIDUALIZE_FOR_INR else 'without_ses_residualization'"
            in notebook_text
        )
        assert (
            "EFA_SCORE_PREFIX = 'ses_cognitive' if RESIDUALIZE_FOR_INR else 'no_ses_cognitive'"
            in notebook_text
        )
        assert (
            "EFA_SCORE_PATH = PCA_RESULTS_DIR / f'efa_cognitive_scores_{EFA_SCORE_MODEL_NAME}.csv'"
            in notebook_text
        )
        assert "cognitive_efa_measures = [" in notebook_text
        assert "analysis_measures = cognitive_efa_measures.copy()" in notebook_text
        assert "cognitive_efa_results_df = cognitive_results_df.copy()" in notebook_text
        assert "apply_multiple_comparison_corrections(results_df, alpha=0.05)" in notebook_text
        assert "sig_fdr" in notebook_text
        assert "p_fdr" in notebook_text
        assert "software_test_results_df['p_bonf']" not in notebook_text
        assert "software_test_results_df['sig_bonf']" not in notebook_text
        assert "software_test_results_df['p_fdr']" not in notebook_text
        assert "software_test_results_df['sig_fdr']" not in notebook_text
        assert "p_bonf" in notebook_text
        assert "sig_bonf" in notebook_text
        assert "measure_map = {" in notebook_text
        assert "EFA Factor" in notebook_text


def test_brain_behavior_notebooks_exclude_removed_behavioral_variables():
    stop_signal_and_cbcl_paths = [
        PROJECT_ROOT / "analysis_utils.py",
        PROJECT_ROOT / "brain_behavior_without_inr_residualization.ipynb",
        PROJECT_ROOT / "brain_behavior_with_inr_residualization.ipynb",
        PROJECT_ROOT / "compare_ses_residualization_brain_behavior.ipynb",
        PROJECT_ROOT / "vertexwise_brain_behavior_without_ses_residualization.ipynb",
        PROJECT_ROOT / "vertexwise_brain_behavior_with_ses_residualization.ipynb",
    ]
    removed_terms = [
        "tfmri_sst_all_beh_total_issrt",
        "SST ISSRT",
        "issrt",
        "mh_p_cbcl",
        "CBCL",
        "cbcl",
    ]

    for path in stop_signal_and_cbcl_paths:
        text = path.read_text()
        for term in removed_terms:
            assert term not in text

    brain_behavior_paths = [
        PROJECT_ROOT / "brain_behavior_without_inr_residualization.ipynb",
        PROJECT_ROOT / "brain_behavior_with_inr_residualization.ipynb",
        PROJECT_ROOT / "vertexwise_brain_behavior_without_ses_residualization.ipynb",
        PROJECT_ROOT / "vertexwise_brain_behavior_with_ses_residualization.ipynb",
        PROJECT_ROOT / "PCA_FC.ipynb",
    ]
    removed_pc_terms = [
        "cognitive_pc",
        "cognitive_PC",
        "Cognitive Task PC",
        "cognitive-PC",
    ]

    for path in brain_behavior_paths:
        text = path.read_text()
        for term in removed_pc_terms:
            assert term not in text

    raw_cognitive_task_paths = [
        PROJECT_ROOT / "compare_ses_residualization_brain_behavior.ipynb",
        PROJECT_ROOT / "vertexwise_brain_behavior_without_ses_residualization.ipynb",
        PROJECT_ROOT / "vertexwise_brain_behavior_with_ses_residualization.ipynb",
        PROJECT_ROOT / "PCA_FC.ipynb",
    ]
    raw_cognitive_task_terms = [
        "COGNITIVE_MEASURES_RAW",
        "ravlt_immediate_2y",
        "ravlt_short_delay_2y",
        "ravlt_long_delay_2y",
        "nihtbx_picvocab_uncorrected_2y",
        "nihtbx_reading_uncorrected_2y",
        "nihtbx_flanker_uncorrected_2y",
        "nihtbx_picture_uncorrected_2y",
    ]

    for path in raw_cognitive_task_paths:
        text = path.read_text()
        for term in raw_cognitive_task_terms:
            assert term not in text


def test_vertexwise_brain_behavior_ses_flags_are_explicit():
    without_ses = (
        PROJECT_ROOT / "vertexwise_brain_behavior_without_ses_residualization.ipynb"
    ).read_text()
    with_ses = (
        PROJECT_ROOT / "vertexwise_brain_behavior_with_ses_residualization.ipynb"
    ).read_text()

    assert "RESIDUALIZE_FOR_SES = False" in without_ses
    assert "RESIDUALIZE_FOR_SES = True" in with_ses
    assert "SES_VARIABLE = 'inr'" in without_ses
    assert "SES_VARIABLE = 'inr'" in with_ses
    assert "SES_RESIDUALIZATION_COVARIATE = 'inr'" in without_ses
    assert "SES_RESIDUALIZATION_COVARIATE = 'inr'" in with_ses
    assert "required_nonmissing=[SES_VARIABLE] if RESIDUALIZE_FOR_SES else None" in without_ses
    assert "required_nonmissing=[SES_VARIABLE] if RESIDUALIZE_FOR_SES else None" in with_ses


def test_vertexwise_brain_behavior_notebooks_use_efa_fdr_only():
    notebook_paths = [
        PROJECT_ROOT / "vertexwise_brain_behavior_without_ses_residualization.ipynb",
        PROJECT_ROOT / "vertexwise_brain_behavior_with_ses_residualization.ipynb",
    ]

    for notebook_path in notebook_paths:
        notebook_text = notebook_path.read_text()
        assert "EFA_SCORE_MODEL_NAME = ANALYSIS_SLUG" in notebook_text
        assert "EFA_SCORE_PREFIX = 'ses_cognitive' if RESIDUALIZE_FOR_SES else 'no_ses_cognitive'" in notebook_text
        assert "EFA_SCORE_PATH = PCA_RESULTS_DIR / f'efa_cognitive_scores_{EFA_SCORE_MODEL_NAME}.csv'" in notebook_text
        assert "cognitive_efa_measures = [" in notebook_text
        assert "analysis_measures = cognitive_efa_measures.copy()" in notebook_text
        assert "apply_multiple_comparison_corrections(results_df, alpha=0.05)" in notebook_text
        assert "sig_fdr" in notebook_text
        assert "p_fdr" in notebook_text
        assert "cognitive_efa_fdr_significant.csv" in notebook_text
        assert "columns=['p_bonf', 'sig_bonf']" in notebook_text
        assert "Bonferroni" not in notebook_text


def test_compare_ses_brain_behavior_uses_efa_fdr_outputs():
    notebook_text = (
        PROJECT_ROOT / "compare_ses_residualization_brain_behavior.ipynb"
    ).read_text()

    assert "def load_matched_efa_scores" in notebook_text
    assert "measure_family" in notebook_text
    assert "cognitive_efa" in notebook_text
    assert "p_fdr" in notebook_text
    assert "sig_fdr" in notebook_text
    assert "apply_multiple_comparison_corrections" in notebook_text
    assert "fdrcorrection" not in notebook_text
    assert "residualize_measures" not in notebook_text
    assert "columns=['p_bonf', 'sig_bonf']" in notebook_text
    assert "Bonferroni" not in notebook_text


def test_clustering_variant_flags_are_explicit():
    without_inr = (PROJECT_ROOT / "clustering_without_inr_residualization.ipynb").read_text()
    with_inr = (PROJECT_ROOT / "clustering_with_inr_residualization.ipynb").read_text()

    assert "RESIDUALIZE_FC_FOR_INR = False" in without_inr
    assert "RESIDUALIZE_FC_FOR_INR = True" in with_inr
    assert "INR_RESIDUALIZATION_COVARIATE = 'inr'" in without_inr
    assert "INR_RESIDUALIZATION_COVARIATE = 'inr'" in with_inr
    assert "cluster_covariates = cluster_covariates + [INR_RESIDUALIZATION_COVARIATE]" in with_inr


def test_pca_notebooks_include_residual_normality_diagnostics():
    tasks_text = (PROJECT_ROOT / "PCA_tasks.ipynb").read_text()
    fc_text = (PROJECT_ROOT / "PCA_FC.ipynb").read_text()

    for notebook_text in [tasks_text, fc_text]:
        assert "from scipy import stats" in notebook_text
        assert "def plot_residual_normality_diagnostics" in notebook_text
        assert "histogram" in notebook_text
    assert "Q-Q plot" in notebook_text

    assert "cognitive_residual_normality = plot_residual_normality_diagnostics" in tasks_text
    assert "'cognitive_residual_normality': cognitive_residual_normality" in tasks_text
    assert "fc_residual_normality = plot_residual_normality_diagnostics" in fc_text
    assert "'fc_residual_normality': fc_residual_normality" in fc_text


def test_pca_fc_uses_revised_cognitive_efa_factor_scores():
    fc_text = (PROJECT_ROOT / "PCA_FC.ipynb").read_text()

    assert "efa_cognitive_scores" in fc_text
    assert "cognitive_efa_factor" in fc_text
    assert "pca_cognitive_scores" not in fc_text
    assert "cognitive_pc" not in fc_text


def test_pca_fc_drops_incomplete_rows_before_fc_pca_fit():
    fc_text = (PROJECT_ROOT / "PCA_FC.ipynb").read_text()

    assert "fc_complete = fc_input.dropna().copy()" in fc_text
    assert "scores = pd.DataFrame(np.nan, index=fc_input.index" in fc_text
    assert "scores.loc[fc_complete.index, score_cols] = scores_array" in fc_text
    assert "FC PCA input contains missing values" not in fc_text


def test_pca_tasks_includes_exploratory_factor_analysis():
    tasks_text = (PROJECT_ROOT / "PCA_tasks.ipynb").read_text()

    assert "from factor_analyzer import FactorAnalyzer" in tasks_text
    assert "calculate_bartlett_sphericity" in tasks_text
    assert "calculate_kmo" in tasks_text
    assert "EFA_N_FACTORS = None" in tasks_text
    assert "def fit_efa_scores" in tasks_text
    assert "def plot_cognitive_efa_summary" in tasks_text
    assert "cognitive_efa_result = fit_efa_scores" in tasks_text
    assert "'cognitive_efa': cognitive_efa_result" in tasks_text
    assert "efa_cognitive_scores" in tasks_text
    assert "efa_cognitive_loadings" in tasks_text
    assert "efa_cognitive_diagnostics" in tasks_text

    assert "ConfirmatoryFactorAnalyzer" not in tasks_text
    assert "ModelSpecificationParser" not in tasks_text


def test_site_and_handedness_covariate_updates_are_reflected_in_notebooks():
    notebook_paths = [
        PROJECT_ROOT / "data_wrangling.ipynb",
        PROJECT_ROOT / "clustering_without_inr_residualization.ipynb",
        PROJECT_ROOT / "clustering_with_inr_residualization.ipynb",
        PROJECT_ROOT / "PCA_FC.ipynb",
        PROJECT_ROOT / "brain_behavior_without_inr_residualization.ipynb",
        PROJECT_ROOT / "brain_behavior_with_inr_residualization.ipynb",
        PROJECT_ROOT / "vertexwise_brain_behavior_without_ses_residualization.ipynb",
        PROJECT_ROOT / "vertexwise_brain_behavior_with_ses_residualization.ipynb",
        PROJECT_ROOT / "compare_ses_residualization_brain_behavior.ipynb",
    ]

    for notebook_path in notebook_paths:
        notebook_text = notebook_path.read_text()
        assert "'site_id_l'" in notebook_text
        assert "'ehi1b'" in notebook_text

    tasks_text = (PROJECT_ROOT / "PCA_tasks.ipynb").read_text()
    assert (
        "COGNITIVE_TASK_COVARIATES = ['interview_age', 'demo_sex_v2', 'site_id_l', 'ehi1b']"
        in tasks_text
    )

    for notebook_path in [
        PROJECT_ROOT / "brain_behavior_without_inr_residualization.ipynb",
        PROJECT_ROOT / "brain_behavior_with_inr_residualization.ipynb",
    ]:
        notebook_text = notebook_path.read_text()
        assert (
            "COGNITIVE_TASK_RESIDUAL_COVARIATES = ['interview_age', 'demo_sex_v2', 'site_id_l', 'ehi1b']"
            in notebook_text
        )
        assert "CLUSTER_GROUP_COL = 'kmeans_2_consensus'" in notebook_text
        assert "df_subset[CLUSTER_GROUP_COL]" in notebook_text
