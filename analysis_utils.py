"""Shared utilities for the pMTG functional-connectivity analyses.

The notebooks in this project do the high-level analysis work.  This module
keeps reusable data-cleaning, residualization, and statistics helpers in a
plain Python file so they can be tested with pytest.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import fdrcorrection


# ABCD missing/invalid response codes used throughout the behavioral files.
INVALID_CODES = (555, 777, 888, 999)

# Standard FC covariates used by the data-wrangling notebook.  mean_fd_0.20 is
# loaded from the motion QA file and controls for framewise displacement.
STANDARD_FC_COVARIATES = [
    "demo_sex_v2",
    "interview_age",
    "site_id_l",
    "ehi1b",
    "mean_fd_0.20",
]

# Covariates treated as categorical before one-hot encoding.
DEFAULT_CATEGORICAL_COVARIATES = {
    "demo_sex_v2",
    "site_id_l",
    "ehi1b",
}

# Midpoints for the ABCD household-income categories used to compute INR.
INCOME_MEDIANS = {
    1.0: 2500,
    2.0: 8500,
    3.0: 14000,
    4.0: 20500,
    5.0: 30000,
    6.0: 42500,
    7.0: 62500,
    8.0: 87500,
    9.0: 150000,
    10.0: 250000,
}

# 2017 Federal Poverty Guidelines for household sizes 1-8.
POVERTY_LINES_2017 = {
    1.0: 12060,
    2.0: 16240,
    3.0: 20420,
    4.0: 24600,
    5.0: 28780,
    6.0: 32960,
    7.0: 37140,
    8.0: 41320,
}


def save_data(df, output_path, **kwargs):
    """Save a DataFrame to disk without changing the notebook call signature."""
    df.to_csv(output_path, **kwargs)


def load_data(data_path, columns=None, test=False, **kwargs):
    """Load a CSV file and optionally restrict it to selected columns.

    The ``test`` argument is retained for compatibility with the original
    notebook helper, even though the current workflow does not use it.
    """
    _ = test
    return pd.read_csv(data_path, usecols=columns, **kwargs)


def standardize_subject_id(subject_ids):
    """Match subject IDs across ABCD and FC files by removing known prefixes."""
    return (
        subject_ids.astype(str)
        .str.replace("_", "", regex=False)
        .str.replace("^sub-", "", regex=True)
    )


def _validate_columns(df, columns, context):
    """Raise a helpful error when required columns are absent."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required {context} column(s): {missing}")


def preprocess_ravlt_data(ravlt_data):
    """Calculate RAVLT summary scores and keep complete immediate-recall rows."""
    trial_columns = [
        "pea_ravlt_sd_trial_i_tc",
        "pea_ravlt_sd_trial_ii_tc",
        "pea_ravlt_sd_trial_iii_tc",
        "pea_ravlt_sd_trial_iv_tc",
        "pea_ravlt_sd_trial_v_tc",
    ]
    required_columns = [
        "src_subject_id",
        *trial_columns,
        "pea_ravlt_sd_trial_vi_tc",
        "pea_ravlt_ld_trial_vii_tc",
    ]
    _validate_columns(ravlt_data, required_columns, "RAVLT")

    # Work on a copy so callers do not get surprise in-place edits.
    ravlt_data = ravlt_data.copy()
    ravlt_data[trial_columns] = ravlt_data[trial_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Immediate recall is only interpretable when all five trials are present.
    ravlt_data_clean = ravlt_data.dropna(subset=trial_columns).copy()
    ravlt_data_clean["ravlt_immediate"] = ravlt_data_clean[trial_columns].sum(axis=1)

    # Standardize delayed-recall variable names for the downstream notebooks.
    ravlt_data_clean.rename(
        columns={
            "pea_ravlt_sd_trial_vi_tc": "ravlt_short_delay",
            "pea_ravlt_ld_trial_vii_tc": "ravlt_long_delay",
            "pea_ravlt_sd_listb_tc": "ravlt_listb",
        },
        inplace=True,
    )
    ravlt_data_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

    return ravlt_data_clean[
        ["src_subject_id", "ravlt_immediate", "ravlt_short_delay", "ravlt_long_delay"]
    ]


def load_motion_qa(motion_qa_path, motion_column="mean_fd_0.20"):
    """Load the motion QA file and return subject ID plus mean FD."""
    motion_qa = load_data(motion_qa_path)
    _validate_columns(motion_qa, ["src_subject_id", motion_column], "motion QA")

    motion_qa = motion_qa[["src_subject_id", motion_column]].copy()
    motion_qa["src_subject_id"] = standardize_subject_id(motion_qa["src_subject_id"])
    motion_qa = motion_qa.drop_duplicates(subset=["src_subject_id"])
    motion_qa[motion_column] = pd.to_numeric(motion_qa[motion_column], errors="coerce")
    return motion_qa


def merge_motion_qa(df, motion_qa_path, motion_column="mean_fd_0.20", how="left"):
    """Merge motion QA values into a DataFrame without dropping subjects by default."""
    _validate_columns(df, ["src_subject_id"], "analysis")
    motion_qa = load_motion_qa(motion_qa_path, motion_column=motion_column)

    merged = df.copy()
    merged["src_subject_id"] = standardize_subject_id(merged["src_subject_id"])

    # If an older copy of the motion column is present, fill its missing values
    # from the QA file instead of creating hard-to-track suffixes.
    if motion_column in merged.columns:
        existing_motion = merged.groupby("src_subject_id")[motion_column].first()
        merged = merged.drop(columns=[motion_column])
        merged = merged.merge(motion_qa, on="src_subject_id", how=how)
        merged[motion_column] = merged[motion_column].fillna(
            merged["src_subject_id"].map(existing_motion)
        )
    else:
        merged = merged.merge(motion_qa, on="src_subject_id", how=how)

    return merged


def merge_abcd_data(conn_path, abcd_path, output_path=None, motion_qa_path=None):
    """Merge ABCD demographic, behavioral, QA, and FC data for analysis.

    ``output_path`` is retained for compatibility with the original notebook.
    Saving is handled explicitly by the notebook so intermediate review remains
    visible.
    """

    def rename_columns_with_suffix(df, suffix, ignore_cols=None):
        """Add a suffix to non-ID columns after timepoint filtering."""
        ignore_cols = ["src_subject_id"] if ignore_cols is None else list(ignore_cols)
        return df.rename(
            columns=lambda column: column + suffix
            if column not in ignore_cols
            else column
        )

    _ = output_path
    abcd_path = Path(abcd_path)

    # Load 2-year MRI administration metadata.
    abcd_img = load_data(
        abcd_path / "imaging/mri_y_adm_info.csv",
        columns=[
            "src_subject_id",
            "eventname",
            "mri_info_softwareversion",
            "mri_info_studydate",
        ],
    )
    abcd_img = abcd_img[abcd_img["eventname"] == "2_year_follow_up_y_arm_1"]
    abcd_img.drop(columns=["eventname"], inplace=True)

    # Load baseline family IDs so one child per family can be selected later.
    fam_data = load_data(
        abcd_path / "abcd-general/abcd_y_lt.csv",
        columns=["src_subject_id", "eventname", "rel_family_id"],
    )
    fam_data = fam_data[fam_data["eventname"] == "baseline_year_1_arm_1"]
    fam_data.drop(columns=["eventname"], inplace=True)

    # Load 2-year site and age variables used in residualization.
    site_age_data = load_data(
        abcd_path / "abcd-general/abcd_y_lt.csv",
        columns=["src_subject_id", "eventname", "site_id_l", "interview_age"],
    )
    site_age_data = site_age_data[
        site_age_data["eventname"] == "2_year_follow_up_y_arm_1"
    ]
    site_age_data.drop(columns=["eventname"], inplace=True)

    # Load handedness variables collected at baseline.
    hand_data = load_data(
        abcd_path / "neurocognition/nc_y_ehis.csv",
        columns=[
            "src_subject_id",
            "eventname",
            "ehi1b",
            "ehi2b",
            "ehi3b",
            "ehi4b",
            "ehi_y_ss_scoreb",
        ],
    )
    hand_data = hand_data[hand_data["eventname"] == "baseline_year_1_arm_1"]
    hand_data.drop(columns=["eventname"], inplace=True)

    # Load sex assigned at birth for demographic adjustment.
    sex_data = load_data(
        abcd_path / "gender-identity-sexual-health/gish_p_gi.csv",
        columns=["src_subject_id", "eventname", "demo_sex_v2"],
    )
    sex_data = sex_data[sex_data["eventname"] == "baseline_year_1_arm_1"]
    sex_data.drop(columns=["eventname"], inplace=True)

    # Load 2-year NIH Toolbox scores and suffix their variable names.
    nihtb_data_2y = load_data(
        abcd_path / "neurocognition/nc_y_nihtb.csv",
        columns=[
            "src_subject_id",
            "eventname",
            "nihtbx_picvocab_v",
            "nihtbx_picvocab_fc",
            "nihtbx_picvocab_agecorrected",
            "nihtbx_picvocab_uncorrected",
            "nihtbx_reading_v",
            "nihtbx_reading_fc",
            "nihtbx_reading_agecorrected",
            "nihtbx_reading_uncorrected",
            "nihtbx_flanker_v",
            "nihtbx_flanker_fc",
            "nihtbx_flanker_agecorrected",
            "nihtbx_flanker_uncorrected",
            "nihtbx_pattern_v",
            "nihtbx_pattern_fc",
            "nihtbx_pattern_agecorrected",
            "nihtbx_pattern_uncorrected",
            "nihtbx_picture_v",
            "nihtbx_picture_fc",
            "nihtbx_picture_agecorrected",
            "nihtbx_picture_uncorrected",
        ],
    )
    nihtb_data_2y = nihtb_data_2y[
        nihtb_data_2y["eventname"] == "2_year_follow_up_y_arm_1"
    ]
    nihtb_data_2y.drop(columns=["eventname"], inplace=True)
    nihtb_data_2y = rename_columns_with_suffix(nihtb_data_2y, "_2y")

    # Load 2-year RAVLT data and calculate immediate/short/long delay scores.
    ravlt_data = load_data(
        abcd_path / "neurocognition/nc_y_ravlt.csv",
        columns=[
            "src_subject_id",
            "eventname",
            "pea_ravlt_sd_trial_i_tc",
            "pea_ravlt_sd_trial_ii_tc",
            "pea_ravlt_sd_trial_iii_tc",
            "pea_ravlt_sd_trial_iv_tc",
            "pea_ravlt_sd_trial_v_tc",
            "pea_ravlt_sd_listb_tc",
            "pea_ravlt_sd_trial_vi_tc",
            "pea_ravlt_ld_trial_vii_tc",
        ],
    )
    ravlt_data = ravlt_data[ravlt_data["eventname"] == "2_year_follow_up_y_arm_1"]
    ravlt_data.drop(columns=["eventname"], inplace=True)
    ravlt_summary = rename_columns_with_suffix(preprocess_ravlt_data(ravlt_data), "_2y")

    # Load baseline socioeconomic variables.
    ses_data = load_data(
        abcd_path / "abcd-general/abcd_p_demo.csv",
        columns=[
            "src_subject_id",
            "eventname",
            "demo_comb_income_v2",
            "demo_roster_v2",
            "demo_prnt_ed_v2_2yr_l",
            "demo_prtnr_ed_v2_2yr_l",
            "demo_prnt_marital_v2",
            "demo_prnt_prtnr_bio",
        ],
    )
    ses_data = ses_data[ses_data["eventname"] == "baseline_year_1_arm_1"]
    ses_data.drop(columns=["eventname"], inplace=True)

    # Load baseline language and acculturation variables.
    bilingual_data = load_data(
        abcd_path / "culture-environment/ce_y_acc.csv",
        columns=["src_subject_id", "eventname", "accult_q2_y"],
    )
    bilingual_data = bilingual_data[
        bilingual_data["eventname"] == "baseline_year_1_arm_1"
    ]
    bilingual_data.drop(columns=["eventname"], inplace=True)

    dual_lang = load_data(
        abcd_path / "abcd-general/abcd_p_demo.csv",
        columns=["src_subject_id", "eventname", "demo_dual_lang_v2_l"],
    )
    dual_lang = dual_lang[dual_lang["eventname"] == "1_year_follow_up_y_arm_1"]
    dual_lang.drop(columns=["eventname"], inplace=True)

    # Load ABCD scanner-motion summary variables for descriptive checks.
    motion_df = pd.read_csv(abcd_path / "imaging/mri_y_qc_motion.csv")
    motion_df = motion_df[motion_df["eventname"] == "2_year_follow_up_y_arm_1"]
    motion_df = motion_df[
        [
            "src_subject_id",
            "rsfmri_meanmotion",
            "rsfmri_maxmotion",
            "rsfmri_ntpoints",
            "rsfmri_nvols",
            "rsfmri_numtrs",
        ]
    ]

    # Load puberty variables and combine female/male ABCD Tanner summaries.
    pds_full = pd.read_csv(abcd_path / "physical-health/ph_y_pds.csv")
    baseline_sex = pds_full[pds_full["eventname"] == "baseline_year_1_arm_1"][
        ["src_subject_id", "pds_sex_y"]
    ]
    tanner_df = pds_full[pds_full["eventname"] == "2_year_follow_up_y_arm_1"][
        [
            "src_subject_id",
            "pds_bdyhair_y",
            "pds_f4_2_y",
            "pds_f5_y",
            "pds_m4_y",
            "pds_m5_y",
            "pds_y_ss_female_category_2",
            "pds_y_ss_male_cat_2",
        ]
    ]
    tanner_df = tanner_df.merge(baseline_sex, on="src_subject_id", how="left")

    def get_tanner_stage(row):
        """Select the ABCD Tanner score that matches baseline sex coding."""
        if pd.notna(row["pds_y_ss_female_category_2"]) and row["pds_sex_y"] == 2:
            return row["pds_y_ss_female_category_2"]
        if pd.notna(row["pds_y_ss_male_cat_2"]) and row["pds_sex_y"] == 1:
            return row["pds_y_ss_male_cat_2"]
        return None

    tanner_df["tanner_stage"] = tanner_df.apply(get_tanner_stage, axis=1)

    # Load ABCD rsfMRI inclusion flags.
    qa_df = pd.read_csv(abcd_path / "imaging/mri_y_qc_incl.csv")
    qa_df = qa_df[qa_df["eventname"] == "2_year_follow_up_y_arm_1"]
    qa_df = qa_df[["src_subject_id", "imgincl_rsfmri_include"]]

    # Load race and ethnicity variables for demographic summaries.
    demo_df = pd.read_csv(abcd_path / "abcd-general/abcd_p_demo.csv")
    demo_df = demo_df[demo_df["eventname"] == "baseline_year_1_arm_1"]
    demo_df = demo_df[
        [
            "src_subject_id",
            "demo_race_a_p___10",
            "demo_race_a_p___11",
            "demo_race_a_p___12",
            "demo_race_a_p___13",
            "demo_race_a_p___14",
            "demo_race_a_p___15",
            "demo_race_a_p___16",
            "demo_race_a_p___17",
            "demo_race_a_p___18",
            "demo_race_a_p___19",
            "demo_race_a_p___20",
            "demo_race_a_p___21",
            "demo_race_a_p___22",
            "demo_race_a_p___23",
            "demo_race_a_p___24",
            "demo_race_a_p___25",
            "demo_race_a_p___77",
            "demo_race_a_p___99",
            "demo_ethn_v2",
            "demo_ethn2_v2",
        ]
    ]

    # Merge all non-FC subject-level files before adding connectivity.
    combined_df = (
        abcd_img.merge(fam_data, on="src_subject_id")
        .merge(site_age_data, on="src_subject_id")
        .merge(hand_data, on="src_subject_id")
        .merge(sex_data, on="src_subject_id")
        .merge(nihtb_data_2y, on="src_subject_id", how="left")
        .merge(ravlt_summary, on="src_subject_id", how="left")
        .merge(ses_data, on="src_subject_id", how="left")
        .merge(bilingual_data, on="src_subject_id", how="left")
        .merge(motion_df, on="src_subject_id", how="left")
        .merge(tanner_df, on="src_subject_id", how="left")
        .merge(qa_df, on="src_subject_id", how="left")
        .merge(demo_df, on="src_subject_id", how="left")
        .merge(dual_lang, on="src_subject_id", how="left")
        .drop_duplicates()
    )

    pd.set_option("display.max_columns", None)
    print("Columns to be combined with connectivity data:", combined_df.columns)
    combined_df["src_subject_id"] = standardize_subject_id(
        combined_df["src_subject_id"]
    )

    # Load connectivity profiles and normalize their subject ID column.
    conn_df = load_data(conn_path)
    if "subject_id" in conn_df.columns:
        conn_df.rename(columns={"subject_id": "src_subject_id"}, inplace=True)
    _validate_columns(conn_df, ["src_subject_id"], "connectivity")
    conn_df["src_subject_id"] = standardize_subject_id(conn_df["src_subject_id"])
    conn_df = conn_df.dropna(subset=conn_df.columns[1:])
    print(f"Connectivity data shape after dropping NaNs: {conn_df.shape}")

    # Right-join keeps the FC subjects as the analysis universe.
    merged_data = pd.merge(combined_df, conn_df, on="src_subject_id", how="right")

    # Add the framewise-displacement value requested for FC residualization.
    if motion_qa_path is not None:
        merged_data = merge_motion_qa(merged_data, motion_qa_path, how="left")

    print(f"Merged data shape: {merged_data.shape}")
    print(merged_data.head())
    return merged_data


def select_one_per_family(df, family_col="rel_family_id", seed=42):
    """Randomly select one participant from each family to reduce relatedness."""
    _validate_columns(df, [family_col], "family")
    df = df.copy()
    df[family_col] = df[family_col].astype(str)
    selected_indices = df.groupby(family_col, group_keys=False).apply(
        lambda family_df: family_df.sample(1, random_state=seed).index[0]
    )
    return df.loc[selected_indices].reset_index(drop=True)


def get_fc_profile_columns(df, network_labels):
    """Find left, right, and combined left/right pMTG FC profile columns."""
    left_columns = []
    right_columns = []

    # Match the original naming convention while requiring ``_fz`` so unrelated
    # network-like columns are not accidentally residualized.
    for network in network_labels:
        for column in df.columns:
            if network in column and "_fz" in column and "_L_" in column:
                left_columns.append(column)
            if network in column and "_fz" in column and "_R_" in column:
                right_columns.append(column)

    left_right_columns = left_columns + right_columns
    return left_columns, right_columns, left_right_columns


def _complete_covariate_mask(
    df,
    covariates,
    categorical_covariates=DEFAULT_CATEGORICAL_COVARIATES,
):
    """Return rows with all covariates available for regression."""
    _validate_columns(df, covariates, "covariate")
    categorical_covariates = set(categorical_covariates)
    complete = pd.Series(True, index=df.index)

    for covariate in covariates:
        is_categorical = (
            df[covariate].dtype == "object" or covariate in categorical_covariates
        )
        if is_categorical:
            complete &= df[covariate].notna()
        else:
            complete &= pd.to_numeric(df[covariate], errors="coerce").notna()

    return complete


def _encode_regression_covariates(
    df,
    covariates,
    categorical_covariates=DEFAULT_CATEGORICAL_COVARIATES,
):
    """Encode covariates while preserving numeric variables as numeric terms."""
    encoded_parts = []
    categorical_covariates = set(categorical_covariates)

    for covariate in covariates:
        if df[covariate].dtype == "object" or covariate in categorical_covariates:
            encoded_parts.append(
                pd.get_dummies(
                    df[covariate],
                    prefix=covariate,
                    drop_first=True,
                    dtype=float,
                )
            )
        else:
            encoded_parts.append(
                pd.to_numeric(df[covariate], errors="coerce").to_frame(covariate)
            )

    if not encoded_parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(encoded_parts, axis=1)


def residualize_fc_profiles(df, fc_columns, covariates=STANDARD_FC_COVARIATES):
    """Residualize FC profiles against covariates using grouped multi-output fits.

    Columns with the same missing-data pattern are fit together.  This is
    mathematically equivalent to fitting one linear regression per column, but
    is substantially faster and avoids DataFrame fragmentation for wide
    vertex-wise FC tables.
    """
    _validate_columns(df, [*fc_columns, *covariates], "FC residualization")
    df = df.copy()
    covariate_complete = _complete_covariate_mask(df, covariates)

    # Most FC columns have the same valid rows, so grouping by validity mask
    # usually reduces thousands of model fits to one multi-output regression.
    validity_groups = {}
    for column in fc_columns:
        valid_idx = df[column].notnull() & covariate_complete
        key = valid_idx.to_numpy(dtype=np.bool_).tobytes()
        if key not in validity_groups:
            validity_groups[key] = (valid_idx, [])
        validity_groups[key][1].append(column)

    residual_frames = []
    for valid_idx, columns in validity_groups.values():
        output_columns = [column + "_resid" for column in columns]
        residuals = pd.DataFrame(np.nan, index=df.index, columns=output_columns)
        if valid_idx.sum() > 0:
            observed = df.loc[valid_idx, columns].to_numpy()
            covariate_matrix = _encode_regression_covariates(
                df.loc[valid_idx],
                covariates,
            )
            if covariate_matrix.shape[1] == 0:
                predicted = np.tile(observed.mean(axis=0), (len(observed), 1))
            else:
                model = LinearRegression()
                model.fit(covariate_matrix, observed)
                predicted = model.predict(covariate_matrix)
            residuals.loc[valid_idx, output_columns] = observed - predicted
        residual_frames.append(residuals)

    if residual_frames:
        df = pd.concat([df, *residual_frames], axis=1)

    return df


def parse_vertexwise_fc_column(column):
    """Parse ``<network>_<Python index>_<L/R>_fz`` vertex-wise FC labels."""
    if not column.endswith("_fz"):
        return None

    parts = column[:-3].rsplit("_", 2)
    if len(parts) != 3:
        return None
    network, vertex, hemisphere = parts
    if not network or not vertex.isdigit() or hemisphere not in {"L", "R"}:
        return None
    return network, int(vertex), hemisphere


def get_vertexwise_fc_columns(df):
    """Return valid raw vertex-wise FC columns in their existing order."""
    columns = []
    for column in df.columns:
        parsed = parse_vertexwise_fc_column(column)
        if parsed is None:
            continue
        network, _, _ = parsed
        if network.endswith("_full"):
            continue
        columns.append(column)
    return columns


def drop_full_fc_columns(df):
    """Drop generated full-network Fisher-z FC columns from an analysis table."""
    full_fc_columns = [
        column
        for column in df.columns
        if "_fz" in str(column) and "_full" in str(column)
    ]
    if not full_fc_columns:
        return df, []
    return df.drop(columns=full_fc_columns), full_fc_columns


def get_region_fc_columns(df):
    """Return raw, non-full Fisher-z region FC columns."""
    return [
        column
        for column in df.columns
        if str(column).endswith("_fz") and "_full" not in str(column)
    ]


def remove_outliers(df, columns, threshold=8):
    """Remove rows with values beyond ``threshold`` SDs in any selected column."""
    df = df.copy()
    for column in columns:
        if column not in df.columns:
            continue

        mean = df[column].mean()
        std = df[column].std()
        if pd.isna(std) or std == 0:
            continue

        upper_limit = mean + threshold * std
        lower_limit = mean - threshold * std
        df = df[(df[column] <= upper_limit) & (df[column] >= lower_limit)]

    return df


def get_poverty_line(household_size, poverty_lines=POVERTY_LINES_2017):
    """Return the 2017 poverty line for a household size."""
    if (
        pd.isna(household_size)
        or household_size in INVALID_CODES
        or household_size < 1
    ):
        return np.nan
    if household_size <= 8:
        return poverty_lines[float(int(household_size))]

    extra = (int(household_size) - 8) * 4180
    return poverty_lines[8.0] + extra


def calculate_income_to_needs(
    df,
    income_col="demo_comb_income_v2",
    household_size_col="demo_roster_v2",
    output_col="inr",
    missing_indicator_col=None,
):
    """Compute income-to-needs ratio from observed income and household size.

    ``output_col`` preserves the observed INR value so analyses can still
    exclude participants whose INR could not be computed from income and
    household size.
    """
    _validate_columns(df, [income_col, household_size_col], "INR")
    df = df.copy()
    missing_indicator_col = missing_indicator_col or f"{output_col}_missing"

    income = pd.to_numeric(df[income_col], errors="coerce").replace(
        list(INVALID_CODES), np.nan
    )
    household_size = pd.to_numeric(df[household_size_col], errors="coerce").replace(
        list(INVALID_CODES), np.nan
    )

    df["income_median"] = income.map(INCOME_MEDIANS)
    df["poverty_line_2017"] = household_size.apply(get_poverty_line)
    df[output_col] = df["income_median"] / df["poverty_line_2017"]
    df[output_col] = df[output_col].replace([np.inf, -np.inf], np.nan)
    df[missing_indicator_col] = df[output_col].isna()
    return df


def residualize_measure(
    df,
    measure,
    covariates,
    output_col=None,
    invalid_codes=INVALID_CODES,
):
    """Create residuals for one measure after removing invalid ABCD codes."""
    _validate_columns(df, ["src_subject_id", measure, *covariates], "residualization")
    df = df.copy()
    output_col = output_col or f"{measure}_resid"
    if output_col in df.columns:
        df = df.drop(columns=[output_col])

    temp_df = df[["src_subject_id", measure, *covariates]].copy()
    temp_df[measure] = temp_df[measure].replace(list(invalid_codes), np.nan)
    temp_df = temp_df.dropna()

    if len(temp_df) == 0:
        df[output_col] = np.nan
        return df

    covariate_matrix = _encode_regression_covariates(temp_df, covariates)
    if covariate_matrix.shape[1] == 0:
        predicted = np.repeat(temp_df[measure].mean(), len(temp_df))
    else:
        model = LinearRegression().fit(covariate_matrix, temp_df[measure])
        predicted = model.predict(covariate_matrix)
    temp_df[output_col] = temp_df[measure] - predicted

    return df.merge(
        temp_df[["src_subject_id", output_col]],
        on="src_subject_id",
        how="left",
    )


def residualize_measures(df, measures, covariates, invalid_codes=INVALID_CODES):
    """Residualize a list of measures with the same covariate set."""
    df = df.copy()
    for measure in measures:
        df = residualize_measure(
            df,
            measure=measure,
            covariates=covariates,
            invalid_codes=invalid_codes,
        )
    return df


def calculate_effect_size(data, group_col="kmeans_2_consensus", value_col="value"):
    """Calculate Cohen's d for groups coded 0 and 1."""
    _validate_columns(data, [group_col, value_col], "effect-size")
    group_1 = data[data[group_col] == 0][value_col].dropna()
    group_2 = data[data[group_col] == 1][value_col].dropna()

    n1 = len(group_1)
    n2 = len(group_2)
    if n1 < 2 or n2 < 2:
        return np.nan

    std_1 = group_1.std(ddof=1)
    std_2 = group_2.std(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * std_1**2 + (n2 - 1) * std_2**2) / (n1 + n2 - 2))
    if pooled_std == 0 or pd.isna(pooled_std):
        return np.nan

    return (group_1.mean() - group_2.mean()) / pooled_std


def clean_group_measure(
    df,
    measure,
    group_col="kmeans_2_consensus",
    invalid_codes=INVALID_CODES,
):
    """Keep cluster labels and one cleaned measure for group comparisons."""
    _validate_columns(df, [group_col, measure], "group-comparison")
    return (
        df[[group_col, measure]]
        .replace(list(invalid_codes), np.nan)
        .dropna()
        .copy()
    )


def compute_correlations(df, measures, fc_columns, required_nonmissing=None):
    """Compute Pearson correlations for each measure-by-FC pair."""
    results = []
    required_nonmissing = list(required_nonmissing or [])

    for measure in measures:
        for column in fc_columns:
            _validate_columns(
                df,
                [measure, column, *required_nonmissing],
                "correlation",
            )
            analysis_columns = list(
                dict.fromkeys([column, measure, *required_nonmissing])
            )
            temp_df = df[analysis_columns].dropna()

            # scipy.pearsonr needs at least two observations and non-constant
            # vectors; return NaNs so the results table still records the test.
            if (
                len(temp_df) < 2
                or temp_df[column].nunique(dropna=True) < 2
                or temp_df[measure].nunique(dropna=True) < 2
            ):
                r_value = np.nan
                p_value = np.nan
            else:
                r_value, p_value = stats.pearsonr(temp_df[column], temp_df[measure])

            results.append(
                {
                    "measure": measure,
                    "col": column,
                    "r": r_value,
                    "p": p_value,
                    "n": len(temp_df),
                }
            )

    return pd.DataFrame(results)


def apply_multiple_comparison_corrections(results_df, alpha=0.05, fdr_group_col=None):
    """Add Bonferroni and FDR columns to a correlation results table.

    Bonferroni correction always uses the full results table. When
    ``fdr_group_col`` is provided, FDR correction is applied independently
    within each value of that column.
    """
    if results_df.empty:
        return results_df.assign(
            p_bonf=pd.Series(dtype=float),
            sig_bonf=pd.Series(dtype=bool),
            p_fdr=pd.Series(dtype=float),
            sig_fdr=pd.Series(dtype=bool),
        )

    corrected = results_df.copy()
    n_tests = len(corrected)
    corrected["p_bonf"] = np.minimum(corrected["p"] * n_tests, 1.0)
    corrected["sig_bonf"] = corrected["p_bonf"] < alpha

    valid_p = corrected["p"].notna()
    corrected["p_fdr"] = np.nan
    corrected["sig_fdr"] = False

    if fdr_group_col is None:
        fdr_families = pd.Series("all", index=corrected.index)
    else:
        _validate_columns(corrected, [fdr_group_col], "FDR grouping")
        fdr_families = corrected[fdr_group_col].astype("string").fillna("<missing>")

    for family in fdr_families.unique():
        family_valid_p = valid_p & fdr_families.eq(family)
        if family_valid_p.any():
            reject, p_fdr = fdrcorrection(
                corrected.loc[family_valid_p, "p"],
                alpha=alpha,
            )
            corrected.loc[family_valid_p, "p_fdr"] = p_fdr
            corrected.loc[family_valid_p, "sig_fdr"] = reject

    return corrected


def pairwise_rand_scores(df, columns):
    """Calculate pairwise Rand scores for all unique column pairs."""
    from sklearn.metrics import rand_score

    _validate_columns(df, columns, "Rand-score")
    rand_values = []
    for i, column_i in enumerate(columns):
        for column_j in columns[i + 1 :]:
            rand_values.append(rand_score(df[column_i], df[column_j]))
    return rand_values


def mean_pairwise_rand_score(df, columns):
    """Calculate the mean Rand score across all unique partition pairs.

    The running sum avoids retaining every pairwise value when hundreds or
    thousands of clustering runs are compared.
    """
    from sklearn.metrics import rand_score

    _validate_columns(df, columns, "Rand-score")
    if len(columns) < 2:
        raise ValueError("At least two assignment columns are required.")

    total = 0.0
    n_pairs = 0
    for i, column_i in enumerate(columns):
        for column_j in columns[i + 1 :]:
            total += rand_score(df[column_i], df[column_j])
            n_pairs += 1
    return total / n_pairs


def collapse_bootstrap_assignments(sample_indices, bootstrap_labels):
    """Collapse duplicate bootstrap draws to one modal label per subject.

    Bootstrap samples contain repeated subjects.  Identical observations will
    normally receive the same label, but a deterministic smallest-label tie
    break is used when a stochastic community detector assigns duplicates to
    different communities.
    """
    sample_indices = np.asarray(sample_indices)
    bootstrap_labels = np.asarray(bootstrap_labels)
    if sample_indices.ndim != 1 or bootstrap_labels.ndim != 1:
        raise ValueError("Bootstrap indices and labels must be one-dimensional.")
    if sample_indices.size != bootstrap_labels.size:
        raise ValueError("Bootstrap indices and labels must have equal length.")

    assignments = pd.DataFrame(
        {
            "subject_index": sample_indices,
            "bootstrap_label": bootstrap_labels,
        }
    )

    def modal_label(values):
        modes = values.mode(dropna=False)
        return modes.sort_values().iloc[0]

    collapsed = (
        assignments.groupby("subject_index", sort=True)["bootstrap_label"]
        .agg(modal_label)
    )
    return collapsed.index.to_numpy(dtype=int), collapsed.to_numpy()


def cluster_jaccard_scores(reference_labels, candidate_labels):
    """Return best-match Jaccard recovery for each reference cluster.

    Labels must describe the same aligned subjects.  Each reference cluster is
    compared with every candidate cluster, and its largest overlap is kept.
    This remains defined when a bootstrap solution contains a different number
    of communities from the reference solution.
    """
    reference_labels = np.asarray(reference_labels)
    candidate_labels = np.asarray(candidate_labels)
    if reference_labels.ndim != 1 or candidate_labels.ndim != 1:
        raise ValueError("Reference and candidate labels must be one-dimensional.")
    if reference_labels.size != candidate_labels.size:
        raise ValueError("Reference and candidate labels must have equal length.")
    if reference_labels.size == 0:
        raise ValueError("At least one aligned assignment is required.")

    scores = {}
    candidate_clusters = np.unique(candidate_labels)
    for reference_cluster in np.unique(reference_labels):
        reference_mask = reference_labels == reference_cluster
        best_score = 0.0
        for candidate_cluster in candidate_clusters:
            candidate_mask = candidate_labels == candidate_cluster
            intersection = np.count_nonzero(reference_mask & candidate_mask)
            union = np.count_nonzero(reference_mask | candidate_mask)
            if union:
                best_score = max(best_score, intersection / union)
        scores[reference_cluster] = best_score
    return scores
