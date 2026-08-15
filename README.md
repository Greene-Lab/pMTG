# pMTG Functional Connectivity and Brain-Behavior Analyses

This repository contains notebooks and helper code for analyzing posterior middle temporal gyrus (pMTG) functional connectivity profiles, clustering participants into connectivity subtypes, and testing associations with cognition and socioeconomic measures.

## Repository Contents

- `calculating_fc.py`: Generates subject-level pMTG-to-network functional connectivity profiles from CIFTI time series.
- `data_wrangling.ipynb`: Merges ABCD demographic, behavioral, imaging QA, motion QA, and pMTG FC data; filters the sample; residualizes FC profiles; and saves the analysis table.
- `brain_behavior_without_inr_residualization.ipynb`: Explicit no-INR-covariate brain-behavior notebook; it uses the no-SES cognitive EFA factor scores.
- `brain_behavior_with_inr_residualization.ipynb`: Brain-behavior notebook that controls FC for income-to-needs ratio (INR); it uses the SES-residualized cognitive EFA factor scores.
- `clustering.ipynb`: One parameterized clustering notebook. `RESIDUALIZE_FC_FOR_INR` selects standard-covariate or standard-plus-INR FC residualization without duplicating the analysis. It assesses K-Means and Louvain stability with 5,000 with-replacement subject bootstraps, summarizes agreement across 1,001 runs, and plots two- and four-cluster solutions using the matching saved FC-PCA scores.
- `variants.ipynb`: Self-contained Workbench workflow for pMTG spatial validation. It thresholds unthresholded subject spatial-correlation maps at the bottom 5%, 10%, and 15%, clusters subject and group maps at 30 mm2, identifies group pMTG parcels, and reports Dice coefficient overlap for threshold, diagnosis-exclusion, and 10% jackknife analyses.
- `PCA_tasks.ipynb`: Cognitive-task dimensionality-reduction notebook. It runs matched no-SES and SES-residualized PCA and exploratory factor analysis (EFA) workflows, displays cognitive residual normality diagnostics with histograms and Q-Q plots, and exports cognitive PCA/EFA scores and parameter tables.
- `PCA_FC.ipynb`: Functional-connectivity PCA notebook. It loads the cognitive EFA factor-score exports from `PCA_tasks.ipynb`, runs matched no-SES and SES-residualized FC PCA workflows, exports the fitted FC-PCA scores and variance tables for downstream clustering visualization, displays FC residual normality diagnostics with paginated histograms and Q-Q plots, computes cognitive-EFA-factor-by-FC association tables, and tests FDR-corrected associations between INR and FC residualized for the standard covariates without INR.
- `figures.ipynb`: Downstream subtype visualization notebook.
- `variant_spatial_validation_results/`: Generated CSV summaries, figures, and CIFTI maps from the Workbench-based pMTG spatial validation analyses.

## Workflow

1. `calculating_fc.py` identifies pMTG vertices, extracts network time courses, computes Fisher-z transformed pMTG-to-network FC profiles, and writes the subject-level FC CSV.
2. `data_wrangling.ipynb` loads ABCD behavioral/demographic data, merges FC profiles, merges `mean_fd_0.20` from `motion_QA_results.csv`, filters rsfMRI QA exclusions, selects one participant per family, residualizes FC profiles, removes extreme FC outliers, and exports the final wrangled CSV.
3. `PCA_FC.ipynb` exports matched FC-PCA coordinates, tests the direct relationship between INR and each standard-covariate-residualized FC measure with FDR correction, and saves the full FC-SES result table. The parameterized clustering notebook uses the fixed PCA coordinates to visualize subtype solutions. Clustering stability is evaluated with cluster-wise Jaccard recovery across 5,000 with-replacement subject bootstraps; Rand Index is reported separately across repeated full-sample runs and between final algorithms. `PCA_tasks.ipynb` exports cognitive EFA factor scores used in FC PCA and brain-behavior analyses.
4. The brain-behavior notebooks compute INR, residualize FC as configured, keep non-task subtype comparisons, correlate FC with cognitive EFA factors, FDR-correct the cognitive factor tests, plot significant effects, and test mediation models where appropriate.
5. `variants.ipynb` derives subject variant maps from the unthresholded spatial-correlation CIFTIs, applies Workbench component filtering, clusters group consensus maps, exports candidate pMTG labels for manual confirmation, and quantifies pMTG spatial overlap with Dice coefficient.

## Residualization

The standard FC residualization covariates are:

- `demo_sex_v2`
- `interview_age`
- `site_id_l`
- `ehi1b`
- `mean_fd_0.20`

`mean_fd_0.20` is expected in `motion_QA_results.csv` and is merged by subject ID before FC residualization in `data_wrangling.ipynb`.
Sex, scanner site, and handedness (`ehi1b`) are treated as categorical variables in residualization models.

For brain-behavior analyses:

- `brain_behavior_without_inr_residualization.ipynb` merges no-SES cognitive EFA factors, residualizes FC for the standard covariates, and tests FC associations with the EFA factors. It also compares KMeans subtype groups on the individual task scores used by `PCA_tasks.ipynb`, including Flanker, residualized for age, sex, site, and handedness.
- `brain_behavior_with_inr_residualization.ipynb` adds observed INR (`inr`) to the FC residualization covariate set and merges SES-residualized cognitive EFA factors. It also compares KMeans subtype groups on the individual task scores used by `PCA_tasks.ipynb`, including Flanker, residualized for age, sex, site, handedness, and INR; raw INR remains available for descriptive subtype comparisons.
- `clustering.ipynb` recomputes FC residuals from raw FC columns using the standard covariates and adds observed INR when `RESIDUALIZE_FC_FOR_INR = True`.

`calculate_income_to_needs` keeps observed INR in `inr` and records missingness in `inr_missing`. SES-residualized FC and behavioral models use observed `inr`; participants missing any required residualization inputs keep `NaN` FC residuals instead of receiving filled covariates.

## Notebook Organization

The active analyses are written as step-by-step research notebooks. Small functions are kept inside the notebook where they are used, primarily when an operation is repeated many times (for example, FC residualization or fitting a bootstrap clustering solution). Data loading, output writing, summaries, and validation metrics are shown directly in the analysis flow rather than hidden behind utility modules.

The clustering notebook presents validation in four explicit stages: repeated-run Rand Index, bootstrap setup, 5,000 with-replacement bootstrap fits, and cluster-wise Jaccard summaries.

Every clustering CSV filename includes either `with_inr_residualization` or `without_inr_residualization`; motion is not included in the output label.

## Path Configuration

The notebooks currently use absolute local paths for ABCD data, FC profile CSVs, and output files. Update the path variables near the top of each notebook before running on another machine.

## Assumptions

- `motion_QA_results.csv` contains `src_subject_id` and `mean_fd_0.20`.
- Subject IDs may appear with underscores or a `sub-` prefix; helper functions normalize these before merging.
- INR covariate control means adding observed INR to the same residualization model as the other FC covariates and using the matched SES-residualized cognitive EFA scores from `PCA_tasks.ipynb`.
- Private ABCD source files and generated analysis outputs are not included in this repository.
- The reported cPFM cohort comprises `MSCPI05`, `MSCPI07`, `MSCPI08`, `MSCPI10`, `MSCPI12`, `MSCPI14`, `MSCPI15`, `MSCPI17`, `MSCPI18`, and `MSCPI19`; participants `MSCPI14`, `MSCPI15`, and `MSCPI18` form the neurodevelopmental-diagnosis group.
- Variant spatial validation starts from unthresholded participant-level `*spatialCorrMap.dtseries.nii` maps in the local `Variants` folder.
- Connectome Workbench is expected at `/Applications/workbench/bin_macosx64/wb_command`, with HCP1200 32k midthickness surfaces available at the paths configured in `variants.ipynb`.
- The notebook automatically ranks candidate pMTG parcels by distance from the manuscript pMTG centers, but exported labels should be visually confirmed in Workbench and overridden in the notebook if needed.
