# pMTG Functional Connectivity and Brain–Behavior Analyses

This repository contains the analysis code for characterizing posterior middle temporal gyrus (pMTG) functional connectivity, identifying connectivity-based subtypes, and testing relationships with cognition and socioeconomic status in early adolescence.

## Analysis Workflow

1. `variants.ipynb` evaluates the spatial robustness of the pMTG regions using threshold, diagnosis-exclusion, and jackknife analyses.
2. `calculating_fc.py` calculates Fisher-z-transformed connectivity between left and right pMTG regions and large-scale functional networks.
3. `data_wrangling.ipynb` merges functional connectivity, demographic, behavioral, imaging-quality, and motion data; applies the sample-selection criteria; and prepares the analysis table.
4. `PCA_tasks.ipynb` performs exploratory factor analysis of the cognitive measures and repeats the analysis after controlling for income-to-needs ratio (INR).
5. `PCA_FC.ipynb` performs principal component analysis of pMTG connectivity, exports PCA scores for visualization, tests associations between connectivity and the cognitive factors, and tests direct associations between connectivity and INR.
6. `clustering.ipynb` identifies pMTG connectivity subtypes using Infomap, Louvain, and K-Means. It also evaluates repeated-run agreement and bootstrap stability and visualizes two- and four-cluster solutions in the previously estimated PCA space.
7. `brain_behavior_without_inr_residualization.ipynb` and `brain_behavior_with_inr_residualization.ipynb` run matched brain–behavior and subtype analyses without and with INR residualization. The no-INR analysis also tests mediation by pMTG connectivity.
8. `figures.ipynb` produces downstream subtype figures.

## Covariate Models

The primary functional-connectivity models control for age, sex, study site, handedness, and mean framewise displacement. Cognitive measures are adjusted for age, sex, study site, and handedness. SES-adjusted analyses add INR to the corresponding functional-connectivity and cognitive models.

Sex, study site, and handedness are modeled as categorical variables. Participants missing variables required for a given model are excluded from that analysis.

## Data and Software

The ABCD and precision functional mapping data used in this project are access-controlled and are not distributed with the repository. Generated analysis outputs are also not included. Input and output paths must therefore be configured in the relevant notebooks before execution.

The CIFTI spatial analyses require Connectome Workbench and the HCP S1200 32k midthickness surfaces. Python dependencies are imported within each script or notebook.
