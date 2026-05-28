#!/usr/bin/env python3

import nibabel as nib
import numpy as np
import pandas as pd
import os
from glob import glob
import subprocess
from nilearn import datasets, image

# Path to Workbench
workbench = '/labs/greene-lab/Shared_Tools/workbench/bin_linux64/wb_command'

# Surface files to use in workbench commands
surface_files = {
    'Left': '/labs/greene-lab/lab_members/emily/mtg/surfaces/HCP1200/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii',
    'Right': '/labs/greene-lab/lab_members/emily/mtg/surfaces/HCP1200/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii'
}

# Input paths
density_path = '/labs/greene-lab/lab_members/emily/mtg/maps/Variants_Density_cPFM_not06not11not13_185avrg.dtseries.nii'
avg_path = '/labs/greene-lab/lab_members/emily/mtg/maps/abcd_template_matching_combined_clusters_thresh0.61.dlabel.nii'
motion_path = '/labs/greene-lab/lab_members/emily/mtg/data/motion_data_summary.csv'
base_dir = '/labs/greene-lab/ABCD/derivatives/GL_Smoothing/'
subject_dirs = glob(os.path.join(base_dir, "sub-*"))

# Output paths
cluster_output_path = '/labs/greene-lab/lab_members/emily/mtg/maps/Variants_Density_cPFM_not06not11not13_185avrg_thresholded_clusters.dtseries.nii'
mtg_output_path = '/labs/greene-lab/lab_members/emily/mtg/maps/pMTG_regions.dtseries.nii'

# Step 1: Run cifti-find-clusters on the dtseries with 4 as the threshold and 10 vertices as the minimum cluster size
try:
    subprocess.run([
        workbench,
        '-cifti-find-clusters',
        density_path,
        '4',
        '40',
        '0',
        '0',
        'COLUMN',
        cluster_output_path,
        '-left-surface', surface_files['Left'],
        '-right-surface', surface_files['Right'],
    ], check=True)
    print("Clustering completed successfully.")
except subprocess.CalledProcessError as e:
    print("Error running wb_command -cifti-find-clusters:")
    print(e)
    exit(1)

# Step 2: Extract MTG indices (clusters 2 and 7 in the output) 
clusters_img = nib.load(cluster_output_path)
clusters = clusters_img.get_fdata()
mtg_left_list = np.where(clusters[0, :] == 2)[0]
print("Left MTG indices:", mtg_left_list)
mtg_right_list = np.where(clusters[0, :] == 7)[0]
print("Right MTG indices:", mtg_right_list)
mtg_list = np.concatenate((mtg_left_list, mtg_right_list))
print("Combined MTG indices:", mtg_list)
mtg_data = np.zeros(clusters.shape[1])
mtg_data[mtg_left_list] = 1
mtg_data[mtg_right_list] = 2
mtg_img = nib.Cifti2Image(mtg_data.reshape(1, -1), header=clusters_img.header)
nib.save(mtg_img, mtg_output_path)
print("MTG vertices saved to:", mtg_output_path)

# Step 3: Load average network map and compute group-average network time courses (excluding MTG vertices)
avg_img = nib.load(avg_path)
avg_data = avg_img.get_fdata().squeeze()

print('Shape of group average:', avg_data.shape)
print('Unique numeric labels in average map:', np.unique(avg_data))
network_labels = {'DMN_left': [1, 16, 18, 28, 30, 73, 74], 
                  'DMN_right': [33, 49, 58, 60, 62, 81, 84], 
                  'SMl_left': [2, 79, 96, 104], 
                  'SMl_right': [34, 90, 100, 106], 
                  'Vis_left': [3, 77, 91, 103], 
                  'Vis_right': [35, 89, 92, 105], 
                  'FP_left': [4, 21, 27, 29, 71, 76], 
                  'FP_right': [36, 37, 53, 59, 61, 82, 86, 87], 
                  'Aud_left': [5, 98], 
                  'Aud_right': [38, 102], 
                  'Tpole_left': [6, 64], 
                  'Tpole_right': [39, 65], 
                  'PMN_left': [7, 24], 
                  'PMN_right': [40, 55], 
                  'PON_left': [8, 26], 
                  'PON_right': [41, 57], 
                  'MTL_left': [9, 93], 
                  'MTL_right': [42, 94], 
                  'CO_left': [10, 17, 20, 32, 69, 75, 97], 
                  'CO_right': [43, 48, 51, 63, 101, 107], 
                  'Sal_left': [11, 23, 31, 66, 95], 
                  'Sal_right': [44, 54, 67, 99], 
                  'SMd_left': [12, 13, 68, 78], 
                  'SMd_right': [45, 80, 88], 
                  'DAN_left': [14, 15, 25, 70, 72], 
                  'DAN_right': [46, 47, 56, 83, 85], 
                  'VAN_left': [19, 22], 
                  'VAN_right': [50, 52],
                  'DMN_full': [1, 16, 18, 28, 30, 33, 49, 58, 60, 62, 73, 74, 81, 84], 
                  'SMl_full': [2, 34, 79, 90, 96, 100, 104, 106], 
                  'Vis_full': [3, 35, 77, 89, 91, 92, 103, 105], 
                  'FP_full': [4, 21, 27, 29, 36, 37, 53, 59, 61, 71, 76, 82, 86, 87], 
                  'Aud_full': [5, 38, 98, 102], 
                  'Tpole_full': [6, 39, 64, 65], 
                  'PMN_full': [7, 24, 40, 55], 
                  'PON_full': [8, 26, 41, 57], 
                  'MTL_full': [9, 42, 93, 94], 
                  'CO_full': [10, 17, 20, 32, 43, 48, 51, 63, 69, 75, 97, 101, 107], 
                  'Sal_full': [11, 23, 31, 44, 54, 66, 67, 95, 99], 
                  'SMd_full': [12, 13, 45, 68, 78, 80, 88], 
                  'DAN_full': [14, 15, 25, 46, 47, 56, 70, 72, 83, 85], 
                  'VAN_full': [19, 22, 50, 52]}
print('Defined network labels and their numeric labels.')
# find vertices in each network
network_indices = {}
for network, labels in network_labels.items():
    indices = np.where(np.isin(avg_data, labels))[0]
    # remove any MTG indices from the network mask
    indices = list(set(indices) - set(mtg_list))
    network_indices[network] = indices
    print(f'Network {network} has {len(indices)} vertices.')
print('Network indices computed.')


# Step 4: Average pMTG timecourses for each hemisphere, then compute 30-dimensional vector representing concatenated FC profiles to each large-scale network for the participant
motion = pd.read_csv(motion_path)
eligible_subjects = motion.loc[motion['good_frames'] >= 600, 'src_subject_id'].tolist()
print('Number of eligible subjects:', len(eligible_subjects))
results = []

for subject_path in subject_dirs:
    subject_id = os.path.basename(subject_path)
    if subject_id not in eligible_subjects:
        continue

    smoothed_dir = os.path.join(subject_path, "ses-2YearFollowUpYArm1", "Smoothed")
    if not os.path.isdir(smoothed_dir):
        continue

    dt_files = [f for f in os.listdir(smoothed_dir) if f.endswith("_CENSORED_6.0mm_SMOOTHED.dtseries.nii")]
    if not dt_files:
        continue

    dt_path = os.path.join(smoothed_dir, dt_files[0])

    try:
        # Load data
        dt = nib.load(dt_path).get_fdata() 

        subject_result = {'subject_id': subject_id}
        for network in network_labels.keys():
            # Left pMTG
            left_network_fcs = []
            left_mtg_indices = mtg_left_list
            left_mtg_timecourses = dt[:, left_mtg_indices]
            network_indices_current = network_indices[network]
            network_timecourse = np.mean(dt[:, network_indices_current], axis=1)
            for i, mtg_index in enumerate(left_mtg_indices):
                fc = np.corrcoef(left_mtg_timecourses[:, i], network_timecourse)[0, 1]
                fz_fc = np.arctanh(fc)  # Fisher z-transform
                left_network_fcs.append(fz_fc)

            # Right pMTG
            right_network_fcs = []
            right_mtg_indices = mtg_right_list
            right_mtg_timecourses = dt[:, right_mtg_indices]
            for i, mtg_index in enumerate(right_mtg_indices):
                fc = np.corrcoef(right_mtg_timecourses[:, i], network_timecourse)[0, 1]
                fz_fc = np.arctanh(fc)  # Fisher z-transform
                right_network_fcs.append(fz_fc)

            subject_result[f'{network}_L_fz'] = np.mean(left_network_fcs)
            subject_result[f'{network}_R_fz'] = np.mean(right_network_fcs)
        results.append(subject_result)
        # break # remove to process all subjects

    except Exception as e:
        print(f"Error processing subject {subject_id}: {e}")
        continue

# Step 5: Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("/labs/greene-lab/lab_members/emily/mtg/data/pMTG_FC_profiles_midb61_meanFC.csv", index=False)
print("Saved FC profiles to CSV.")




