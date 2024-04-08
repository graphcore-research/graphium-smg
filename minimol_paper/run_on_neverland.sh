#!/bin/bash
date; hostname; pwd;
cd /nethome/blazejb/graphium-smg
source enable_ipu.sh

model_name="MPNN_SameAsGine" fingerprints_path="results/ids_to_fingerprint_MPNN_SameAsGine_50M.pt"

# model_name="GINE_AdmetFiltered_k2" fingerprints_path="results/ids_to_fingerprint_GINE_AdmetFiltered_k2_10M.pt"



python sweep_checkpoint_on_all_tdc.py --model-name "$model_name" --fingerprints-path "$fingerprints_path"

echo "All Python programs have completed."





# Done
# model_name="GINE_AdmetFiltered_k1" fingerprints_path="results/ids_to_fingerprint_GINE_AdmetFiltered_k1_10M.pt"
# model_name="GINE_AdmetFiltered_k10" fingerprints_path="results/ids_to_fingerprint_GINE_AdmetFiltered_k10_10M.pt"

# model_name="GINE_AdmetFiltered_seed11" fingerprints_path="results/ids_to_fingerprint_GINE_AdmetFiltered_seed11_10M.pt"
# model_name="GINE_AdmetFiltered_seed6" fingerprints_path="results/ids_to_fingerprint_GINE_AdmetFiltered_seed6_10M.pt"

# model_name="GINE_AdmetFiltered_maxpool" fingerprints_path="results/ids_to_fingerprint_GINE_AdmetFiltered_maxpool_10M.pt"
# model_name="GINE_AdmetFiltered_meanpool" fingerprints_path="results/ids_to_fingerprint_GINE_AdmetFiltered_meanpool_10M.pt"


# model_name="agbt_baseline_ss=18" fingerprints_path="results/ids_to_fingerprintagbt.pt"
# model_name="bet_baseline_ss=18" fingerprints_path="results/ids_to_fingerprintbet.pt"
# model_name="molformer_baseline_ss=18" fingerprints_path="results/ids_to_fingerprintmolformer.pt"

# model_name="mpnn_baseline_ss=432" fingerprints_path="results/ids_to_fingerprint_MPNN_AdmetFiltered_lastnorm_10M.pt"

# model_name="gcn_baseline_ss=18" fingerprints_path="results/ids_to_fingerprint_GCN_AdmetFiltered_lastnorm_10M.pt"
# model_name="mpnn_baseline_ss=18" fingerprints_path="results/ids_to_fingerprint_MPNN_AdmetFiltered_lastnorm_10M.pt"
# model_name="gine_baseline_ss=18" fingerprints_path="results/ids_to_fingerprint_GINE_AdmetFiltered_lastnorm_10M.pt"

# model_name="gine_baseline_ss=5" fingerprints_path="results/ids_to_fingerprint_GINE_AdmetFiltered_lastnorm_10M.pt"

# model_name="GINE_filtered_pcba" fingerprints_path="results/ids_to_fingerprint_GINE_filtered_pcba_10M.pt"
# model_name="GINE_filtered_pcba_lr3e-4_40M" fingerprints_path="results/ids_to_fingerprint_GINE_filtered_pcba_lr3e-4_40M.pt"
# model_name="GINE_filtered_pcba_lr4e-4_40M" fingerprints_path="results/ids_to_fingerprint_GINE_filtered_pcba_lr4e-4_40M.pt"
# model_name="GCN_AdmetFiltered_ensemble_10M" fingerprints_path="results/ids_to_fingerprint_GCN_AdmetFiltered_10M.pt"

# model_name="GINE_AdmetFiltered_ensemble_10M" fingerprints_path="results/ids_to_fingerprint_GINE_AdmetFiltered_10M.pt"
# model_name="GINE_AdmetFiltered_lastnorm_ensemble_10M" fingerprints_path="results/ids_to_fingerprint_GINE_AdmetFiltered_lastnorm_10M.pt"

# model_name="ENSEMBLE_GINE_large-sweep_10M" fingerprints_path="results/ids_to_fingerprint_GINE_10M.pt"
# model_name="ENSEMBLE_GCN_10M" fingerprints_path="results/ids_to_fingerprint_GCN_10M.pt"
# model_name="ENSEMBLE_10M" fingerprints_path="results/ids_to_fingerprint_FilteredAdmet_10M.pt"

# baselines
# model_name="GCN_10M" fingerprints_path="results/ids_to_fingerprint_GCN_10M.pt"
# model_name="GINE_10M" fingerprints_path="results/ids_to_fingerprint_GINE_10M.pt"
# model_name="FilteredAdmet_10M" fingerprints_path="results/ids_to_fingerprint_FilteredAdmet_10M.pt"

# BugFixOg_last-layernorm
# model_name="BugFixOg_last-layernorm_10M_best" fingerprints_path="results/ids_to_fingerprint_BugFixOg_last-layernorm_10M_best.pt"
# model_name="BugFixOg_last-layernorm_10M_last" fingerprints_path="results/ids_to_fingerprint_BugFixOg_last-layernorm_10M_last.pt"
# model_name="BugFixOg_last-layernorm_35M_best" fingerprints_path="results/ids_to_fingerprint_BugFixOg_last-layernorm_35M_best.pt"
# model_name="BugFixOg_last-layernorm_35M_last" fingerprints_path="results/ids_to_fingerprint_BugFixOg_last-layernorm_35M_last.pt"
# model_name="BugFixOg_last-layernorm_130M_best" fingerprints_path="results/ids_to_fingerprint_BugFixOg_last-layernorm_130M_best.pt"
# model_name="BugFixOg_last-layernorm_130M_last" fingerprints_path="results/ids_to_fingerprint_BugFixOg_last-layernorm_130M_last.pt"
# model_name="BugFixOg_last-layernorm_300M_best" fingerprints_path="results/ids_to_fingerprint_BugFixOg_last-layernorm_300M_best.pt"

# BugFixOg_2e-1-g25-loss_g25do
# model_name="BugFixOg_2e-1-g25-loss_g25do_35M_best" fingerprints_path="results/ids_to_fingerprint_BugFixOg_2e-1-g25-loss_g25do_35M_best.pt"
# model_name="BugFixOg_2e-1-g25-loss_g25do_35M_last" fingerprints_path="results/ids_to_fingerprint_BugFixOg_2e-1-g25-loss_g25do_35M_last.pt"
# model_name="BugFixOg_2e-1-g25-loss_g25do_130M_best" fingerprints_path="results/ids_to_fingerprint_BugFixOg_2e-1-g25-loss_g25do_130M_best.pt"
# model_name="BugFixOg_2e-1-g25-loss_g25do_130M_last" fingerprints_path="results/ids_to_fingerprint_BugFixOg_2e-1-g25-loss_g25do_130M_last.pt"

# BugFixOg_5e-1-g25-loss
# model_name="BugFixOg_5e-1-g25-loss_10M_best" fingerprints_path="results/ids_to_fingerprint_BugFixOg_5e-1-g25-loss_10M_best.pt"
# model_name="BugFixOg_5e-1-g25-loss_10M_last" fingerprints_path="results/ids_to_fingerprint_BugFixOg_5e-1-g25-loss_10M_last.pt"
# model_name="BugFixOg_5e-1-g25-loss_35M_best" fingerprints_path="results/ids_to_fingerprint_BugFixOg_5e-1-g25-loss_35M_best.pt"
# model_name="BugFixOg_5e-1-g25-loss_130M_best" fingerprints_path="results/ids_to_fingerprint_BugFixOg_5e-1-g25-loss_130M_best.pt"
# model_name="BugFixOg_5e-1-g25-loss_300M_best" fingerprints_path="results/ids_to_fingerprint_BugFixOg_5e-1-g25-loss_300M_best.pt"

# BugFixOg g25-loss
# model_name="BugFixOg_10M_g25-loss" fingerprints_path="results/ids_to_fingerprint_BugFixOg_10M_g25-loss.pt"
# model_name="BugFixOg_10M_g25-loss_last" fingerprints_path="results/ids_to_fingerprint_BugFixOg_10M_g25-loss_last.pt"
# model_name="BugFixOg_35M_g25-loss_best" fingerprints_path="results/ids_to_fingerprint_BugFixOg_35M_g25-loss_best.pt"
# model_name="BugFixOg_35M_g25-loss_last" fingerprints_path="results/ids_to_fingerprint_BugFixOg_35M_g25-loss_last.pt"
# model_name="BugFixOg_130M_g25-loss_last" fingerprints_path="results/ids_to_fingerprint_BugFixOg_130M_g25-loss_last.pt"
# model_name="BugFixOg_130M_g25-loss_best" fingerprints_path="results/ids_to_fingerprint_BugFixOg_130M_g25-loss_best.pt"
# model_name="BugFixOg_300M_g25-loss_best" fingerprints_path="results/ids_to_fingerprint_BugFixOg_300M_g25-loss_best.pt"

# BugFixOg 1e-4
# model_name="BugFixOg_10M_1e-4" fingerprints_path="results/ids_to_fingerprint_BugFixOg_10M_1e-4_best.pt"
# model_name="BugFixOg_35M_1e-4" fingerprints_path="results/ids_to_fingerprint_BugFixOg_35M_1e-4_best.pt"
# model_name="BugFixOg_130M_1e-4_best" fingerprints_path="results/ids_to_fingerprint_BugFixOg_130M_1e-4_best.pt"
# model_name="BugFixOg_300M_1e-4_best" fingerprints_path="results/ids_to_fingerprint_BugFixOg_300M_1e-4_best.pt"

# BugFixOg 2e-4
# model_name="BugFixOg_10M" fingerprints_path="results/ids_to_fingerprint_BugFixOg_10M.pt"
# model_name="BugFixOg_10M_last" fingerprints_path="results/ids_to_fingerprint_BugFixOg_10M_last.pt"
# model_name="BugFixOg_35M" fingerprints_path="results/ids_to_fingerprint_BugFixOg_35M.pt"
# model_name="BugFixOg_35M_last" fingerprints_path="results/ids_to_fingerprint_BugFixOg_35M_last.pt"
# model_name="BugFixOg_130M_best" fingerprints_path="results/ids_to_fingerprint_BugFixOg_130M_best.pt"
# model_name="BugFixOg_130M_last" fingerprints_path="results/ids_to_fingerprint_BugFixOg_130M_last.pt"
# model_name="BugFixOg_300M_best" fingerprints_path="results/ids_to_fingerprint_BugFixOg_300M_best.pt"






# BackToOg + easy-th + mup + fraction-sampling
# model_name="BackToOg_35M_easy-th_mup_frac-sampl" fingerprints_path="results/ids_to_fingerprint_BackToOg_35M_easy-th_mup_frac-sampl.pt"
# model_name="BackToOg_130M_easy-th_mup_frac-sampl" fingerprints_path="results/ids_to_fingerprint_BackToOg_130M_easy-th_mup_frac-sampl.pt"

# BackToOg + easy-th + mup
# model_name="BackToOg_10M_easy-th_mup_last"; fingerprints_path="results/ids_to_fingerprint_BackToOg_10M_easy-th_mup_last.pt"
# model_name="BackToOg_10M_easy-th_mup_best"; fingerprints_path="results/ids_to_fingerprint_BackToOg_10M_easy-th_mup_best.pt"
# model_name="BackToOg_35M_easy-th_mup_best"; fingerprints_path="results/ids_to_fingerprint_BackToOg_35M_easy-th_mup.pt"
# model_name="BackToOg_130M_easy-th_mup_best"; fingerprints_path="results/ids_to_fingerprint_BackToOg_130M_easy-th_mup.pt"
# model_name="BackToOg_300M_easy-th_mup_best"; fingerprints_path="results/ids_to_fingerprint_BackToOg_300M_easy-th_mup.pt"

# BackToOg
# model_name="BackToOg_10M"; fingerprints_path="results/ids_to_fingerprint_BackToOg_10M.pt"
# model_name="BackToOg_35M"; fingerprints_path="results/ids_to_fingerprint_BackToOg_35M.pt"
# model_name="BackToOg_130M"; fingerprints_path="results/ids_to_fingerprint_BackToOg_130M.pt"

# MPNN + easy-th
# model_name="11M-easy-th"; fingerprints_path="results/ids_to_fingerprint_11M-easy-th.pt"
# model_name="40M-MPNN-easy-th"; fingerprints_path="results/ids_to_fingerprint_40M-MPNN-easy-th.pt"

# Others
# model_name="38M-GPS++"; fingerprints_path="results/ids_to_fingerprint_38M-GPS++.pt"
# model_name="13M"; fingerprints_path="results/ids_to_fingerprint_13M.pt"
# model_name="11M-old-GPS++"; fingerprints_path="results/ids_to_fingerprint_11M-old-GPS++.pt"
