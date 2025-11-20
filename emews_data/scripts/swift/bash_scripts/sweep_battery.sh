#! /bin/bash

# RUNS WITH AGSv2 -- now with a different initial setup
SWEEP_PARAMS_TXT_PATH='./data/JSON/sweep/sweep_txt'
SETTINGS_PATH_CONTROL='./data/physiboss_config/control'
SETTINGS_PATH_SINGLE_DRUG='./data/physiboss_config/single_drug'
SETTINGS_PATH_SINGLE_DRUG_LINEAR_MAPPING='./data/physiboss_config/single_drug_linear_mapping'
SETTINGS_PATH_COMBINED_DRUG='./data/physiboss_config/combined_drug'
SETTINGS_PATH_COMBINED_DRUG_LOWPULESE='./data/physiboss_config/combined_drug_lowpulse_sweep'
SETTINGS_PATH_MED_EXPERIMENTS='./data/physiboss_config/minimum_effective_dose_configs'
SETTINGS_PATH_3D_DRUG_ADDITION='./data/physiboss_config/3D_above_drugtreatment'
SETTINGS_PATH_DOSE_RESPONSE_EXPERIMENTS='./data/physiboss_config/dose_curves_experiments'
SETTINGS_PATH_COMBINED_DRUG_LINEAR_MAPPING='./data/physiboss_config/combined_drug_linear_mapping'
SETTINGS_PATH_COMBINED_DRUG_TRANSIENT='./data/physiboss_config/combined_drug_transient'
SETTINGS_PATH_SINGLE_DRUG_TRANSIENT='./data/physiboss_config/single_drug_transient'
RUN_SWEEP_PATH='./swift/run_sweep_ags.sh'
CURRENT_DATE_TIME=$(date +"%d%m-%H%M")


#### EXAMPLE RUN COMMANDS ####
# bash RUN_SWIFT.sh NAMEFILE PARAMS_TO_EXPLORE PHYSIBOSS_SETTINGS_FILE EXPERIMENT_NAME
# bash $RUN_SWEEP_PATH sweep-$CURRENT_DATE_TIME-11p_synergy_PI3KMEK   $SWEEP_PARAMS_TXT_PATH/sweep_11p_PI3KMEK_synergy.txt $SETTINGS_PATH_CONTROL/settings_AGSv2_SYN_PI3K_MEK_halfGI50.xml PI3K_MEK RMSE_DTW



################## DOSE-RESPONSE EXPERIMENTS ##################
# bash $RUN_SWEEP_PATH DR_CURVE_PI3Ki-sweep-$CURRENT_DATE_TIME-100-drug_threshold-1p   $SWEEP_PARAMS_TXT_PATH/sweep_1p_dose_response_curve_PI3K_uniform_100.txt $SETTINGS_PATH_DOSE_RESPONSE_EXPERIMENTS/PI3Ki_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_top_10p_averaged_top10p_averaged.xml PI3K FINAL_NUMBER_OF_ALIVE_CELLS
# bash $RUN_SWEEP_PATH DR_CURVE_MEKi-sweep-$CURRENT_DATE_TIME-100-drug_threshold-1p   $SWEEP_PARAMS_TXT_PATH/sweep_1p_dose_response_curve_MEK_uniform_100.txt $SETTINGS_PATH_DOSE_RESPONSE_EXPERIMENTS/MEKi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_top_10p_averaged_top10p_averaged.xml MEK FINAL_NUMBER_OF_ALIVE_CELLS
# bash $RUN_SWEEP_PATH DR_CURVE_AKTi-sweep-$CURRENT_DATE_TIME-100-drug_threshold-1p   $SWEEP_PARAMS_TXT_PATH/sweep_1p_dose_response_curve_AKT_uniform_100.txt $SETTINGS_PATH_DOSE_RESPONSE_EXPERIMENTS/AKTi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_top_10p_averaged_top10p_averaged.xml AKT FINAL_NUMBER_OF_ALIVE_CELLS


################## SYNERGY SWEEPS ##################

# PI3K-MEK synergy sweep with transient delayed effect and 18 parameters
# bash $RUN_SWEEP_PATH synergy_sweep-pi3k_mek-$CURRENT_DATE_TIME-18p_transient_delayed_uniform_5k_10p   $SWEEP_PARAMS_TXT_PATH/sweep_combined_PI3Ki_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_and_MEKi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_top_10p_uniform_5000.txt $SETTINGS_PATH_COMBINED_DRUG_TRANSIENT/settings_AGSv2_SYN_PI3K_MEK_halfGI50_transient.xml PI3K_MEK RMSE_SK_POSTDRUG
# bash $RUN_SWEEP_PATH synergy_sweep-pi3k_mek-$CURRENT_DATE_TIME-18p_PI3K_transient_delayed_uniform_5k_10p   $SWEEP_PARAMS_TXT_PATH/PI3Ki_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_single_drug_sweep_combined_PI3Ki_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_MEKi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_top_10p_uniform_5000.txt $SETTINGS_PATH_SINGLE_DRUG_TRANSIENT/settings_AGSv2_PI3K_GI50_transient_delayed.xml PI3K RMSE_SK_POSTDRUG
# bash $RUN_SWEEP_PATH synergy_sweep-pi3k_mek-$CURRENT_DATE_TIME-18p_MEK_transient_delayed_uniform_5k_10p   $SWEEP_PARAMS_TXT_PATH/MEKi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_single_drug_sweep_combined_PI3Ki_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_MEKi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_top_10p_uniform_5000.txt $SETTINGS_PATH_SINGLE_DRUG_TRANSIENT/settings_AGSv2_MEK_GI50_transient_delayed.xml MEK RMSE_SK_POSTDRUG

# # AKT-MEK synergy sweep with transient delayed effect and 18 parameters
# bash $RUN_SWEEP_PATH synergy_sweep-akt_mek-$CURRENT_DATE_TIME-18p_transient_delayed_uniform_5k_10p   $SWEEP_PARAMS_TXT_PATH/sweep_combined_AKTi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_and_MEKi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_top_10p_uniform_5000.txt $SETTINGS_PATH_COMBINED_DRUG_TRANSIENT/settings_AGSv2_SYN_AKT_MEK_halfGI50_transient.xml AKT_MEK RMSE_SK_FINAL
# bash $RUN_SWEEP_PATH synergy_sweep-akt_mek-$CURRENT_DATE_TIME-18p_transient_delayed_uniform_postdrug_RMSE_5k   $SWEEP_PARAMS_TXT_PATH/sweep_combined_AKTi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_and_MEKi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_top_10p_uniform_5000.txt $SETTINGS_PATH_COMBINED_DRUG_TRANSIENT/settings_AGSv2_SYN_AKT_MEK_halfGI50_transient.xml AKT_MEK RMSE_SK_POSTDRUG
# bash $RUN_SWEEP_PATH synergy_sweep-akt_mek-$CURRENT_DATE_TIME-18p_AKT_transient_delayed_uniform_5k_singledrug   $SWEEP_PARAMS_TXT_PATH/AKTi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_single_drug_sweep_combined_AKTi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_MEKi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_top_10p_uniform_5000.txt $SETTINGS_PATH_SINGLE_DRUG_TRANSIENT/settings_AGSv2_AKT_GI50_transient_delayed.xml AKT RMSE_SK_POSTDRUG
# bash $RUN_SWEEP_PATH synergy_sweep-akt_mek-$CURRENT_DATE_TIME-18p_MEK_transient_delayed_uniform_5k_singledrug   $SWEEP_PARAMS_TXT_PATH/MEKi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_single_drug_sweep_combined_AKTi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_MEKi_CMA-0704-1815-18p_delayed_transient_rmse_postdrug_25gen_top_10p_uniform_5000.txt $SETTINGS_PATH_SINGLE_DRUG_TRANSIENT/settings_AGSv2_MEK_GI50_transient_delayed.xml MEK RMSE_SK_POSTDRUG


################## 3D DRUG TIMING AND DIFFUSION EXPERIMENTS ##################
# RE-RUNNING EXPERIMENTS WITH CONSENSUS PARAMETERS (updated generate_consensus_parameter.py)
# 3D drug addition Drug diffusion coefficient parameter sweep  + drug timing (4p) WITH CONSENSUS PARAMETERS (updated generate_consensus_parameter.py)
bash $RUN_SWEEP_PATH synergy_sweep-akt_mek-$CURRENT_DATE_TIME-4p_3D_drugtiming_synonly_consensus_hybrid_20   $SWEEP_PARAMS_TXT_PATH/sweep_consensus_akt_mek_top1p_structured_hybrid_20.txt $SETTINGS_PATH_3D_DRUG_ADDITION/settings_AGSv2_3D_SYN_AKT_MEK_synergy_only_consensus.xml AKT_MEK FINAL_NUMBER_OF_ALIVE_CELLS
bash $RUN_SWEEP_PATH synergy_sweep-pi3k_mek-$CURRENT_DATE_TIME-4p_3D_drugtiming_synonly_consensus_hybrid_20   $SWEEP_PARAMS_TXT_PATH/sweep_consensus_pi3k_mek_top1p_structured_hybrid_20.txt $SETTINGS_PATH_3D_DRUG_ADDITION/settings_AGSv2_3D_SYN_PI3K_MEK_synergy_only_consensus.xml PI3K_MEK FINAL_NUMBER_OF_ALIVE_CELLS

# 2D drug addition Drug diffusion coefficient parameter sweep  + drug timing (4p) WITH CONSENSUS PARAMETERS (updated generate_consensus_parameter.py)
bash $RUN_SWEEP_PATH synergy_sweep-akt_mek-$CURRENT_DATE_TIME-4p_2D_drugtiming_synonly_consensus_hybrid_20   $SWEEP_PARAMS_TXT_PATH/sweep_consensus_akt_mek_top1p_structured_hybrid_20.txt $SETTINGS_PATH_3D_DRUG_ADDITION/settings_AGSv2_2D_SYN_AKT_MEK_synergy_only_consensus.xml AKT_MEK FINAL_NUMBER_OF_ALIVE_CELLS
bash $RUN_SWEEP_PATH synergy_sweep-pi3k_mek-$CURRENT_DATE_TIME-4p_2D_drugtiming_synonly_consensus_hybrid_20   $SWEEP_PARAMS_TXT_PATH/sweep_consensus_pi3k_mek_top1p_structured_hybrid_20.txt $SETTINGS_PATH_3D_DRUG_ADDITION/settings_AGSv2_2D_SYN_PI3K_MEK_synergy_only_consensus.xml PI3K_MEK FINAL_NUMBER_OF_ALIVE_CELLS



# OTHER EXPERIMENTS

# 3D Negative control: Just cells without drugs growing in a 3D cylinder
# Doesn't matter which drug we use here
# bash $RUN_SWEEP_PATH synergy_sweep-3D-$CURRENT_DATE_TIME-control_nodrug   $SWEEP_PARAMS_TXT_PATH/sweep_3D_drugfromabove_nodrug_control.txt $SETTINGS_PATH_3D_DRUG_ADDITION/settings_AGSv2_3D_nodrug.xml AKT_MEK FINAL_NUMBER_OF_ALIVE_CELLS

# 3D Positive control: Drug diffusion coefficients for 3D experiments with only one drug (6, 60, 600 6000 for each drugs)
# each drug of the pi3k-mek synergy pair has its own best SD experiment
# bash $RUN_SWEEP_PATH synergy_sweep-pi3k_mek-3D-$CURRENT_DATE_TIME-logscale_singledrug_pi3k     $SWEEP_PARAMS_TXT_PATH/logscale_singledrug_6_to_6000.txt $SETTINGS_PATH_3D_DRUG_ADDITION/settings_AGSv2_3D_SYN_PI3K_MEK_BESTSD_PI3K.xml PI3K FINAL_NUMBER_OF_ALIVE_CELLS
# bash $RUN_SWEEP_PATH synergy_sweep-pi3k_mek-3D-$CURRENT_DATE_TIME-logscale_singledrug_mek     $SWEEP_PARAMS_TXT_PATH/logscale_singledrug_6_to_6000.txt $SETTINGS_PATH_3D_DRUG_ADDITION/settings_AGSv2_3D_SYN_PI3K_MEK_BESTSD_MEK.xml MEK FINAL_NUMBER_OF_ALIVE_CELLS
# for the akt-mek synergy pair
# bash $RUN_SWEEP_PATH synergy_sweep-akt_mek-3D-$CURRENT_DATE_TIME-logscale_singledrug_akt     $SWEEP_PARAMS_TXT_PATH/logscale_singledrug_6_to_6000.txt $SETTINGS_PATH_3D_DRUG_ADDITION/settings_AGSv2_3D_SYN_AKT_MEK_BESTSD_AKT.xml AKT FINAL_NUMBER_OF_ALIVE_CELLS
# bash $RUN_SWEEP_PATH synergy_sweep-akt_mek-3D-$CURRENT_DATE_TIME-logscale_singledrug_mek     $SWEEP_PARAMS_TXT_PATH/logscale_singledrug_6_to_6000.txt $SETTINGS_PATH_3D_DRUG_ADDITION/settings_AGSv2_3D_SYN_AKT_MEK_BESTSD_MEK.xml MEK FINAL_NUMBER_OF_ALIVE_CELLS


# 3D drug addition Drug diffusion coefficient parameter sweep (2p)
# bash $RUN_SWEEP_PATH synergy_sweep-akt_mek-$CURRENT_DATE_TIME-2p_3D_singledrugparams  $SWEEP_PARAMS_TXT_PATH/sweep_2p_3D_drugaddition_logscale_0.txt $SETTINGS_PATH_3D_DRUG_ADDITION/settings_AGSv2_3D_SYN_AKT_MEK_drugfromabove_top1p_average_singledrug.xml AKT_MEK FINAL_NUMBER_OF_ALIVE_CELLS
# bash $RUN_SWEEP_PATH synergy_sweep-pi3k_mek-$CURRENT_DATE_TIME-2p_3D_singledrugparams   $SWEEP_PARAMS_TXT_PATH/sweep_2p_3D_drugaddition_logscale_0.txt $SETTINGS_PATH_3D_DRUG_ADDITION/settings_AGSv2_3D_SYN_PI3K_MEK_drugfromabove_top1p_average_singledrug.xml PI3K_MEK FINAL_NUMBER_OF_ALIVE_CELLS

# 3D drug addition Drug diffusion coefficient parameter sweep  + drug timing (4p)
# bash $RUN_SWEEP_PATH synergy_sweep-akt_mek-$CURRENT_DATE_TIME-4p_3D_drugtiming   $SWEEP_PARAMS_TXT_PATH/sweep_4p_3D_drugaddition_drugtiming_logscale_0.txt $SETTINGS_PATH_3D_DRUG_ADDITION/settings_AGSv2_3D_SYN_AKT_MEK_drugfromabove_top1p_average_singledrug.xml AKT_MEK FINAL_NUMBER_OF_ALIVE_CELLS
# bash $RUN_SWEEP_PATH synergy_sweep-pi3k_mek-$CURRENT_DATE_TIME-4p_3D_drugtiming   $SWEEP_PARAMS_TXT_PATH/sweep_4p_3D_drugaddition_drugtiming_logscale_0.txt $SETTINGS_PATH_3D_DRUG_ADDITION/settings_AGSv2_3D_SYN_PI3K_MEK_drugfromabove_top1p_average_singledrug.xml PI3K_MEK FINAL_NUMBER_OF_ALIVE_CELLS


#deberes jestragu
# 1 full node per sim (112 cores)
# bash $RUN_SWEEP_PATH sweep-56_cores_test   $SWEEP_PARAMS_TXT_PATH/sweep_2p_control.txt $SETTINGS_PATH_CONTROL/settings_AGSv2_CONTROL.xml WT RMSE

# half node per sim (56 cores)
