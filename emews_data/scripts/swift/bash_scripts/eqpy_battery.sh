
DEAP_PARAMS_PATH='./data/JSON/deap'
SETTINGS_PATH_CONTROL='./data/physiboss_config/control'
SETTINGS_PATH_SINGLE_DRUG='./data/physiboss_config/single_drug'
SETTINGS_PATH_SINGLE_DRUG_LINEAR_MAPPING='./data/physiboss_config/single_drug_linear_mapping'
SETTINGS_PATH_SINGLE_DRUG_TRANSIENT='./data/physiboss_config/single_drug_transient'
RUN_EQPY_PATH='./swift/run_eqpy_ags.sh'
SWIFT_FILE='swift_run_eqpy.swift'
CURRENT_DATE_TIME=$(date +"%d%m-%H%M")

# Re-calibrating control 
# bash $RUN_EQPY_PATH CTRL_CMA-$CURRENT_DATE_TIME-5p $DEAP_PARAMS_PATH/deap_5p_control.json CMA RMSE_SK WT $SWIFT_FILE $SETTINGS_PATH_CONTROL/settings_AGSv2_CONTROL.xml
# bash $RUN_EQPY_PATH CTRL_GA-$CURRENT_DATE_TIME-5p $DEAP_PARAMS_PATH/deap_5p_control.json GA RMSE_SK WT $SWIFT_FILE $SETTINGS_PATH_CONTROL/settings_AGSv2_CONTROL.xml

# NEW CALIBRATIONS
bash $RUN_EQPY_PATH PI3Ki_CMA-$CURRENT_DATE_TIME-18p_delayed_transient_rmse_postdrug_25gen $DEAP_PARAMS_PATH/deap_18p_single_drug_exp_v2.json CMA RMSE_SK_POSTDRUG PI3K $SWIFT_FILE $SETTINGS_PATH_SINGLE_DRUG_TRANSIENT/settings_AGSv2_PI3K_GI50_transient_delayed.xml
bash $RUN_EQPY_PATH PI3Ki_GA-$CURRENT_DATE_TIME-18p_delayed_transient_rmse_postdrug_25gen $DEAP_PARAMS_PATH/deap_18p_single_drug_exp_v2.json GA RMSE_SK_POSTDRUG PI3K $SWIFT_FILE $SETTINGS_PATH_SINGLE_DRUG_TRANSIENT/settings_AGSv2_PI3K_GI50_transient_delayed.xml
bash $RUN_EQPY_PATH MEKi_CMA-$CURRENT_DATE_TIME-18p_delayed_transient_rmse_postdrug_25gen $DEAP_PARAMS_PATH/deap_18p_single_drug_exp_v2.json CMA RMSE_SK_POSTDRUG MEK $SWIFT_FILE $SETTINGS_PATH_SINGLE_DRUG_TRANSIENT/settings_AGSv2_MEK_GI50_transient_delayed.xml
bash $RUN_EQPY_PATH MEKi_GA-$CURRENT_DATE_TIME-18p_delayed_transient_rmse_postdrug_25gen $DEAP_PARAMS_PATH/deap_18p_single_drug_exp_v2.json GA RMSE_SK_POSTDRUG MEK $SWIFT_FILE $SETTINGS_PATH_SINGLE_DRUG_TRANSIENT/settings_AGSv2_MEK_GI50_transient_delayed.xml
bash $RUN_EQPY_PATH AKTi_CMA-$CURRENT_DATE_TIME-18p_delayed_transient_rmse_postdrug_25gen $DEAP_PARAMS_PATH/deap_18p_single_drug_exp_v2.json CMA RMSE_SK_POSTDRUG AKT $SWIFT_FILE $SETTINGS_PATH_SINGLE_DRUG_TRANSIENT/settings_AGSv2_AKT_GI50_transient_delayed.xml
bash $RUN_EQPY_PATH AKTi_GA-$CURRENT_DATE_TIME-18p_delayed_transient_rmse_postdrug_25gen $DEAP_PARAMS_PATH/deap_18p_single_drug_exp_v2.json GA RMSE_SK_POSTDRUG AKT $SWIFT_FILE $SETTINGS_PATH_SINGLE_DRUG_TRANSIENT/settings_AGSv2_AKT_GI50_transient_delayed.xml
