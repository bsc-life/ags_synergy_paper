/*
 * ags_boolean_model_interface.cpp
 */

#include "../core/PhysiCell.h"
#include "../modules/PhysiCell_standard_modules.h" 
#include "./drug_transport_model.h"

using namespace BioFVM; 
using namespace PhysiCell;

#include "./submodel_data_structures.h" 

void boolean_model_interface_setup();
void update_boolean_model_inputs( Cell* pCell, Phenotype& phenotype, double dt );
void update_cell_from_boolean_model(Cell* pCell, Phenotype& phenotype, double dt);
void update_cell_from_boolean_model_response_rates(Cell* pCell, Phenotype& phenotype, double dt);
double obtain_Hill_apoptosis_rate_from_boolean_model(Cell* pCell, Phenotype& phenotype);
double obtain_Hill_growth_rate_from_boolean_model(Cell* pCell, Phenotype& phenotype);
double obtain_linear_growth_rate_from_boolean_model(Cell* pCell, Phenotype& phenotype);
double obtain_linear_apoptosis_rate_from_boolean_model(Cell* pCell, Phenotype& phenotype);
bool check_lethal_drug_concentration(Cell* pCell, Phenotype& phenotype, std::string drug_name);
double calculate_drug_effect_with_binding(Cell* pCell, std::string drug_name, double dt);

void ags_bm_interface_main(Cell* pCell, Phenotype& phenotype, double dt); 

void print_maboss_params(Cell* pCell);
void update_maboss_params(Cell* pCell, std::string drug_name);
void restart_original_maboss_params(Cell* pCell, std::string target_node);
// void update_maboss_params(Cell* pCell, Phenotype& phenotype, std::string drug_name);
void update_maboss_params_v2(Cell* pCell);

void pre_update_intracellular_ags(Cell* pCell, Phenotype& phenotype, double dt);
void post_update_intracellular_ags(Cell* pCell, Phenotype& phenotype, double dt);

std::string get_drug_target(std::string drug_name);
double get_mapping_type(Cell* pCell, std::string drug_name);
double get_growth_mapping_type(Cell* pCell);
double get_apoptosis_mapping_type(Cell* pCell);

double get_boolean_prosurvival_outputs(Cell* pCell, Phenotype& phenotype);
double get_boolean_antisurvival_outputs(Cell* pCell, Phenotype& phenotype);

