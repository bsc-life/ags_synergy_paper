#include "./boolean_model_interface.h"
#include "../addons/PhysiBoSS/src/maboss_intracellular.h"  // Add this line
#include <math.h>

using namespace PhysiCell; 

Submodel_Information bm_interface_info;

// Auxiliar functions
std::string get_drug_target(std::string drug_name){
    std::string param_name = drug_name + "_target";
    std::string drug_target = parameters.strings(param_name);
    return drug_target;
}

void boolean_model_interface_setup()
{
    bm_interface_info.name = "AGS Boolean model interface"; 
	bm_interface_info.version = "0.0.2";
	
    bm_interface_info.main_function = ags_bm_interface_main; 

	// These are just auxiliary variables to keep track of some BN nodes

    bm_interface_info.cell_variables.push_back( "mek_node" );
    bm_interface_info.cell_variables.push_back( "pi3k_node" );
    bm_interface_info.cell_variables.push_back( "tak1_node" );
    bm_interface_info.cell_variables.push_back( "akt_node" );
    bm_interface_info.cell_variables.push_back( "anti_mek_node" );
    bm_interface_info.cell_variables.push_back( "anti_pi3k_node" );
    bm_interface_info.cell_variables.push_back( "anti_tak1_node" );
    bm_interface_info.cell_variables.push_back( "anti_akt_node" );

    // apoptosis nodes
    bm_interface_info.cell_variables.push_back( "node_FOXO" );
    bm_interface_info.cell_variables.push_back( "node_Caspase8" );
    bm_interface_info.cell_variables.push_back( "node_Caspase9" );
    // extended apoptosis nodes
    bm_interface_info.cell_variables.push_back( "node_BCL2" );
    bm_interface_info.cell_variables.push_back( "node_p53" );
    bm_interface_info.cell_variables.push_back( "node_BAD" );
    bm_interface_info.cell_variables.push_back( "node_BAX" );
    bm_interface_info.cell_variables.push_back( "node_CytochromeC" );
    // the weighted sum of the apoptosis nodes - not in the XML file
    bm_interface_info.cell_variables.push_back( "S_anti_real" );
    bm_interface_info.cell_variables.push_back( "S_anti_extended" );
    // apoptosis rate
    bm_interface_info.cell_variables.push_back( "apoptosis_rate" );

    // growth mapping nodes
    bm_interface_info.cell_variables.push_back( "node_RSK" );
    bm_interface_info.cell_variables.push_back( "node_TCF" );
    bm_interface_info.cell_variables.push_back( "node_cMYC" );
    // extended growth nodes
    bm_interface_info.cell_variables.push_back( "node_S6K" );
    bm_interface_info.cell_variables.push_back( "node_PDK1" );
    bm_interface_info.cell_variables.push_back( "node_mTORC1" );
    bm_interface_info.cell_variables.push_back( "node_LEF" );
    // the weighted sum of the growth nodes - not in the XML file
    bm_interface_info.cell_variables.push_back( "S_pro_real" );
    bm_interface_info.cell_variables.push_back( "S_pro_extended" );     
    // growth rate
    bm_interface_info.cell_variables.push_back( "growth_rate" );

    // maboss model parameters
    bm_interface_info.cell_variables.push_back( "intracellular_dt" );
    bm_interface_info.cell_variables.push_back( "scaling" );
    bm_interface_info.cell_variables.push_back( "needs_network_reset_drug_X" );
    bm_interface_info.cell_variables.push_back( "needs_network_reset_drug_Y" );
    bm_interface_info.cell_variables.push_back( "needs_network_reset" );
    bm_interface_info.cell_variables.push_back( "network_reset_counter_drug_X" );
    bm_interface_info.cell_variables.push_back( "network_reset_counter_drug_Y" );

    // parameters for the delayed response rates
    bm_interface_info.cell_variables.push_back( "response_rate_apoptosis" );
    bm_interface_info.cell_variables.push_back( "response_rate_growth" );
    bm_interface_info.cell_variables.push_back( "drug_X_node_rate_scaling" );
    bm_interface_info.cell_variables.push_back( "drug_X_node_rate_threshold" );
    bm_interface_info.cell_variables.push_back( "drug_X_node_rate" );
    // drug-target kinetic binding parameters
    bm_interface_info.cell_variables.push_back( "drug_X_kon" );
    bm_interface_info.cell_variables.push_back( "drug_X_koff" );
    bm_interface_info.cell_variables.push_back( "drug_X_target_conc" );
    bm_interface_info.cell_variables.push_back( "drug_X_DT_conc" );
    // drug-target kinetic binding parameters
    bm_interface_info.cell_variables.push_back( "drug_Y_kon" );
    bm_interface_info.cell_variables.push_back( "drug_Y_koff" );
    bm_interface_info.cell_variables.push_back( "drug_Y_target_conc" );
    bm_interface_info.cell_variables.push_back( "drug_Y_DT_conc" );


    // Could add here output of transfer functions
	bm_interface_info.register_model();
}

// @oth: This is not really needed, except for the setup function above
void ags_bm_interface_main (Cell* pCell, Phenotype& phenotype, double dt){
    
	if( phenotype.death.dead == true )
	{
		pCell->functions.update_phenotype = NULL;
		return;
	}
}


// @othmane: New PhysiBoSS, this might not be even needed, can be tracked through the BM states CSV in the output folder
void update_monitor_variables(Cell* pCell ) 
{

    // Drug target nodes
	static int mek_node_ix = pCell->custom_data.find_variable_index("mek_node");
	static int akt_node_ix = pCell->custom_data.find_variable_index("akt_node");
	static int pi3k_node_ix = pCell->custom_data.find_variable_index("pi3k_node");
	static int tak1_node_ix = pCell->custom_data.find_variable_index("tak1_node");
    static int anti_mek_node_ix = pCell->custom_data.find_variable_index("anti_mek_node");
    static int anti_akt_node_ix = pCell->custom_data.find_variable_index("anti_akt_node");
    static int anti_pi3k_node_ix = pCell->custom_data.find_variable_index("anti_pi3k_node");
    static int anti_tak1_node_ix = pCell->custom_data.find_variable_index("anti_tak1_node");
    pCell->custom_data[mek_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "MEK" );
    pCell->custom_data[akt_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "AKT" );
    pCell->custom_data[pi3k_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "PI3K" );
    pCell->custom_data[tak1_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "TAK1" );
    pCell->custom_data[anti_mek_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "anti_MEK" );
    pCell->custom_data[anti_akt_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "anti_AKT" );
    pCell->custom_data[anti_pi3k_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "anti_PI3K" );
    pCell->custom_data[anti_tak1_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "anti_TAK1" );

    // apoptosis nodes
    static int foxo_node_ix = pCell->custom_data.find_variable_index("node_FOXO");
    static int casp8_node_ix = pCell->custom_data.find_variable_index("node_Caspase8");
    static int casp9_node_ix = pCell->custom_data.find_variable_index("node_Caspase9");
    pCell->custom_data[foxo_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "FOXO" );
    pCell->custom_data[casp8_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "Caspase8" );
    pCell->custom_data[casp9_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "Caspase9" );
    // extended apoptosis nodes
    static int bcl2_node_ix = pCell->custom_data.find_variable_index("node_BCL2");
    static int p53_node_ix = pCell->custom_data.find_variable_index("node_p53");
    static int bad_node_ix = pCell->custom_data.find_variable_index("node_BAD");
    static int bax_node_ix = pCell->custom_data.find_variable_index("node_BAX");
    static int cytochromeC_node_ix = pCell->custom_data.find_variable_index("node_CytochromeC");
    pCell->custom_data[bcl2_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "BCL2" );
    pCell->custom_data[p53_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "p53" );
    pCell->custom_data[bad_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "BAD" );
    pCell->custom_data[bax_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "BAX" );
    pCell->custom_data[cytochromeC_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "CytochromeC" );

    // growth mapping nodes
    static int rsk_node_ix = pCell->custom_data.find_variable_index("node_RSK");
    static int tcf_node_ix = pCell->custom_data.find_variable_index("node_TCF");
    static int cmyc_node_ix = pCell->custom_data.find_variable_index("node_cMYC");
    pCell->custom_data[rsk_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "RSK" );
    pCell->custom_data[tcf_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "TCF" );
    pCell->custom_data[cmyc_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "cMYC" );
    // extended growth nodes
    static int s6k_node_ix = pCell->custom_data.find_variable_index("node_S6K");
    static int pdk1_node_ix = pCell->custom_data.find_variable_index("node_PDK1");
    static int mtorc1_node_ix = pCell->custom_data.find_variable_index("node_mTORC1");
    static int lef1_node_ix = pCell->custom_data.find_variable_index("node_LEF");
    pCell->custom_data[s6k_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "S6K" );
    pCell->custom_data[pdk1_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "PDK1" );
    pCell->custom_data[mtorc1_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "mTORC1" );
    pCell->custom_data[lef1_node_ix] = pCell->phenotype.intracellular->get_boolean_variable_value( "LEF" );


    return;
}


// Define a map for each drug and its GI50 (in mM)
std::map<std::string, double> drug_gi50_map = {
    {"PI3K", 0.683e-03},
    {"MEK", 31e-06},
    {"AKT", 0.01}
};

bool check_lethal_drug_concentration(Cell* pCell, Phenotype& phenotype, std::string drug_name) {
    // Get drug concentration
    static int drug_idx = microenvironment.find_density_index(drug_name);
    
    // Calculate internal drug concentration
    double cell_volume = pCell->phenotype.volume.total;
    double ic_drug_total = pCell->phenotype.molecular.internalized_total_substrates[drug_idx];
    double ic_drug_conc = ic_drug_total / cell_volume;
    
    // Get parameters
    // To get the concentration - we use the target node of the drug
    std::string target_node = get_drug_target(drug_name);
    // Take out the "anti_" from the target node
    std::string target_node_clean = target_node.substr(5);
    double gi50_conc = drug_gi50_map[target_node_clean];
    double apoptosis_threshold = 5.0 * gi50_conc;  // Changed from 2x to 5x
    double necrosis_threshold = 10.0 * gi50_conc;  // New threshold for necrosis

    // std::cout << "Target node: " << target_node_clean << std::endl;
    // std::cout << "GI50 concentration: " << gi50_conc << std::endl;
    // std::cout << "Apoptosis threshold: " << apoptosis_threshold << std::endl;
    // std::cout << "Necrosis threshold: " << necrosis_threshold << std::endl;

    static int apoptosis_model_index = phenotype.death.find_death_model_index("Apoptosis");
    static int necrosis_model_index = phenotype.death.find_death_model_index("Necrosis");

    double max_apoptosis_rate = get_custom_data_variable(pCell, "max_apoptosis_rate");
    double max_necrosis_rate = 10E12; // Necrosis rate is in 1/min - A really high number indicates faster necrosis
    double lethal_apoptosis_rate = 2.0 * max_apoptosis_rate;


    if (ic_drug_conc >= necrosis_threshold) {
        phenotype.death.rates[necrosis_model_index] = max_necrosis_rate; // Go into necrosis immediately
        return true;
    } // Or if its between the apoptosis and necrosis threshold - Here we can use a linear interpolation for the necrosis rate
    else if ((ic_drug_conc >= apoptosis_threshold) && (ic_drug_conc < necrosis_threshold)) {
        // Linear interpolation between 0.0 and maximum_necrosis_rate
        double fraction = (ic_drug_conc - gi50_conc) / (necrosis_threshold - gi50_conc);
        double interpolated_rate = max_necrosis_rate + 
            fraction * (max_necrosis_rate - 0.0);
        phenotype.death.rates[necrosis_model_index] = interpolated_rate;
        // For apoptosis, we set the maximum apoptosis rate
        phenotype.death.rates[apoptosis_model_index] = lethal_apoptosis_rate;
        return true;
    } 
    else if (ic_drug_conc > gi50_conc) {
        // Linear interpolation between max_apoptosis_rate and lethal_apoptosis_rate
        double fraction = (ic_drug_conc - gi50_conc) / (apoptosis_threshold - gi50_conc);
        double interpolated_rate = max_apoptosis_rate + 
            fraction * (lethal_apoptosis_rate - max_apoptosis_rate);
        phenotype.death.rates[apoptosis_model_index] = interpolated_rate;
        return true;
    }
    return false;  // No modification to apoptosis or necrosis rate
}





// // Functions used to update the Boolean model just before running MaBoSS
// double calculate_drug_effect_with_binding(Cell* pCell, std::string drug_name, double dt) {

//     // std::cout << "Calculating drug effect with binding for drug " << drug_name << std::endl;
//     std::string p_half_max_name   = drug_name + "_half_max";
//     std::string p_hill_coeff_name = drug_name + "_Hill_coeff";
//     std::string p_kon_name        = drug_name + "_kon";
//     std::string p_koff_name       = drug_name + "_koff";
//     std::string p_KD_name         = drug_name + "_KD";
//     std::string p_target_conc_name = drug_name + "_target_conc";
//     std::string p_DT_conc_name    = drug_name + "_DT_conc";

//     static int drug_idx          = microenvironment.find_density_index(drug_name);
//     static int p_half_max_idx    = pCell->custom_data.find_variable_index(p_half_max_name);
//     static int p_hill_coeff_idx  = pCell->custom_data.find_variable_index(p_hill_coeff_name);
//     static int p_kon_idx         = pCell->custom_data.find_variable_index(p_kon_name);
//     static int p_koff_idx        = pCell->custom_data.find_variable_index(p_koff_name);
//     static int p_target_conc_idx = pCell->custom_data.find_variable_index(p_target_conc_name);
//     static int p_DT_conc_idx     = pCell->custom_data.find_variable_index(p_DT_conc_name);
    
//     double cell_volume      = pCell->phenotype.volume.total;
//     double ic_drug_total    = pCell->phenotype.molecular.internalized_total_substrates[drug_idx];
//     double ic_drug_conc     = ic_drug_total / cell_volume; // Convert to concentration
    
//     double p_half_max       = pCell->custom_data[p_half_max_idx];
//     double p_hill_coeff     = pCell->custom_data[p_hill_coeff_idx];
//     double kon              = pCell->custom_data[p_kon_idx] * 1e6; // Scaling to compoensate for CMA
//     double koff             = pCell->custom_data[p_koff_idx];
//     double target_conc      = pCell->custom_data[p_target_conc_idx];
    
//     // Calculate Kd (dissociation constant)
//     double Kd = koff / kon;
    
//     // Calculate concentration of drug-target complex [DT] using binding equilibrium
//     // [DT] = [D][T] / (Kd + [D])
//     // Where [D] is drug concentration and [T] is total target concentration
    
//     double DT_conc = (ic_drug_conc * target_conc) / (Kd + ic_drug_conc);

//     // then map to the custom data variable
//     pCell->custom_data[p_DT_conc_idx] = DT_conc;
    
//     // Use the drug-target complex concentration to calculate effect via Hill function
//     double effect = Hill_response_function(DT_conc, p_half_max, p_hill_coeff);

//     std::cout << "Cell " << pCell->ID << " drug effect: " << effect <<  " DT_conc: " << DT_conc << std::endl;

    
//     return effect;
// }


double calculate_drug_effect_with_binding(Cell* pCell, std::string drug_name, double dt) {
    // Get drug concentration
    static int drug_idx = microenvironment.find_density_index(drug_name);
    double cell_volume = pCell->phenotype.volume.total;
    double ic_drug_total = pCell->phenotype.molecular.internalized_total_substrates[drug_idx];
    double ic_drug_conc = ic_drug_total / cell_volume;
    
    // Get binding parameters
    std::string p_kon_name = drug_name + "_kon";
    std::string p_koff_name = drug_name + "_koff";
    std::string p_target_conc_name = drug_name + "_target_conc";
    std::string p_DT_conc_name = drug_name + "_DT_conc";
    // std::string p_target_hill_coeff_name = drug_name + "_target_Hill_coeff";
    
    static int p_kon_idx = pCell->custom_data.find_variable_index(p_kon_name);
    static int p_koff_idx = pCell->custom_data.find_variable_index(p_koff_name);
    static int p_target_conc_idx = pCell->custom_data.find_variable_index(p_target_conc_name);
    static int p_DT_conc_idx = pCell->custom_data.find_variable_index(p_DT_conc_name);
    // static int p_target_hill_coeff_idx = pCell->custom_data.find_variable_index(p_target_hill_coeff_name);
    
    double kon = pCell->custom_data[p_kon_idx] * 1e6; // Scaling to compensate for CMA
    double koff = pCell->custom_data[p_koff_idx];
    double target_conc = pCell->custom_data[p_target_conc_idx];
    // double target_hill_coeff = pCell->custom_data[p_target_hill_coeff_idx];
    
    // Get current DT concentration
    double current_DT_conc = pCell->custom_data[p_DT_conc_idx];
    
    // Calculate free drug and target concentrations
    double free_drug = ic_drug_conc - current_DT_conc;
    double free_target = target_conc - current_DT_conc;
    
    // Previous approach - no saturation effect
    // ODE for drug-target binding:
    // d[DT]/dt = kon*[D]*[T] - koff*[DT]
    double dDT = (kon * free_drug * free_target - koff * current_DT_conc) * dt;

    
    // Update DT concentration
    double new_DT_conc = current_DT_conc + dDT;
    
    // Ensure concentrations stay non-negative and don't exceed total target
    new_DT_conc = std::max(0.0, new_DT_conc);
    new_DT_conc = std::min(new_DT_conc, target_conc);
    
    // Store updated DT concentration
    pCell->custom_data[p_DT_conc_idx] = new_DT_conc;
    
    // Calculate effect using Hill function
    static int p_half_max_idx = pCell->custom_data.find_variable_index(drug_name + "_half_max");
    static int p_hill_coeff_idx = pCell->custom_data.find_variable_index(drug_name + "_Hill_coeff");
    double p_half_max = pCell->custom_data[p_half_max_idx];
    double p_hill_coeff = pCell->custom_data[p_hill_coeff_idx];
    
    // Calculate base effect from binding
    double effect = Hill_response_function(new_DT_conc, p_half_max, p_hill_coeff);
    
    // Log difference between DT complex concentration and drug half max 
    // This is what is going to give us a good idea of the saturation effect
    if (effect > 0.9){
        // std::cout << "Cell " << pCell->ID << " drug effect: " << effect << " DT_conc: " << new_DT_conc << " half_max: " << p_half_max << std::endl;
        std::cout << "Cell " << pCell->ID << " DT_conc: " << new_DT_conc << " - half_max: " << p_half_max << " - Difference: " << (p_half_max - new_DT_conc) << " - Effect: " << effect << std::endl;
        // If difference is negative, then we are over the half max
        // If difference is positive, then we are under the half max  
    }
    
    return effect;
}


double calculate_drug_effect_sigmoidal(Cell* pCell, std::string drug_name){
    
	std::string p_half_max_name   = drug_name + "_half_max";
    std::string p_hill_coeff_name = drug_name + "_Hill_coeff";
    
    static int drug_idx         = microenvironment.find_density_index( drug_name );
    static int p_half_max_idx   = pCell->custom_data.find_variable_index(p_half_max_name);
    static int p_hill_coeff_idx = pCell->custom_data.find_variable_index(p_hill_coeff_name);
	
    double cell_volume   = pCell->phenotype.volume.total;
    double ic_drug_total = pCell->phenotype.molecular.internalized_total_substrates[drug_idx];
    double ic_drug_conc  = ic_drug_total / cell_volume; // Convert to concentration

    // std::cout << pCell->custom_data[p_half_max_idx] << std::endl;

    double p_half_max    = pCell->custom_data[p_half_max_idx];
    double p_hill_coeff  = pCell->custom_data[p_hill_coeff_idx];
    
    double effect = Hill_response_function(ic_drug_conc, p_half_max, p_hill_coeff);
    return effect;
}


double calculate_drug_effect_linear(Cell* pCell, std::string drug_name) {
    // Get indices for drug and its half-max parameter
    static int drug_idx = microenvironment.find_density_index(drug_name);
    static int p_half_max_idx = pCell->custom_data.find_variable_index(drug_name + "_half_max");
    
    // Calculate drug concentration inside the cell
    double cell_volume = pCell->phenotype.volume.total;
    double ic_drug_total = pCell->phenotype.molecular.internalized_total_substrates[drug_idx];
    double ic_drug_conc = ic_drug_total / cell_volume;
    
    // Get half-max value and calculate maximum effect point (2 * half_max)
    double half_max = pCell->custom_data[p_half_max_idx];
    double max_effect = 2.0 * half_max;
    
    // Linear mapping:
    // Below 0: No effect
    // Between half_max and max_effect: linear increase from 0 to 1
    // Above max_effect: maximum effect (1)

    if (ic_drug_conc <= 0) {
        return 0.0;  // No effect at zero or negative concentration
    } else if (ic_drug_conc >= max_effect) {
        return 1.0;  // Maximum effect
    } else {
        // Linear interpolation from 0 to 1
        // At half_max, effect should be 0.5
        double drug_effect = ic_drug_conc / (2 * half_max);  
        
        // Ensure effect stays between 0 and 1
        drug_effect = std::min(1.0, std::max(0.0, drug_effect));
        
        return drug_effect;
    }
}

double get_mapping_type(Cell* pCell, std::string drug_name){
    std::string mapping_type_name = drug_name + "_mapping_type";
    return get_custom_data_variable(pCell, mapping_type_name);
}

double get_growth_mapping_type(Cell* pCell){
    std::string mapping_type_name = "growth_mapping_type";
    return get_custom_data_variable(pCell, mapping_type_name);
}

double get_apoptosis_mapping_type(Cell* pCell){
    std::string mapping_type_name = "apoptosis_mapping_type";
    return get_custom_data_variable(pCell, mapping_type_name);
}

void update_boolean_model_inputs_smooth(Cell* pCell, Phenotype& phenotype, double dt) {
    if(pCell->phenotype.death.dead == true)
        return;

    int n_drugs = 2;
    std::string drugs[n_drugs] = { "drug_X", "drug_Y" };
    double new_intracellular_dt = parameters.doubles("new_intracellular_dt");
    double new_scaling = parameters.doubles("new_scaling");
    static int needs_network_reset_ix = pCell->custom_data.find_variable_index("needs_network_reset");
    pCell->custom_data[needs_network_reset_ix] = 0;
    
    for (int i = 0; i < n_drugs; i++) {
        std::string drug_name = drugs[i];
        std::string target_node = get_drug_target(drug_name);

        if (target_node == "null")
            continue;

        double drug_effect = (get_mapping_type(pCell, drug_name) == 1) ? 
            calculate_drug_effect_linear(pCell, drug_name) :
            calculate_drug_effect_with_binding(pCell, drug_name, dt);

        // Apply the drug effect to gradually increase activation rate
        if (drug_effect > 0) {
            // std::cout << "Cell " << pCell->ID << " drug_effect: " << drug_effect << " target_node: " << target_node << std::endl;
            
            // Gillespie-like stochastic behavior
            if (uniform_random() < drug_effect) {
                // Get current state of target node
                bool current_state = pCell->phenotype.intracellular->get_boolean_variable_value(target_node);
                
                // Only update if the state needs to change
                if (current_state == 0) {
                    // Update network parameters and set target node
                    if (pCell->custom_data[needs_network_reset_ix] == 0){ 
                        std::cout << "Cell " << pCell->ID << " updating network parameters for target node " << target_node << " with effect " << drug_effect << std::endl;
                        update_maboss_params(pCell, target_node);
                        pCell->custom_data[needs_network_reset_ix] = 1;                    
                    } else {
                        // if the gillespie was OK, the node was OFF, but another drug already reset the network
                        pCell->phenotype.intracellular->set_boolean_variable_value(target_node, 1);  
                    }
                }

                if (current_state == 1){
                    // TO-DO: ADD HYSTERESIS HERE
                    // BN goes back to the original state and runs again 
                    if (pCell->custom_data[needs_network_reset_ix] == 1){
                        restart_original_maboss_params(pCell, target_node);
                        std::cout << "Cell " << pCell->ID << " goes back to the original state and runs again " << std::endl;
                        pCell->custom_data[needs_network_reset_ix] = 0;
                    } else {
                        // if the gillespie was OK, the node was ON, but another drug already reset the network
                        pCell->phenotype.intracellular->set_boolean_variable_value(target_node, 0);
                    }
                }
            }
        }
    }
}

// void update_boolean_model_inputs_smooth(Cell* pCell, Phenotype& phenotype, double dt) {
//     if(pCell->phenotype.death.dead == true)
//         return;

//     int n_drugs = 2;
//     std::string drugs[n_drugs] = { "drug_X", "drug_Y" };
//     double new_intracellular_dt = parameters.doubles("new_intracellular_dt");
//     double new_scaling = parameters.doubles("new_scaling");
//     static int needs_network_reset_ix = pCell->custom_data.find_variable_index("needs_network_reset");
    
//     for (int i = 0; i < n_drugs; i++) {
//         std::string drug_name = drugs[i];
//         std::string target_node = get_drug_target(drug_name);

//         if (target_node == "null")
//             continue;

//         double drug_effect = (get_mapping_type(pCell, drug_name) == 1) ? 
//             calculate_drug_effect_linear(pCell, drug_name) :
//             calculate_drug_effect_with_binding(pCell, drug_name, dt);

//         // Apply the drug effect to gradually increase activation rate
//         if (drug_effect > 0) {
//             // std::cout << "Cell " << pCell->ID << " drug_effect: " << drug_effect << " target_node: " << target_node << std::endl;
            
//             // Gillespie-like stochastic behavior
//             if (uniform_random() < drug_effect) {
//                 // Get current state of target node
//                 bool current_state = pCell->phenotype.intracellular->get_boolean_variable_value(target_node);
                
//                 // Only update if the state needs to change
//                 if (current_state == 0) {
//                     // Update network parameters and set target node
//                     std::cout << "Cell " << pCell->ID << " updating network parameters for target node " << target_node << " with effect " << drug_effect << std::endl;
//                     update_maboss_params(pCell, target_node);
//                     pCell->custom_data[needs_network_reset_ix] = 1;                    
//                 }

//                 if (current_state == 1){
//                     // TO-DO: ADD HYSTERESIS HERE
//                     // BN goes back to the original state and runs again 
//                     restart_original_maboss_params(pCell, target_node);
//                     std::cout << "Cell " << pCell->ID << " goes back to the original state and runs again " << std::endl;
//                 }
//             }
//         }
//     }
// }




// PREVIOUS VERSION OF THE FUNCTION (THIS WORKED)
// void update_boolean_model_inputs( Cell* pCell, Phenotype& phenotype, double dt )
// {
//     if( pCell->phenotype.death.dead == true )
// 	{ return; }

//     int n_drugs = 2;
//     std::string drugs[n_drugs] = { "drug_X", "drug_Y" };
//     double new_intracellular_dt = parameters.doubles("new_intracellular_dt");
//     double new_scaling = parameters.doubles("new_scaling");
  
//     for (int i = 0; i < n_drugs; i++){
//         std::string drug_name = drugs[i];
//         std::string target_node = get_drug_target(drug_name);
        

//         // Single drug cases, drug_Y is not used so it's set to "null" in the config file
//         if (target_node == "null") {
//             return; 
//         }

//         double drug_effect;

//         if (get_mapping_type(pCell, drug_name) == 1){
//             // std::cout << "Linear mapping for drug " << drug_name << std::endl;
//             drug_effect = calculate_drug_effect_linear(pCell, drug_name);
//         } else if (get_mapping_type(pCell, drug_name) == 0){
//             // std::cout << "Sigmoidal mapping for drug " << drug_name << std::endl;
//             drug_effect = calculate_drug_effect_sigmoidal(pCell, drug_name);
//         } else {
//             std::cout << "Error: Invalid mapping type for drug " << drug_name << std::endl;
//             return;
//         }

//         // Apply the drug effect to the target node
//         if (drug_effect > 0){ 
//             // Apply Gillespie only for the target_node obtained
//             if (uniform_random() < drug_effect && pCell->custom_data[needs_network_reset_ix] == 0){
//                 update_maboss_params(pCell, target_node);

//                 if (pCell->custom_data[needs_network_reset_ix] == 0){
//                     pCell->custom_data[needs_network_reset_ix] = 1;
//                     std::cout << "for cell " << pCell->ID << " network state changed" << std::endl;
//                 }
//             }
//         } else {
//             // Reset the network state for this drug's target
//             pCell->phenotype.intracellular->set_boolean_variable_value(target_node, 0);
//             // Reset the activation level
//             static int activation_level_idx = pCell->custom_data.find_variable_index(drug_name + "_activation_level");
//             pCell->custom_data[activation_level_idx] = 0.0;
//         }
//     }

//     return;
// }



void update_maboss_params(Cell* pCell, std::string target_node){
    double new_intracellular_dt = parameters.doubles("new_intracellular_dt");
    double new_scaling = parameters.doubles("new_scaling");
    auto* maboss_model = static_cast<MaBoSSIntracellular*>(pCell->phenotype.intracellular);

    if (maboss_model) { 
        // Update both network and class parameters
        maboss_model->maboss.set_update_time_step(new_intracellular_dt);
        maboss_model->time_step = new_intracellular_dt;
        maboss_model->maboss.set_scaling(new_scaling);
        maboss_model->scaling = new_scaling;

        // and restart the model
        // maboss_model->maboss.restart_node_values();

        pCell->phenotype.intracellular->set_boolean_variable_value(target_node, 1);  
        // std::cout << "target node " << target_node << " set to 1 for cell " << pCell->ID << std::endl;

        maboss_model->start();
    }

}

void restart_original_maboss_params(Cell* pCell, std::string target_node){
    double original_intracellular_dt = 10;
    double original_scaling = 1;
    auto* maboss_model = static_cast<MaBoSSIntracellular*>(pCell->phenotype.intracellular);

    if (maboss_model) { 
        // Update both network and class parameters
        maboss_model->maboss.set_update_time_step(original_intracellular_dt);
        maboss_model->time_step = original_intracellular_dt;
        maboss_model->maboss.set_scaling(original_scaling);
        maboss_model->scaling = original_scaling;

        // and restart the model
        // maboss_model->maboss.restart_node_values();

        pCell->phenotype.intracellular->set_boolean_variable_value(target_node, 0);  
        // std::cout << "target node " << target_node << " set to 1 for cell " << pCell->ID << std::endl;

        maboss_model->start();
    }
}




void update_maboss_params_v2(Cell* pCell){
    double new_intracellular_dt = parameters.doubles("new_intracellular_dt");
    
    // Get current scaling and increase it
    static int current_scaling_idx = pCell->custom_data.find_variable_index("current_scaling");
    double current_scaling = pCell->custom_data[current_scaling_idx];
    double new_scaling = current_scaling + 10.0;  // Increase by 10
    
    // Update the cell's current scaling value
    pCell->custom_data[current_scaling_idx] = new_scaling;
    auto* maboss_model = static_cast<MaBoSSIntracellular*>(pCell->phenotype.intracellular);
    
    if (maboss_model) { 
        // Update both network and class parameters
        maboss_model->maboss.set_update_time_step(new_intracellular_dt);
        maboss_model->time_step = new_intracellular_dt;
        maboss_model->maboss.set_scaling(new_scaling);
        maboss_model->scaling = new_scaling;
        // and restart the model
        // maboss_model->maboss.restart_node_values();
        // std::cout << "Cell " << pCell->ID << " scaling updated to: " << new_scaling << std::endl;
    }
}



void print_maboss_params(Cell* pCell) {
    auto* maboss_model = static_cast<MaBoSSIntracellular*>(pCell->phenotype.intracellular);
    if (maboss_model) {
        static int intracellular_dt_ix = pCell->custom_data.find_variable_index("intracellular_dt");
        pCell->custom_data[intracellular_dt_ix] = maboss_model->time_step;
        static int scaling_ix = pCell->custom_data.find_variable_index("scaling");
        pCell->custom_data[scaling_ix] = maboss_model->scaling;
    }
}

void pre_update_intracellular_ags(Cell* pCell, Phenotype& phenotype, double dt)
{
    if( phenotype.death.dead == true )
	{
		pCell->functions.update_phenotype = NULL;
		return;
	}

    if (check_lethal_drug_concentration(pCell, phenotype, "drug_X")) {
        return;
    }

    // Update MaBoSS input nodes based on the environment and cell state
    // update_boolean_model_inputs(pCell, phenotype, dt);
    update_boolean_model_inputs_smooth(pCell, phenotype, dt);
    // This function can be use for the reactivation mechanisms
    // pathway_reactivation( Cell* pCell, Phenotype& phenotype, double dt )

    print_maboss_params(pCell);

    
    return;
}


// @oth: added these functions to compute the readout nodes from the BM
double get_boolean_antisurvival_outputs(Cell* pCell, Phenotype& phenotype){

    // Antisurvival node outputs
    // bool casp37 = pCell->phenotype.intracellular->get_boolean_variable_value( "Caspase37" );
    bool FOXO = pCell->phenotype.intracellular->get_boolean_variable_value( "FOXO" );
    bool casp8 = pCell->phenotype.intracellular->get_boolean_variable_value( "Caspase8" );
    bool casp9 = pCell->phenotype.intracellular->get_boolean_variable_value( "Caspase9" );

    double w_anti_FOXO = get_custom_data_variable(pCell, "w_anti_FOXO");
    double w_anti_Caspase8 = get_custom_data_variable(pCell, "w_anti_Caspase8");
    double w_anti_Caspase9 = get_custom_data_variable(pCell, "w_anti_Caspase9");
    double total_anti = w_anti_FOXO + w_anti_Caspase8 + w_anti_Caspase9;
    double w_anti_FOXO_scaled = w_anti_FOXO / total_anti;
    double w_anti_Caspase8_scaled = w_anti_Caspase8 / total_anti;
    double w_anti_Caspase9_scaled = w_anti_Caspase9 / total_anti;

    double S_anti_real = (w_anti_FOXO_scaled*FOXO) + (w_anti_Caspase8_scaled * casp8) + (w_anti_Caspase9_scaled * casp9);

    // then map to the custom data variable
    static int S_anti_real_ix = pCell->custom_data.find_variable_index("S_anti_real");
    pCell->custom_data[S_anti_real_ix] = S_anti_real;

    // For the Control case 
    if (total_anti == 0.0){
        return 0.0;
    } else {
        return S_anti_real;
    }
}

// @oth: added these functions to compute the readout nodes from the BM
double get_boolean_antisurvival_extended_outputs(Cell* pCell, Phenotype& phenotype){

    // Antisurvival node outputs
    bool FOXO = pCell->phenotype.intracellular->get_boolean_variable_value( "FOXO" );
    bool casp8 = pCell->phenotype.intracellular->get_boolean_variable_value( "Caspase8" );
    bool casp9 = pCell->phenotype.intracellular->get_boolean_variable_value( "Caspase9" );
    bool cytC = pCell->phenotype.intracellular->get_boolean_variable_value( "CytochromeC" );
    bool BAX = pCell->phenotype.intracellular->get_boolean_variable_value( "BAX" );
    bool BAD = pCell->phenotype.intracellular->get_boolean_variable_value( "BAD" );
    bool BCL2 = pCell->phenotype.intracellular->get_boolean_variable_value( "BCL2" ); // WARNING: This is negative
    bool p53 = pCell->phenotype.intracellular->get_boolean_variable_value( "p53" );

    double w_anti_FOXO = get_custom_data_variable(pCell, "w_anti_FOXO");
    double w_anti_Caspase8 = get_custom_data_variable(pCell, "w_anti_Caspase8");
    double w_anti_Caspase9 = get_custom_data_variable(pCell, "w_anti_Caspase9");
    double w_anti_CytochromeC = get_custom_data_variable(pCell, "w_anti_CytochromeC");
    double w_anti_BAX = get_custom_data_variable(pCell, "w_anti_BAX");
    double w_anti_BAD = get_custom_data_variable(pCell, "w_anti_BAD");
    double w_anti_BCL2 = get_custom_data_variable(pCell, "w_anti_BCL2");
    double w_anti_p53 = get_custom_data_variable(pCell, "w_anti_p53");

    double total_anti = w_anti_FOXO + w_anti_Caspase8 + w_anti_Caspase9 + w_anti_CytochromeC + w_anti_BAX + w_anti_BAD - w_anti_BCL2 + w_anti_p53;

    if (total_anti < 0){
        total_anti = 0;
    }

    double w_anti_FOXO_scaled = w_anti_FOXO / total_anti;
    double w_anti_Caspase8_scaled = w_anti_Caspase8 / total_anti;
    double w_anti_Caspase9_scaled = w_anti_Caspase9 / total_anti;
    double w_anti_CytochromeC_scaled = w_anti_CytochromeC / total_anti;
    double w_anti_BAX_scaled = w_anti_BAX / total_anti;
    double w_anti_BAD_scaled = w_anti_BAD / total_anti;
    double w_anti_BCL2_scaled = w_anti_BCL2 / total_anti;
    double w_anti_p53_scaled = w_anti_p53 / total_anti;

    double S_anti_extended = (w_anti_FOXO_scaled*FOXO) + (w_anti_Caspase8_scaled * casp8) + (w_anti_Caspase9_scaled * casp9) + (w_anti_CytochromeC_scaled * cytC) + (w_anti_BAX_scaled * BAX) + (w_anti_BAD_scaled * BAD) - (w_anti_BCL2_scaled * BCL2) + (w_anti_p53_scaled * p53);

    if (S_anti_extended < 0){
        S_anti_extended = 0;
    }

    if (S_anti_extended > 1){
        S_anti_extended = 1;
    }

    // then map to the custom data variable
    static int S_anti_extended_ix = pCell->custom_data.find_variable_index("S_anti_extended");
    pCell->custom_data[S_anti_extended_ix] = S_anti_extended;

    // For the Control case 
    if (total_anti == 0.0){
        return 0.0;
    } else {
        return S_anti_extended;
    }
}

double get_boolean_prosurvival_outputs(Cell* pCell, Phenotype& phenotype){

    // bool CCND1 = pCell->phenotype.intracellular->get_boolean_variable_value( "CCND1" );
    bool cMYC = pCell->phenotype.intracellular->get_boolean_variable_value( "cMYC" );
    bool TCF = pCell->phenotype.intracellular->get_boolean_variable_value( "TCF" );
    bool RSK = pCell->phenotype.intracellular->get_boolean_variable_value( "RSK" );

    double w_pro_cMYC = get_custom_data_variable(pCell, "w_pro_cMYC");
    double w_pro_TCF = get_custom_data_variable(pCell, "w_pro_TCF");
    double w_pro_RSK = get_custom_data_variable(pCell, "w_pro_RSK");
    double total_pro = w_pro_cMYC + w_pro_TCF + w_pro_RSK;
    double w_pro_cMYC_scaled = w_pro_cMYC / total_pro;
    double w_pro_TCF_scaled = w_pro_TCF / total_pro;
    double w_pro_RSK_scaled = w_pro_RSK / total_pro;

    double S_pro_real = (w_pro_cMYC_scaled * cMYC) + (w_pro_TCF_scaled * TCF) + (w_pro_RSK_scaled * RSK);

    // then map to the custom data variable
    static int S_pro_real_ix = pCell->custom_data.find_variable_index("S_pro_real");
    pCell->custom_data[S_pro_real_ix] = S_pro_real;

    // For the Control curve, when all weights are 0, the output should be 1.0 (maximum growth rate)
    // This avoids the -nan value as an input for the Hill function
    if (total_pro == 0.0){
        return 1.0;
    } else {
        return S_pro_real;
    }
}


// @oth: added these functions to compute the readout nodes from the BM
double get_boolean_prosurvival_extended_outputs(Cell* pCell, Phenotype& phenotype){

    // Antisurvival node outputs
    // bool casp37 = pCell->phenotype.intracellular->get_boolean_variable_value( "Caspase37" );
    bool cMYC = pCell->phenotype.intracellular->get_boolean_variable_value( "cMYC" );
    bool TCF = pCell->phenotype.intracellular->get_boolean_variable_value( "TCF" );
    bool RSK = pCell->phenotype.intracellular->get_boolean_variable_value( "RSK" );
    bool S6K = pCell->phenotype.intracellular->get_boolean_variable_value( "S6K" );
    bool PDK1 = pCell->phenotype.intracellular->get_boolean_variable_value( "PDK1" );
    bool mTORC1 = pCell->phenotype.intracellular->get_boolean_variable_value( "mTORC1" );
    bool LEF = pCell->phenotype.intracellular->get_boolean_variable_value( "LEF" );

    double w_pro_cMYC = get_custom_data_variable(pCell, "w_pro_cMYC");
    double w_pro_TCF = get_custom_data_variable(pCell, "w_pro_TCF");
    double w_pro_RSK = get_custom_data_variable(pCell, "w_pro_RSK");
    double w_pro_S6K = get_custom_data_variable(pCell, "w_pro_S6K");
    double w_pro_PDK1 = get_custom_data_variable(pCell, "w_pro_PDK1");
    double w_pro_mTORC1 = get_custom_data_variable(pCell, "w_pro_mTORC1");
    double w_pro_LEF = get_custom_data_variable(pCell, "w_pro_LEF");

    double total_pro = w_pro_cMYC + w_pro_TCF + w_pro_RSK + w_pro_S6K + w_pro_PDK1 + w_pro_mTORC1 + w_pro_LEF;


    if (total_pro < 0){
        total_pro = 0;
    }

    double w_pro_cMYC_scaled = w_pro_cMYC / total_pro;
    double w_pro_TCF_scaled = w_pro_TCF / total_pro;
    double w_pro_RSK_scaled = w_pro_RSK / total_pro;
    double w_pro_S6K_scaled = w_pro_S6K / total_pro;
    double w_pro_PDK1_scaled = w_pro_PDK1 / total_pro;
    double w_pro_mTORC1_scaled = w_pro_mTORC1 / total_pro;
    double w_pro_LEF_scaled = w_pro_LEF / total_pro;

    double S_pro_extended = (w_pro_cMYC_scaled * cMYC) + (w_pro_TCF_scaled * TCF) + (w_pro_RSK_scaled * RSK) + (w_pro_S6K_scaled * S6K) + (w_pro_PDK1_scaled * PDK1) + (w_pro_mTORC1_scaled * mTORC1) + (w_pro_LEF_scaled * LEF);

    if (S_pro_extended < 0){
        S_pro_extended = 0;
    }

    if (S_pro_extended > 1){
        S_pro_extended = 1;
    }

    // then map to the custom data variable
    static int S_pro_extended_ix = pCell->custom_data.find_variable_index("S_pro_extended");
    pCell->custom_data[S_pro_extended_ix] = S_pro_extended;

    // For the Control case 
    if (total_pro == 0.0){
        std::cout << "total_pro is 0" << std::endl;
        return 0.0;
    } else {
        return S_pro_extended;
    }
}


// Super simple arrest function for now, but could use other arguments
bool prosurvival_arrest_function( Cell* pCell, Phenotype& phenotype, double dt){
    return true;
}

double obtain_Hill_apoptosis_rate_from_boolean_model(Cell* pCell, Phenotype& phenotype){
    
    // Connect output from model to actual cell variables
    double apoptosis_rate_basal = get_custom_data_variable(pCell, "apoptosis_rate_basal");
    double maximum_apoptosis_rate =  get_custom_data_variable(pCell, "max_apoptosis_rate");
    double hill_coeff_apoptosis = get_custom_data_variable(pCell, "hill_coeff_apoptosis");
    double K_half_apoptosis = get_custom_data_variable(pCell, "K_half_apoptosis");

    double S_anti_real = get_boolean_antisurvival_outputs(pCell, phenotype);
    double S_anti_extended = get_boolean_antisurvival_extended_outputs(pCell, phenotype);

    // sigmoidal mapping
    double apoptosis_value_Hill = maximum_apoptosis_rate * (Hill_response_function(S_anti_real, K_half_apoptosis, hill_coeff_apoptosis));
    apoptosis_value_Hill += apoptosis_rate_basal;

    // sigmoidal mapping with extended readouts
    double apoptosis_value_Hill_extended = maximum_apoptosis_rate * (Hill_response_function(S_anti_extended, K_half_apoptosis, hill_coeff_apoptosis));
    apoptosis_value_Hill_extended += apoptosis_rate_basal;

    return apoptosis_value_Hill_extended;

}

double obtain_linear_apoptosis_rate_from_boolean_model(Cell* pCell, Phenotype& phenotype) {
    // Get basal and maximum rates
    double apoptosis_rate_basal = get_custom_data_variable(pCell, "apoptosis_rate_basal");
    double maximum_apoptosis_rate = get_custom_data_variable(pCell, "max_apoptosis_rate");
    
    // Get antisurvival output (varies between 0 and 1)
    double S_anti_real = get_boolean_antisurvival_outputs(pCell, phenotype);

    // Linear mapping between basal and maximum rates
    // When S_anti_real = 0: output = basal_rate
    // When S_anti_real = 1: output = maximum_rate
    double apoptosis_value_linear = apoptosis_rate_basal + 
        (S_anti_real * (maximum_apoptosis_rate - apoptosis_rate_basal));

    return apoptosis_value_linear;
}

double obtain_Hill_growth_rate_from_boolean_model(Cell* pCell, Phenotype& phenotype){

     // Effect on the growth rate
    double basal_growth_rate = get_custom_data_variable(pCell, "basal_growth_rate");
    double hill_coeff_growth = get_custom_data_variable(pCell, "hill_coeff_growth");
    double K_half_growth = get_custom_data_variable(pCell, "K_half_growth");

    double S_pro_real = get_boolean_prosurvival_outputs(pCell, phenotype);
    double S_pro_extended = get_boolean_prosurvival_extended_outputs(pCell, phenotype);


    // sigmoidal mapping
    double growth_value_Hill =  (Hill_response_function(S_pro_real, K_half_growth, hill_coeff_growth));
    growth_value_Hill *= basal_growth_rate; // Max value is the basal growth rate

    // sigmoidal mapping with extended readouts
    double growth_value_Hill_extended =  (Hill_response_function(S_pro_extended, K_half_growth, hill_coeff_growth));
    growth_value_Hill_extended *= basal_growth_rate; // Max value is the basal growth rate

    return growth_value_Hill_extended;
    
}

double obtain_linear_growth_rate_from_boolean_model(Cell* pCell, Phenotype& phenotype) {
    // Get basal growth rate as maximum
    double basal_growth_rate = get_custom_data_variable(pCell, "basal_growth_rate");
    
    // Get prosurvival output (varies between 0 and 1)
    double S_pro_real = get_boolean_prosurvival_outputs(pCell, phenotype);
    double S_pro_extended = get_boolean_prosurvival_extended_outputs(pCell, phenotype);

    // Linear mapping: output = input * max_value
    double growth_value_linear = S_pro_real * basal_growth_rate;
    double growth_value_linear_extended = S_pro_extended * basal_growth_rate;

    return growth_value_linear;
}



void update_cell_from_boolean_model_response_rates(Cell* pCell, Phenotype& phenotype, double dt){
   if(pCell->phenotype.death.dead == true) 
        return;

    auto* maboss_model = static_cast<MaBoSSIntracellular*>(pCell->phenotype.intracellular);
    static int apoptosis_model_index = phenotype.death.find_death_model_index("Apoptosis");
    static int apoptosis_rate_idx = pCell->custom_data.find_variable_index("apoptosis_rate");
    static int growth_rate_idx = pCell->custom_data.find_variable_index("growth_rate");
    
    // Get current rates
    double current_apoptosis_rate = pCell->custom_data[apoptosis_rate_idx];
    double current_growth_rate = pCell->custom_data[growth_rate_idx];
    
    // Calculate target rates from Boolean model
    double target_apoptosis_rate;
    if (get_apoptosis_mapping_type(pCell) == 0) {
        target_apoptosis_rate = obtain_Hill_apoptosis_rate_from_boolean_model(pCell, phenotype);
    } else {
        target_apoptosis_rate = obtain_linear_apoptosis_rate_from_boolean_model(pCell, phenotype);
    }
    
    double target_growth_rate;
    if (get_growth_mapping_type(pCell) == 0) {
        target_growth_rate = obtain_Hill_growth_rate_from_boolean_model(pCell, phenotype);
    } else {
        target_growth_rate = obtain_linear_growth_rate_from_boolean_model(pCell, phenotype);
    }
    
    // Parameters for rate of change
    double response_rate_apoptosis = get_custom_data_variable(pCell, "response_rate_apoptosis"); // Adjust this to control how quickly rates change
    double response_rate_growth = get_custom_data_variable(pCell, "response_rate_growth");


    // Gradually update rates using exponential smoothing
    double new_apoptosis_rate = current_apoptosis_rate + 
        response_rate_apoptosis * (target_apoptosis_rate - current_apoptosis_rate) * dt;
    double new_growth_rate = current_growth_rate + 
        response_rate_growth * (target_growth_rate - current_growth_rate) * dt;
    
    // Update phenotype and custom data
    pCell->phenotype.death.rates[apoptosis_model_index] = new_apoptosis_rate;
    pCell->phenotype.cycle.data.transition_rate(0, 0) = new_growth_rate;
    pCell->custom_data[apoptosis_rate_idx] = new_apoptosis_rate;
    pCell->custom_data[growth_rate_idx] = new_growth_rate;

    return;
}

void update_cell_from_boolean_model(Cell* pCell, Phenotype& phenotype, double dt){

    if( pCell->phenotype.death.dead == true )
	{ return; } 

    // Apoptosis rate effect
    static int apoptosis_model_index = phenotype.death.find_death_model_index( "Apoptosis" );
    static int necrosis_model_index = phenotype.death.find_death_model_index( "Necrosis" );

    // TESTING EXTENDED READOUTS
    double hill_apoptosis_rate = obtain_Hill_apoptosis_rate_from_boolean_model(pCell, phenotype);
    double linear_apoptosis_rate = obtain_linear_apoptosis_rate_from_boolean_model(pCell, phenotype);

    static int apoptosis_rate_idx = pCell->custom_data.find_variable_index("apoptosis_rate");
    static int growth_rate_idx = pCell->custom_data.find_variable_index("growth_rate");

    if (get_apoptosis_mapping_type(pCell) == 0){
        // std::cout << "Hill mapping for apoptosis" << std::endl;
        pCell-> phenotype.death.rates[apoptosis_model_index] = hill_apoptosis_rate;
        pCell->custom_data[apoptosis_rate_idx] = hill_apoptosis_rate;
    } else if (get_apoptosis_mapping_type(pCell) == 1){
        // std::cout << "Linear mapping for apoptosis" << std::endl;
        pCell-> phenotype.death.rates[apoptosis_model_index] = linear_apoptosis_rate;
        pCell->custom_data[apoptosis_rate_idx] = linear_apoptosis_rate;
    } else {
        std::cout << "Error: Invalid mapping type for apoptosis" << std::endl;
        return;
    }

    // Growth rate effect
    double hill_growth_rate = obtain_Hill_growth_rate_from_boolean_model(pCell, phenotype);
    double linear_growth_rate = obtain_linear_growth_rate_from_boolean_model(pCell, phenotype);
    if (get_growth_mapping_type(pCell) == 0){
        // std::cout << "Hill mapping for growth" << std::endl;
        pCell->phenotype.cycle.data.transition_rate(0, 0) = hill_growth_rate;
        pCell->custom_data[growth_rate_idx] = hill_growth_rate;

    } else if (get_growth_mapping_type(pCell) == 1){
        // std::cout << "Linear mapping for growth" << std::endl;
        pCell->phenotype.cycle.data.transition_rate(0, 0) = linear_growth_rate;
        pCell->custom_data[growth_rate_idx] = linear_growth_rate;

    } else {
        std::cout << "Error: Invalid mapping type for growth" << std::endl;
        return;
    }

    return;
}

void post_update_intracellular_ags(Cell* pCell, Phenotype& phenotype, double dt)
{
    if( phenotype.death.dead == true )
	{
		pCell->functions.update_phenotype = NULL;
		return;
	}

    if (check_lethal_drug_concentration(pCell, phenotype, "drug_X")) {
        return;
    }
    
    // update the cell fate based on the boolean outputs
    update_cell_from_boolean_model(pCell, phenotype, dt);
    // update_cell_from_boolean_model_response_rates(pCell, phenotype, dt);
    
    // Get track of some boolean node values for debugging
    // @oth: Probably not needed anymore with pcdl
    update_monitor_variables(pCell);

    return;
}


// NOT EMPLOYED (For now - could be used for implementing reactivation mechanisms for resistance emergence)
void pathway_reactivation( Cell* pCell, Phenotype& phenotype, double dt ){

    static int reactivation_prob_idx = pCell->custom_data.find_variable_index("reactivation_value");
    double p_reactivation = pCell->custom_data.variables[reactivation_prob_idx].value;
    
    int n_drugs = 2;
    std::string drugs[n_drugs] = { "drug_X", "drug_Y" };
    for (int i = 0; i < n_drugs; i++){
        std::string drug_name = drugs[i];
        std::string target_node = get_drug_target(drug_name);
        if (target_node == "none" )
            continue;
        if ( uniform_random() < p_reactivation )
            pCell->phenotype.intracellular->set_boolean_variable_value(target_node, 1);
    }
    return;
}


// void update_boolean_model_inputs_smooth(Cell* pCell, Phenotype& phenotype, double dt) {
//     if(pCell->phenotype.death.dead == true)
//         return;

//     int n_drugs = 2;
//     std::string drugs[n_drugs] = { "drug_X", "drug_Y" };
//     double new_intracellular_dt = parameters.doubles("new_intracellular_dt");
//     double new_scaling = parameters.doubles("new_scaling");
//     static int needs_network_reset_ix = pCell->custom_data.find_variable_index("needs_network_reset");
    
//     for (int i = 0; i < n_drugs; i++) {
//         std::string drug_name = drugs[i];
//         std::string target_node = get_drug_target(drug_name);

//         if (target_node == "null")
//             continue;

//         double drug_effect = (get_mapping_type(pCell, drug_name) == 1) ? 
//             calculate_drug_effect_linear(pCell, drug_name) :
//             calculate_drug_effect_with_binding(pCell, drug_name, dt);

//         // Apply the drug effect to gradually increase activation rate
//         if (drug_effect > 0) {
//             // Get current state of target node before any changes
//             bool current_state = pCell->phenotype.intracellular->get_boolean_variable_value(target_node);
            
//             // Gillespie-like stochastic behavior
//             if (uniform_random() < drug_effect) {
//                 // Only update if the state needs to change
//                 if (current_state == 0) {
//                     std::cout << "Cell " << pCell->ID << " attempting to activate " << target_node 
//                               << " with drug_effect: " << drug_effect << std::endl;
                    
//                     // Print the state of key nodes before update
//                     std::cout << "Before update - Key nodes state:" << std::endl;
//                     std::cout << "  - PI3K: " << pCell->phenotype.intracellular->get_boolean_variable_value("PI3K") << std::endl;
//                     std::cout << "  - anti_PI3K: " << pCell->phenotype.intracellular->get_boolean_variable_value("anti_PI3K") << std::endl;
//                     std::cout << "  - AKT: " << pCell->phenotype.intracellular->get_boolean_variable_value("AKT") << std::endl;
                    
//                     // Update network parameters and set target node
//                     update_maboss_params(pCell, phenotype, target_node);
//                     pCell->custom_data[needs_network_reset_ix] = 1;
                    
//                     // Get new state after update
//                     bool new_state = pCell->phenotype.intracellular->get_boolean_variable_value(target_node);

//                     // Print the state of key nodes after update
//                     std::cout << "After update - Key nodes state:" << std::endl;
//                     std::cout << "  - PI3K: " << pCell->phenotype.intracellular->get_boolean_variable_value("PI3K") << std::endl;
//                     std::cout << "  - anti_PI3K: " << pCell->phenotype.intracellular->get_boolean_variable_value("anti_PI3K") << std::endl;
//                     std::cout << "  - AKT: " << pCell->phenotype.intracellular->get_boolean_variable_value("AKT") << std::endl;

//                     // Report the state change
//                     if (current_state != new_state) {
//                         std::cout << "SUCCESS: Network state changed for cell " << pCell->ID 
//                                  << " target_node: " << target_node 
//                                  << " from " << current_state << " to " << new_state 
//                                  << " (drug_effect: " << drug_effect << ")" << std::endl;
//                     } else {
//                         std::cout << "WARNING: Failed to change state for cell " << pCell->ID 
//                                  << " target_node: " << target_node 
//                                  << " (drug_effect: " << drug_effect << ")" << std::endl;
//                     }
//                 }
//             }
//         } else {
//             // Get current state before potential change
//             bool current_state = pCell->phenotype.intracellular->get_boolean_variable_value(target_node);
            
//             // Only update if the state needs to change and stochastic condition is met
//             if (current_state == 1 && uniform_random() < std::abs(drug_effect)) {
//                 std::cout << "Cell " << pCell->ID << " attempting to deactivate " << target_node 
//                           << " with drug_effect: " << drug_effect << std::endl;
                
//                 pCell->phenotype.intracellular->set_boolean_variable_value(target_node, 0);
//                 pCell->custom_data[needs_network_reset_ix] = 0;
                
//                 // Get new state after update
//                 bool new_state = pCell->phenotype.intracellular->get_boolean_variable_value(target_node);
                
//                 // Report the state change
//                 if (current_state != new_state) {
//                     std::cout << "SUCCESS: Network state changed for cell " << pCell->ID 
//                              << " target_node: " << target_node 
//                              << " from " << current_state << " to " << new_state 
//                              << " (drug_effect: " << drug_effect << ")" << std::endl;
//                 } else {
//                     std::cout << "WARNING: Failed to change state for cell " << pCell->ID 
//                              << " target_node: " << target_node 
//                              << " (drug_effect: " << drug_effect << ")" << std::endl;
//                 }
//             }
//         }
//     }
// }



// void update_boolean_model_inputs_smooth(Cell* pCell, Phenotype& phenotype, double dt) {
//     // Early return if cell is dead or null
//     if (!pCell || pCell->phenotype.death.dead)
//         return;

//     // Check if intracellular model exists
//     if (!pCell->phenotype.intracellular) {
//         std::cout << "Warning: No intracellular model found for cell " << pCell->ID << std::endl;
//         return;
//     }

//     int n_drugs = 2;
//     std::string drugs[n_drugs] = { "drug_X", "drug_Y" };
    
//     // Get network reset indices with validation
//     static int needs_network_reset_X_ix = -1;
//     static int needs_network_reset_Y_ix = -1;
    
//     // Initialize indices if not already done
//     if (needs_network_reset_X_ix == -1) {
//         needs_network_reset_X_ix = pCell->custom_data.find_variable_index("needs_network_reset_X");
//         needs_network_reset_Y_ix = pCell->custom_data.find_variable_index("needs_network_reset_Y");
        
//         // Validate indices
//         if (needs_network_reset_X_ix == -1 || needs_network_reset_Y_ix == -1) {
//             std::cout << "Error: Required custom data variables not found for cell " << pCell->ID << std::endl;
//             return;
//         }
//     }
    
//     // Check if any drug has already triggered a network reset
//     bool network_already_reset = (pCell->custom_data[needs_network_reset_X_ix] == 1 || 
//                                 pCell->custom_data[needs_network_reset_Y_ix] == 1);
    
//     for (int i = 0; i < n_drugs; i++) {
//         std::string drug_name = drugs[i];
//         std::string target_node = get_drug_target(drug_name);

//         if (target_node == "null")
//             continue;

//         int needs_network_reset_ix = (drug_name == "drug_X") ? needs_network_reset_X_ix : needs_network_reset_Y_ix;

//         // Calculate drug effect with validation
//         double drug_effect;
//         try {
//             drug_effect = (get_mapping_type(pCell, drug_name) == 1) ? 
//                 calculate_drug_effect_linear(pCell, drug_name) :
//                 calculate_drug_effect_with_binding(pCell, drug_name, dt);
//         } catch (...) {
//             std::cout << "Error: Failed to calculate drug effect for " << drug_name << " in cell " << pCell->ID << std::endl;
//             continue;
//         }

//         // Apply the drug effect to gradually increase activation rate
//         if (drug_effect > 0) {
//             if (uniform_random() < drug_effect) {
//                 // Check if this is the first time this drug triggers an effect
//                 if (pCell->custom_data[needs_network_reset_ix] == 0) {
//                     if (!network_already_reset) {
//                         // First drug to trigger - do full network reset
//                         try {
//                             update_maboss_params(pCell, target_node);
//                             network_already_reset = true;
                            
//                             std::cout << "Full network reset for cell " << pCell->ID << ":\n"
//                                      << "  - First drug: " << drug_name << " targeting " << target_node 
//                                      << "  - Network reset counter: " << pCell->custom_data[needs_network_reset_ix] << std::endl;
//                         } catch (...) {
//                             std::cout << "Error: Failed to update MaBoSS parameters for cell " << pCell->ID << std::endl;
//                             continue;
//                         }
//                     } else {
//                         // Another drug is already active - just set the target node
//                         try {
//                             pCell->phenotype.intracellular->set_boolean_variable_value(target_node, 1);
//                             std::cout << "Additional drug effect for cell " << pCell->ID << ":\n"
//                                      << "  - Drug: " << drug_name << " targeting " << target_node 
//                                      << "  - Network already reset by previous drug" << std::endl;
//                         } catch (...) {
//                             std::cout << "Error: Failed to set boolean variable for cell " << pCell->ID << std::endl;
//                             continue;
//                         }
//                     }
//                     pCell->custom_data[needs_network_reset_ix] = 1;
//                 }
//             }
//         } else {
//             // Reset the network state for this drug's target
//             try {
//                 pCell->phenotype.intracellular->set_boolean_variable_value(target_node, 0);
//             } catch (...) {
//                 std::cout << "Error: Failed to reset boolean variable for cell " << pCell->ID << std::endl;
//                 continue;
//             }
            
//             // Reset the network update flag for this drug
//             if (pCell->custom_data[needs_network_reset_ix] == 1) {
//                 pCell->custom_data[needs_network_reset_ix] = 0;
                
//                 // Get the state of other drug's target for logging
//                 std::string other_drug = (drug_name == "drug_X") ? "drug_Y" : "drug_X";
//                 std::string other_target = get_drug_target(other_drug);
//                 bool other_target_active = false;
//                 if (other_target != "null") {
//                     try {
//                         other_target_active = pCell->phenotype.intracellular->get_boolean_variable_value(other_target);
//                     } catch (...) {
//                         std::cout << "Error: Failed to get boolean variable for cell " << pCell->ID << std::endl;
//                         continue;
//                     }
//                 }

//                 std::cout << "Drug effect removed for cell " << pCell->ID << ":\n"
//                          << "  - Reset by: " << drug_name << " targeting " << target_node << "\n"
//                          << "  - Other drug (" << other_drug << ") target " << other_target 
//                          << " remains: " << (other_target_active ? "active" : "inactive") << std::endl;
//             }
//         }
//     }
// }

// void update_maboss_params(Cell* pCell, Phenotype& phenotype, std::string target_node) {
//     if (!pCell || !pCell->phenotype.intracellular) {
//         std::cout << "Error: Invalid cell or missing intracellular model" << std::endl;
//         return;
//     }

//     auto* maboss_model = static_cast<MaBoSSIntracellular*>(pCell->phenotype.intracellular);
//     if (!maboss_model) {
//         std::cout << "Error: Failed to cast to MaBoSSIntracellular" << std::endl;
//         return;
//     }

//     // Get parameters
//     double new_intracellular_dt = parameters.doubles("new_intracellular_dt");
//     double new_scaling = parameters.doubles("new_scaling");

//     // Update MaBoSS parameters
//     maboss_model->time_step = new_intracellular_dt;
//     maboss_model->scaling = new_scaling;

//     // Set the target node to 1 (activated)
//     maboss_model->set_boolean_variable_value(target_node, 1);

//     // Run the simulation to update the network state
//     maboss_model->maboss.run_simulation();

//     // Print debug information
//     std::cout << "Updated MaBoSS parameters for cell " << pCell->ID << ":" << std::endl;
//     std::cout << "  - Target node: " << target_node << std::endl;
//     std::cout << "  - New state: " << maboss_model->get_boolean_variable_value(target_node) << std::endl;
//     std::cout << "  - Time step: " << maboss_model->time_step << std::endl;
//     std::cout << "  - Scaling: " << maboss_model->scaling << std::endl;
// }