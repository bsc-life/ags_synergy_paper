import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def get_strategy_from_filename(filename):
    """
    Determines the optimization strategy from the experiment name.
    """
    if "CMA" in filename:
        return "CMA"
    if "sweep" in filename:
        return "sweep"
    return "Unknown"

def get_drug_from_filename(filename):
    """
    Extracts the drug name(s) from the experiment name.
    """
    if "pi3k_mek" in filename:
        return "PI3K_MEK"
    if "akt_mek" in filename:
        return "AKT_MEK"
    if "PI3K" in filename:
        return "PI3K"
    if "MEK" in filename:
        return "MEK"
    if "AKT" in filename:
        return "AKT"
    if "CTRL" in filename:
        return "WT"
    return "Unknown"

# --- Data Loading ---
def load_rmse_data_for_synergy(experiment_list):
    """
    Loads RMSE data from top_n.csv files for a list of experiments,
    sorts them into a canonical order, and returns a DataFrame.
    """
    all_rmse_data = []

    for experiment in experiment_list:
        strategy = experiment["strategy"]
        exp_name = experiment["name"]
        top_n = experiment["top_n"]

        if strategy == 'sweep':
            csv_path = f'results/{strategy}_summaries/final_summary_{exp_name}.csv'
        else:
            csv_path = f'results/{strategy}_summaries/final_summary_{exp_name}/top_{top_n}.csv'

        if not os.path.exists(csv_path):
            logging.warning(f"CSV file not found, skipping: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
            fitness_column = df.columns[-1]

            if strategy == "sweep":
                top_df = df.sort_values(by=fitness_column, ascending=True).head(top_n)
            else:
                top_df = df.head(top_n)

            rmses = top_df[fitness_column].tolist()

            drug_name = experiment.get("drug_name")
            if drug_name == "WT":
                condition_label = "Control"
            else:
                condition_label = drug_name.replace("_", "+")
                if 'i' not in condition_label[-2:]:
                    condition_label += 'i'

            for rmse in rmses:
                all_rmse_data.append({"Condition": condition_label, "RMSE": rmse})

        except Exception as e:
            logging.error(f"Failed to process {csv_path}: {e}")

    if not all_rmse_data:
        return pd.DataFrame(), []

    # Define a canonical order for plotting
    order_map = {"Control": 0, "PI3Ki": 1, "AKTi": 1, "MEKi": 2, "PI3Ki+MEKi": 3, "AKTi+MEKi": 3}
    rmse_df = pd.DataFrame(all_rmse_data)
    
    # Get unique conditions and sort them based on the map
    unique_conditions = rmse_df['Condition'].unique()
    sorted_conditions = sorted(unique_conditions, key=lambda x: order_map.get(x, 99))
    
    return rmse_df, sorted_conditions

# --- Plotting Function ---
def create_synergy_violin_comparison_plot(pi3kmek_df, pi3kmek_order, aktmek_df, aktmek_order, output_dir, filename_base):
    """
    Creates a side-by-side violin plot comparing RMSE distributions for two synergy experiments.
    """
    logging.info("Creating side-by-side synergy RMSE violin plots...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300, sharey=True)
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")

    # --- Left Plot: PI3K + MEK Synergy ---
    if not pi3kmek_df.empty:
        sns.violinplot(ax=axes[0], x="Condition", y="RMSE", data=pi3kmek_df, order=pi3kmek_order, palette="muted", cut=0)
        axes[0].set_title("PI3Ki + MEKi Synergy", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Condition", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("RMSE", fontsize=12, fontweight="bold")
        axes[0].tick_params(axis='x', rotation=30, labelsize=10)

    # --- Right Plot: AKT + MEK Synergy ---
    if not aktmek_df.empty:
        sns.violinplot(ax=axes[1], x="Condition", y="RMSE", data=aktmek_df, order=aktmek_order, palette="muted", cut=0)
        axes[1].set_title("AKTi + MEKi Synergy", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Condition", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("")  # Remove redundant y-label
        axes[1].tick_params(axis='x', rotation=30, labelsize=10)

    sns.despine()
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{filename_base}.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    fig.savefig(output_path.replace('.png', '.svg'), format='svg', bbox_inches='tight', transparent=True)
    plt.close(fig)

    logging.info(f"Synergy RMSE violin comparison plot saved to {output_path}")

# --- Main Execution ---
def main():
    """
    Main function to define experiments, load data, and generate plots.
    """
    best_control_experiment_name = "CTRL_CMA-1110-1637-5p"
    best_pi3kmek_combined_experiment_name = "synergy_sweep-pi3k_mek-1104-2212-18p_transient_delayed_uniform_5k_10p"
    best_pi3kmek_PI3K_singledrug_experiment_name = "synergy_sweep-pi3k_mek-1104-2212-18p_PI3K_transient_delayed_uniform_5k_10p"
    best_pi3kmek_MEK_singledrug_experiment_name = "synergy_sweep-pi3k_mek-1104-2212-18p_MEK_transient_delayed_uniform_5k_10p"

    best_aktmek_combined_experiment_name = "synergy_sweep-akt_mek-1204-1639-18p_transient_delayed_uniform_postdrug_RMSE_5k"
    best_aktmek_AKT_singledrug_experiment_name = "synergy_sweep-akt_mek-1104-2212-18p_AKT_transient_delayed_uniform_5k_singledrug"
    best_aktmek_MEK_singledrug_experiment_name = "synergy_sweep-akt_mek-1104-2212-18p_MEK_transient_delayed_uniform_5k_singledrug"

    pi3k_mek_synergy_experiments = [
        {"name": best_control_experiment_name, "top_n": 10},
        {"name": best_pi3kmek_PI3K_singledrug_experiment_name, "top_n": 10},
        {"name": best_pi3kmek_MEK_singledrug_experiment_name, "top_n": 10},
        {"name": best_pi3kmek_combined_experiment_name, "top_n": 10}
    ]

    akt_mek_synergy_experiments = [
        {"name": best_control_experiment_name, "top_n": 10},
        {"name": best_aktmek_AKT_singledrug_experiment_name, "top_n": 10},
        {"name": best_aktmek_MEK_singledrug_experiment_name, "top_n": 10},
        {"name": best_aktmek_combined_experiment_name, "top_n": 10}
    ]

    # Add strategy and drug_name to each experiment dict
    for exp_list in [pi3k_mek_synergy_experiments, akt_mek_synergy_experiments]:
        for exp in exp_list:
            exp["strategy"] = get_strategy_from_filename(exp["name"])
            exp["drug_name"] = get_drug_from_filename(exp["name"])

    # Process PI3K+MEK synergy
    pi3kmek_df, pi3kmek_order = load_rmse_data_for_synergy(pi3k_mek_synergy_experiments)

    # Process AKT+MEK synergy
    aktmek_df, aktmek_order = load_rmse_data_for_synergy(akt_mek_synergy_experiments)

    # Create plot
    output_dir = 'results/publication_plots/synergy_comparison'
    os.makedirs(output_dir, exist_ok=True)
    create_synergy_violin_comparison_plot(
        pi3kmek_df, pi3kmek_order,
        aktmek_df, aktmek_order,
        output_dir,
        "synergy_rmse_violin_comparison_standalone"
    )

if __name__ == "__main__":
    main() 