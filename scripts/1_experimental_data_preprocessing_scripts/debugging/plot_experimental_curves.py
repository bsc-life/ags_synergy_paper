import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_data(folder):
    # Get all CSV files in the specified folder
    csv_files = [f for f in os.listdir(folder) if f.endswith('.csv') and os.path.isfile(os.path.join(folder, f))]


    # Create a folder for the plots
    plot_folder = os.path.join(folder, 'plots')
    os.makedirs(plot_folder, exist_ok=True)

    for csv_file in csv_files:
        # Read the CSV file
        df = pd.read_csv(os.path.join(folder, csv_file))
        print("Reading file:", csv_file)

        # Plotting
        plt.figure()

        try:
            plt.plot(df['Time'], df['Average_Cell_Index'], label='Average Cell Index')
        except KeyError:
            pass

        if 'Standard_Deviation_Cell_Index' in df.columns:
            plt.fill_between(df['Time'], 
                             df['Average_Cell_Index'] - df['Standard_Deviation_Cell_Index'], 
                             df['Average_Cell_Index'] + df['Standard_Deviation_Cell_Index'], 
                                 alpha=0.2)
        plt.title(f'Plot for {csv_file}')
        plt.xlabel('Time')
        plt.ylabel('Average Cell Index')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plot_folder, f'plot_{csv_file}.png'), bbox_inches='tight', dpi=300)
        plt.close()

    # Combined plot
    plt.figure()
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(folder, csv_file))
        try:
            plt.plot(df['Time'], df['Average_Cell_Index'], label='Average Cell Index')
        except KeyError:
            pass
        # Start of Selection
        if 'Standard_Deviation_Cell_Index' in df.columns:
            plt.fill_between(df['Time'], 
                             df['Average_Cell_Index'] - df['Standard_Deviation_Cell_Index'], 
                             df['Average_Cell_Index'] + df['Standard_Deviation_Cell_Index'], 
                                 alpha=0.2)

    plt.title('Combined Plot of All CSVs')
    plt.xlabel('Time')
    plt.ylabel('Average Cell Index')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plot_folder, 'combined_plot.png'), bbox_inches='tight', dpi=300)
    plt.close()

    print("Plots saved to:", plot_folder)

# Example usage
# plot_csv_data('data/AGS_data/AGS_growth_data/output/csv')
# plot_csv_data('data/AGS_data/AGS_growth_data/output/csv/processed')
plot_csv_data('data/AGS_data/AGS_growth_data/output/csv/processed/normalized_curves')
