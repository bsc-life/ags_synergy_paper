import pcdl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_dir = "./model/PhysiBoSS/output"

mcdsts = pcdl.TimeSeries(output_path = output_dir, graph=False)


list_of_relevant_vars = list()
all_data = pd.DataFrame()
for mcds in mcdsts.get_mcds_list():
    frame_df = mcds.get_cell_df()
    frame_df.reset_index(inplace=True)
    list_of_relevant_vars.append(frame_df)

all_data = pd.concat(list_of_relevant_vars, ignore_index=True)


S_value_data = all_data[['time', 'S_anti_real']]

print("this is the S_value_data")
print(S_value_data.head(5))

# do a plot of S_value_data
plt.figure(figsize=(10, 6))
plt.plot(S_value_data['time'], S_value_data['S_anti_real'], label='S_anti_real')
plt.title('S_anti_real Over Time')
plt.xlabel('Time')
plt.ylabel('S_anti_real')
plt.legend()
plt.show()


# # subset all_data to only include the "node" columns + time
# node_data = all_data[['time'] + [col for col in all_data.columns if 'node' in col]]
# # substitute all NaN with 0
# # node_data = node_data.fillna(0)

# # print the first 5 rows of node_data
# print(node_data.head())

# # Create a figure with a larger size
# plt.figure(figsize=(10, 6))

# # Plot all three apoptosis nodes
# sns.lineplot(x='time', y='node_FOXO', data=node_data, label='FOXO')
# sns.lineplot(x='time', y='node_Caspase8', data=node_data, label='Caspase8')
# sns.lineplot(x='time', y='node_Caspase9', data=node_data, label='Caspase9')

# # add vertical line at time = 80
# plt.axvline(x=80, color='red', linestyle='--', label='t=80')

# # Add title and labels
# plt.title('Apoptosis Node States Over Time')
# plt.xlabel('Time')
# plt.ylabel('Node State')
# plt.legend()

# # Adjust layout to prevent label clipping
# plt.tight_layout()

# # Save and show the plot
# output_dir = "./results/node_states"
# os.makedirs(output_dir, exist_ok=True)
# plt.savefig(f'{output_dir}/apoptosis_nodes.png')
# plt.show()




