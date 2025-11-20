import pcdl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# output directory is two directories up
output_folder_name = "output_drugY_test"
output_dir = f"/gpfs/projects/bsc08/bsc08494/AGS/EMEWS/model/PhysiBoSS/{output_folder_name}"

mcdsts = pcdl.pyMCDSts(output_path = output_dir, graph=False, verbose=False)

list_of_relevant_vars = list()
all_data = pd.DataFrame()
for mcds in mcdsts.get_mcds_list():
    frame_df = mcds.get_cell_df()
    frame_df.reset_index(inplace=True)
    list_of_relevant_vars.append(frame_df)

all_data = pd.concat(list_of_relevant_vars, ignore_index=True)

print(all_data.columns)


# we want to plot the distribution of number of cells per Z axis bin and their internalized drug concentrations

# first we need to get the number of cells per Z axis bin



