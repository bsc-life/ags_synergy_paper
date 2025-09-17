
# A small script to count the number of rows in a dataframe

import pandas as pd
import os 

def count_rows(df):
    return len(df)


if __name__ == "__main__":
    CMA_summaries_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                       "results", "CMA_summaries")
    emews_data_files = [f for f in os.listdir(CMA_summaries_folder) if f.endswith(".csv")]
    for file in emews_data_files:
        # print(f"Counting rows for {file}")
        df = pd.read_csv(os.path.join(CMA_summaries_folder, file))
        print(f"{file}:\t\t{count_rows(df)}")