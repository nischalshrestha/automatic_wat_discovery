"""
This module is used to query interesting pairs of Python/R snippets from clusters and display
the scores and test cases
"""

import numpy as np
import pandas as pd
import pickle
import sys
from io import StringIO

# "Python","r","test_case","Python result","r result","overall","Row Diff","Col Diff","Semantic","Largest Common","edit_distance"
CSV_PATH = './files/'

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

if __name__ == '__main__':
    if len(sys.argv) > 4:
        try:
            filename = sys.argv[1]+'.csv'
            low_score = max(0.0, float(sys.argv[2]))
            high_score = min(1.0, float(sys.argv[3]))
            edit_score = min(1.0, max(0.0, float(sys.argv[4])))
            cluster = pd.read_csv(CSV_PATH+filename)
            result = cluster.loc[(cluster["overall"] >= low_score) \
                & (cluster["overall"] <= high_score) \
                & (cluster["edit_distance"] >= edit_score) \
                & (cluster["test_case"].notnull()), :]
            result.drop_duplicates(subset=["python"], inplace=True)
            result.dropna()
            result = result.sort_values("overall", ascending=False)
            for index, row in result.iterrows():
                print("~~~~")
                print(row["python"], row["r"])
                print("Row score:", row["row_diff"])
                print("Column score:", row["col_diff"])
                print("Overall score:", row["overall"])
                print("Edit distance:", row["edit_distance"])
                print("Test case:\n")
                print_full(pd.read_csv(StringIO(row["test_case"])))
                print("Python output:\n")
                print_full(pd.read_csv(StringIO(row["python_result"])))
                print("R output:\n")
                print_full(pd.read_csv(StringIO(row["r_result"])))
        except Exception as e:
            print('something went wrong', e)
    else:
        print('invalid command!')
        print("usage: python query.py filename low_score high_score edit_score")
        sys.exit(1)

