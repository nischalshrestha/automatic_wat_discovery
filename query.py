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
    if len(sys.argv) >= 5:
        try:
            filename = sys.argv[1]+'_keep.csv'
            low_score = max(0.0, float(sys.argv[2]))
            high_score = min(1.0, float(sys.argv[3]))
            edit_score = min(1.0, max(0.0, float(sys.argv[4])))
            verbose = True if len(sys.argv) > 5 and sys.argv[5] == '-v' else False
            cluster = pd.read_csv(CSV_PATH+filename)
            result = cluster.loc[(cluster["overall"] >= low_score) \
                & (cluster["overall"] <= high_score) \
                & (cluster["edit_distance"] >= edit_score) \
                & (cluster["test_case"].notnull()), :]
            result.drop_duplicates(subset=["python"], inplace=True)
            result.dropna()
            result = result.sort_values("overall", ascending=False)
            for index, row in result.iterrows():
                if row["python"] == "df.sort_values('col1', ascending=False)" and row["r"] == "arrange(df, desc(col1))":
                    print("\n~~~~\n")
                    print(row["python"], row["r"])
                    print("Overall score:", round(row["overall"], 3))
                    print("Row score:", round(row["row_diff"], 3))
                    print("Column score:", row["col_diff"])
                    print("Semantic score:", row["semantic_score"])
                    print("Edit distance:", round(row["edit_distance"], 3))
                    if verbose:
                        print("Test case:\n")
                        print_full(pd.read_csv(StringIO(row["test_case"])))
                        print("\nPython output:\n")
                        print_full(pd.read_csv(StringIO(row["python_result"])))
                        print("\nR output:\n")
                        print_full(pd.read_csv(StringIO(row["r_result"])))
        except Exception as e:
            print('something went wrong', e)
    else:
        print('invalid command!')
        print("usage: python query.py filename low_score high_score edit_score -v")
        sys.exit(1)



