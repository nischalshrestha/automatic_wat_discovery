import numpy as np
import pandas as pd
import time
import pickle
import sys
sys.path.append("../")

from generate import generate_args_from_df
from execute import DataframeStore

PICKLE_PATH = '/Volumes/TarDisk/snippets/'

# PassengerId      int64
# Survived         int64 (level) - randomize
# Pclass           int64 (level) - randomize
# Name            object 
# Sex             object (level) - randomize
# Age            float64
# SibSp            int64
# Parch            int64
# Ticket          object (level) - randomize
# Fare           float64
# Cabin           object (level) - randomize
# Embarked        object (level) - randomize
df = pd.read_csv("../train.csv")

# Comparison procedure
# Input: pydata, rdata
# Check types
#   If same type, perform comparison for that data type
#       Calculate similarity score between the two snippets for that particular execution
#   If not same type, no need for comparison and similarity is 0
#   Continually update the snippet's similarity score based on:
#       mean of all the similarity scores for each execution comparison
#   Restore the respective python and r data back to the pickle (with similarity score now associated)

# Clustering algorithm:
# Input: F - List of Functions with Input and Output 
# Output: C - List of clusters
# procedure Cluster(F)
#     C←φ
#     for all F ∈ Fdo
#         for all C ∈ C do
#             O ← GetRepresentive(C)
#             if Similarity(O,F) ≥ SIM_T then
#                 C←C∪F
#                 break
#         if ∀C ∈ C, F not in C then
#             C|C|+1 ←F SetRepresentative(C|C|+1,F)
#             C←C∪C return C

if __name__ == '__main__':
    # print(df.dtypes)
    # TODO use compare module to now test comparing many different random dataframes
    args = generate_args_from_df(df, n_args=20)
    half = int(len(args)/2)
    for a,b in zip(args[:half], args[half:]):
        df1 = a.iloc[:np.random.randint(1, a.shape[0] + 1), :np.random.randint(1, a.shape[1] + 1)]
        df2 = a.iloc[:np.random.randint(1, b.shape[0] + 1), :np.random.randint(1, b.shape[1] + 1)]
        print(df1.shape, df2.shape)
        df_diff(df1, df2)

    # pydict = pickle.load(open(PICKLE_PATH+"py_dfs.pkl", "rb")).pairs
    # rdict = pickle.load(open(PICKLE_PATH+"r_dfs.pkl", "rb")).pairs
    # print(len(pydict.items()), len(rdict.items()))
    # 6619 1013
