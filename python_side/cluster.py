"""
This module executes the clustering algorithm and stores a pickle/csv file that
records the pairs with the semantic similary scores.
"""

import numpy as np
import pandas as pd
import time
import pickle
import multiprocessing
import random

import sys
sys.path.append("../")

from compare import compare

from generate import generate_args_from_df
from execute import DataframeStore

NUM_WORKERS = 4
PICKLE_PATH = '/Volumes/TarDisk/snippets/'
# Load executed python and r snippets with their results
pysnips = pickle.load(open(PICKLE_PATH+"py_dfs.pkl", "rb")).pairs
rsnips = pickle.load(open(PICKLE_PATH+"r_dfs.pkl", "rb")).pairs

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
# df = pd.read_csv("../train.csv")

def compare_rpy(r):
    max_score = 0
    for p in range(len(pysnips)):
        # print(r['test_results'][0], pysnips[p]['test_results'][0])
        # print(r['test_results'][0].shape, pysnips[p]['test_results'][0].shape)
        sim_score = compare(r['test_results'][0], pysnips[p]['test_results'][0])
        max_score = sim_score if sim_score > max_score else max_score
        # print('score', sim_score)
    return max_score

if __name__ == '__main__':
    # Comparison procedure
    # Input: pydata, rdata
    # Check types
    #   If same type, perform comparison for that data type
    #       Calculate similarity score between the two snippets for that particular execution
    #   If not same type, no need for comparison and similarity is 0
    #   Continually update the snippet's similarity score based on:
    #       mean of all the similarity scores for each execution comparison
    #   Restore the respective python and r data back to the pickle (with similarity score now associated)
    print(len(pysnips), len(rsnips))
    start = time.time()
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map_async(compare_rpy, rsnips)
        results.wait()
        result = results.get()
        print(result)
    print(f"time taken: {time.time()-start}")

    # TODO Clustering algorithm:
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


    
