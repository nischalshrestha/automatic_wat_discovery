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

from compare import compare

from generate import generate_args_from_df

import sys
sys.path.append("./python_side")
from execute import DataframeStore

NUM_WORKERS = 4
PY_PICKLE_PATH = '/Volumes/TarDisk/snippets/py_dfs.pkl'
R_PICKLE_PATH = '/Volumes/TarDisk/snippets/r_dfs.pkl'
SIM_T = 0.85

flatten = lambda l: [item for sublist in l for item in sublist]

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

# TODO store results into a pickle
def cluster(pysnips, rsnips):
    """
    Clusters the python and R snippet outputs using representative based partitioning
    strategy.

    Clustering algorithm:
    Input: F - List of Functions with Input and Output 
    Output: C - List of clusters
    procedure Cluster(F)
        C ← φ
        for all F ∈ F do
            for all C ∈ C do
                O ← GetRepresentive(C)
                if Similarity(O, F) ≥ SIM_T then
                    C ← C ∪ F
                    break
            if ∀C ∈ C, F not in C then
                C|C|+1 ← F 
                SetRepresentative(C|C|+1, F)
                C ← C ∪ C 
        return C
    """
    snips = pysnips + rsnips
    clusters = [] # list of dicts {"rep": s, snip, snip,...}
    cnt = 0
    start = time.time()
    for s in snips:
        for c in clusters:
            # print(c)
            rep = c['rep']
            # print(rep['expr'])
            score = compare(rep['test_results'][0], s['test_results'][0])
            if score >= SIM_T:
                c['snippets'].append(s['expr'])
                # print(s['expr'])
                break
        if sum([s['expr'] in c['snippets'] for c in clusters]) == 0:
            cluster = {'rep': s, 'snippets':[]}
            clusters.append(cluster)
    print(f"Time taken: {time.time()-start}") # 200 snippets takes ~2min
    return clusters

def print_clusters(clusters):
    """Use this to debug clusters found"""
    for c in clusters:
        print(c['rep']['expr'], len(c['snippets']))

if __name__ == '__main__':
    # TODO add optional argument for SIM_T as well
    if len(sys.argv) > 1:
        num_snippets = sys.argv[1]
        try:
            num_snippets = int(num_snippets)
            if num_snippets >= 2:
                split, remainder = divmod(num_snippets, 2)
                # Load executed python and r snippets with their results
                pysnips = pickle.load(open(PY_PICKLE_PATH, "rb")).pairs[:split+remainder]
                rsnips = pickle.load(open(R_PICKLE_PATH, "rb")).pairs[:split]
                clusters = cluster(pysnips, rsnips)
                print(len(clusters))
                print_clusters(clusters)
            else:
                print("invalid value for number of snippets!")
                print("usage: python cluster.py [number of snippets >= 2]")
        except:
            print("invalid option!")
            print("usage: python cluster.py [number of snippets >= 2]")
            sys.exit(1)


   
    