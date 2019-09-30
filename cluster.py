"""
This module executes the clustering algorithm on the executed Python and R
snippets and stores a pickle/csv file that records the pairs with the semantic 
similary scores.
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
import itertools
import time
import pickle
import multiprocessing
import random
import Levenshtein

from compare import compare

from generate import generate_arg_from_df

import sys
sys.path.append("./python_side")
from execute import DataframeStore

NUM_WORKERS = 4
PY_PICKLE_PATH = './files/py_dfs.pkl'
R_PICKLE_PATH = './files/r_dfs.pkl'
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
    snips = [pair for pair in itertools.zip_longest(pysnips,rsnips)]
    snips = list(filter(None, snips))
    print(len(snips))
    # for i in filtered: 
    #     py = i[0]
    #     if i[1] == None: break
    #     r = i[1]

    # for s in snips:
    #     print(s['expr'])
    clusters = [] # list of dicts {"rep": s, snip, snip,...}
    cnt = 0
    start_time = time.time()
    for s in snips:
        py = s[0]
        if s[1] == None: break
        r = s[1]
        # print('py', py['expr'], 'r', r['expr'])
        for c in clusters:
            # print(c)
            rep = c['rep']
            # print('expr', rep['expr'])
            # TODO scale this to go through all executions when input args > 1
            score = compare(rep['test_results'][0], r['test_results'][0])
            if score >= SIM_T:
                py_expr = rep['expr']
                r_expr = r['expr']
                # print('here', score, py_expr, r_expr)
                edit_distance = jaro(py_expr, r_expr)
                # Saving tuple of (expr, semantic score, edit distance?)
                if len(c['snippets'].items()) == 0:
                    c['snippets'] = {}
                c['snippets'][r_expr] = (round(score, 3), edit_distance)
                # print(c['snippets'])
                # print(py_expr, c['snippets'][r_expr])
                # c['snippets'].append({expr: (expr, round(score,3), edit_distance)})
                # print(s['expr'])
                break
        if sum([py['expr'] in c['rep']['expr'] for c in clusters]) == 0:
            # print('oy', py['expr'])
            cluster = {'rep': py, 'snippets':{}}
            clusters.append(cluster)
    print(f"Time taken: {round((time.time() - start_time), 2)}") # 200 snippets takes ~2min
    return clusters

def levenshtein(r_snippet, py_snippet):
  return Levenshtein.distance(r_snippet, py_snippet)

def hamming(r_snippet, py_snippet):
  return Levenshtein.hamming(r_snippet, py_snippet)

def jaro(r_snippet, py_snippet):
  return 1.0 - Levenshtein.jaro(r_snippet, py_snippet)

def jaro_winkler(r_snippet, py_snippet):
  return 1.0 - Levenshtein.jaro_winkler(r_snippet, py_snippet)

def print_clusters(clusters):
    """Use this to debug clusters found"""
    for c in clusters:
        if len(c['snippets'].items()) > 0:
            print('----\n', c['rep']['expr'], len(c['snippets'].items()), '\n~~~~')
            for k, v in c['snippets'].items():
                print(f"R: semantic score: {v[0]} edit distance: {round(v[1], 3)}) {k}")
            print('----\n')
        else:
            print(c['rep']['expr'], len(c['snippets'].items()))

def store_clusters(clusters):
    # TODO store clusters in a csv file to query for pairs later
    data = OrderedDict()
    for c in clusters:
        data['rep'] = c['rep']
        if len(c['snippets'].items()) > 0:
            for k, v in c['snippets']:
                print(k, v)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        num_snippets = sys.argv[1]
        try:
            num_snippets = int(num_snippets)
            if num_snippets >= 2:
                split, remainder = divmod(num_snippets, 2)
                # Load executed python and r snippets with their results
                pysnips = pickle.load(open(PY_PICKLE_PATH, "rb")).pairs
                rsnips = pickle.load(open(R_PICKLE_PATH, "rb")).pairs
                bigger = max(len(pysnips), len(rsnips))
                if split > bigger:
                    split, remainder = divmod(bigger, 2)
                    print(len(pysnips), len(rsnips))
                pysnips = pysnips[:split+remainder]
                rsnips = rsnips[:split]
                if len(sys.argv) > 2:
                    SIM_T = float(sys.argv[2])
                    SIM_T = min(1.0, max(0, SIM_T)) # lower bound to 0 and upper bound to 1
                clusters = cluster(pysnips, rsnips)
                print_clusters(clusters)
            else:
                print("invalid value for number of snippets!")
                print("usage: python cluster.py [number of snippets >= 2] [0 <= SIM_T <= 1.0]")
        except Exception as e:
            print("invalid option!", e)
            print("usage: python cluster.py [number of snippets >= 2] [0 <= SIM_T <= 1.0]")
            sys.exit(1)
    else:
        print("invalid option!")
        print("usage: python cluster.py [number of snippets >= 2] [0 <= SIM_T <= 1.0]")
        sys.exit(1)



   
    
