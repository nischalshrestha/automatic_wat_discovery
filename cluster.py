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

# from generate import generate_arg_from_df

import sys
sys.path.append("./python_side")
from execute import DataframeStore

NUM_WORKERS = 4
PY_PICKLE_PATH = './files/py_dfs.pkl'
R_PICKLE_PATH = './files/r_dfs.pkl'
CLUSTERS_PATH = './files/'
SIM_T = 0.85

flatten = lambda l: [item for sublist in l for item in sublist]

 # Load executed python and r snippets with their results
pysnips = pickle.load(open(PY_PICKLE_PATH, "rb")).pairs
rsnips = pickle.load(open(R_PICKLE_PATH, "rb")).pairs

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

def compare_results(rep, r):
    """Compares test results of rep and r"""
    # We will call compare on each of the results
    rep_results = rep['test_results']
    r_results = r['test_results']
    py_expr = rep['expr']
    r_expr = r['expr']
    # print(py_expr, r_expr)
    results = []
    for t1 in rep_results:
        # Collect and store the max score from all test results
        edit_distance = round(jaro(py_expr, r_expr), 3)
        for t2 in r_results:
            score = compare(t1, t2)
            # Store  tuple:
            # py expr, r expr, pytest, rtest, row_diff, col_diff, semantic score, edit dist
            if type(score) == tuple and score[0] >= SIM_T:
                rounded = round(score[0],3)
                results.append((py_expr, r_expr, t1, t2, score[1], score[2], rounded, edit_distance))
            elif type(score) != tuple and score >= SIM_T:
                rounded = round(score, 3)
                # For the non-dataframe output case we just report 0 for row/col diff
                results.append((py_expr, r_expr, t1, t2, 0, 0, rounded, edit_distance))
    return flatten(results)

def compare_rpy(py):
    """
    Compares a Python snippet's execution results against all of the R snippets
    and their execution results.
    """
    max_score = 0
    results = []
    for r in rsnips:
        compare_scores = compare_results(py, r)
        if len(compare_scores) > 0:
            results.append(compare_scores)
        # if score[0] >= SIM_T:
            # results.append((py['expr'], r['expr'], py['test_results'][0], r['test_results'][0], score[0], score[1]))
            # print('score', score)
    return results if len(results) > 0 else None

def simple_cluster(pynsips, rsnips):
    """
    A naive pair-wise comparison approach to cluster Python/R snippets according
    to their execution output results. Each Python snippet and its execution result
    gets compared to all the other R snippets and their results.
    
    It is slower than a representative-based approach but seems to yield better results.
    """
    scores = []
    start_time = time.time()
    results = None
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map_async(compare_rpy, pysnips)
        results.wait()
        result = results.get()
        result = list(filter(None, result))
        results = flatten(list(result))
        # print(results)
    end_time = time.time()
    print(f"Time taken: {round((end_time - start_time), 2)} secs")
    return results

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
    clusters = [] # list of dicts {"rep": s, snip, snip,...}
    start_time = time.time()
    for s in snips:
        # print(s)
        py = s[0]
        if s[1] == None: break
        r = s[1]
        # print('py', py['expr'], 'r', r['expr'])
        for c in clusters:
            rep = c['rep']
            scores = compare_results(rep, r)
            if scores[0] >= SIM_T:
                # Saving tuple of (expr, semantic score, edit distance?)
                if len(c['snippets'].items()) == 0:
                    c['snippets'] = {}
                c['snippets'][r['expr']] = (scores[0], scores[1])
                break
        if sum([py['expr'] in c['rep']['expr'] for c in clusters]) == 0:
            cluster = {'rep': py, 'snippets':{}}
            clusters.append(cluster)
    print(f"Time taken: {round((time.time() - start_time), 2)}") # 200 snippets takes ~2min
    return clusters

def levenshtein(r_snippet, py_snippet):
  return Levenshtein.distance(r_snippet, py_snippet)

def hamming(r_snippet, py_snippet):
  return Levenshtein.hamming(r_snippet, py_snippet)

def jaro(r_snippet, py_snippet):
  return Levenshtein.jaro(r_snippet, py_snippet)

def jaro_winkler(r_snippet, py_snippet):
  return 1.0 - Levenshtein.jaro_winkler(r_snippet, py_snippet)

def print_snips(snips):
    for s in snips:
        print(s['expr'], len(s['test_results']))

# TODO turn this function into another store clusters 
def print_clusters(clusters):
    """Use this to debug clusters found"""
    for c in clusters:
        if len(c['snippets'].items()) > 0:
            print('----\n', c['rep']['expr'], len(c['snippets'].items()), '\n~~~~')
            for k, v in c['snippets'].items():
                print(f"{k} {v[0]} {round(v[1], 3)}")
            # print(3*(c['rep']['expr']))
            print('----\n')
        else:
            print(c['rep']['expr'], len(c['snippets'].items()))

def store_clusters(clusters):
    """
    Stores clusters which is a list of tuples where each element is a 
    column value
    """ 
    # TODO store row_diff / col_diff as well
    df = pd.DataFrame(clusters, columns =['Python', 'R', 'Python result', \
            'R result', 'Row Diff', 'Col Diff', 'Semantic', 'Edit Distance'])
    tolerance = round(1-SIM_T, 2)
    df.to_csv(f"{CLUSTERS_PATH}clusters_{tolerance}.csv", index=False)
            
if __name__ == '__main__':
    if len(sys.argv) > 1:
        SIM_T = float(sys.argv[1])
        SIM_T = min(1.0, max(0, SIM_T)) # lower bound to 0 and upper bound to 1
        # George's method
        # clusters = cluster(pysnips, rsnips)
        # print_clusters(clusters)
        # Naive method
        clusters = simple_cluster(pysnips, rsnips)
        store_clusters(clusters)
    else:
        print("invalid option!")
        print("usage: python cluster.py [0 <= SIM_T <= 1.0]")
        sys.exit(1)
   
