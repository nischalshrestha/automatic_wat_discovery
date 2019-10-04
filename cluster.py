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
    """
    Compares test results of rep and r and returns a tuple to store later:

    (py expr, r expr, pytest, rtest, row_diff, col_diff, semantic score, edit distance)
    """
    # We will call compare on each of the results
    rep_results = rep['test_results']
    r_results = r['test_results']
    py_expr = rep['expr']
    r_expr = r['expr']
    py_reformat = py_expr.replace(",", ", ").replace("&", " & ").replace("==", " == ")
    r_reformat = r_expr.replace(",", ", ").replace("&", " & ").replace("==", " == ")
    # print(py_expr, r_expr)
    results = []
    for t1 in rep_results:
        # Collect and store the max score from all test results
        edit_distance = round(jaro(py_expr, r_expr), 3)
        for t2 in r_results:
            # print(t1)
            score = compare(t1, t2)
            # include score in results if 
            if type(score) == tuple and score[0] >= SIM_T:
                results.append((py_reformat, r_reformat, t1, t2, score[1], score[2], round(score[0], 3), score[3], edit_distance))
                return results
            elif type(score) != tuple and score >= SIM_T:
                # For the non-dataframe output case we leave row/col, lca diff blank
                results.append((py_reformat, r_reformat, t1, t2, "", "", round(score, 3), "", edit_distance))
                return results
    return results

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
    return flatten(results) if len(results) > 0 else None

def simple_cluster():
    """
    A naive pair-wise comparison approach to cluster Python/R snippets according
    to their execution output results. Each Python snippet and its execution result
    gets compared to all the other R snippets and their results.
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
        # print(len(results))
    end_time = time.time()
    print(f"Time taken: {round((end_time - start_time), 2)} secs")
    return results

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

def store_clusters(clusters, keep_outputs=False):
    """
    Stores clusters which is a list of tuples where each element is a 
    column value
    """ 
    # TODO provide option to save Python/R execution result in csv
    if keep_outputs:
        df = pd.DataFrame(clusters, columns =['Python', 'R', 'Python result', \
            'R result', 'Row Diff', 'Col Diff', 'Semantic', 'Largest Common', \
            'Edit Distance'])
    else:
        df = pd.DataFrame(clusters, columns =['Python', 'R', 'Row Diff', 'Col Diff', \
            'Semantic', 'Largest Common', 'Edit Distance'])
    tolerance = round(1-SIM_T, 2)
    df.to_csv(f"{CLUSTERS_PATH}clusters_{tolerance}.csv", index=False)
            
if __name__ == '__main__':
    if len(sys.argv) > 1:
        SIM_T = float(sys.argv[1])
        SIM_T = min(1.0, max(0, SIM_T)) # lower bound to 0 and upper bound to 1
        clusters = simple_cluster()
        store_clusters(clusters, keep_outputs=True)
    else:
        print("invalid option!")
        print("usage: python cluster.py [0 <= SIM_T <= 1.0]")
        sys.exit(1)

