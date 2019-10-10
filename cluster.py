"""
This module executes the clustering algorithm on the executed Python and R
snippets and stores a pickle/csv file that records the pairs with the semantic 
similary scores.
"""

import numpy as np
import pandas as pd
import time
import pickle
import multiprocessing
import Levenshtein

import sys
sys.path.append("./python_side")
from execute import DataframeStore
from compare import compare

NUM_WORKERS = multiprocessing.cpu_count()
PY_PICKLE_PATH = './files/py_dfs.pkl'
R_PICKLE_PATH = './files/r_dfs.pkl'
CLUSTERS_PATH = './files/'
SIM_T = 0.85 
KEEP_RESULTS = True

flatten = lambda l: [item for sublist in l for item in sublist]

# Load executed python and r snippets if running module directly
# Note: these two are global vars that are be used with below functions
pysnips = None
rsnips = None

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

def compare_results(py, r):
    """
    Compares test results of a Python and R snippet pair and returns a List of tuple(s) 
    to store later into a csv results file:

    (py expr, r expr, pytest, rtest, row_diff, col_diff, semantic score, edit distance)
    TODO create a class that represents a test case to make it easier to work with
    """
    # We will call compare on each of the results
    py_results = py['test_results']
    r_results = r['test_results']
    py_expr = py['expr']
    r_expr = r['expr']
    edit_distance = round(jaro(py_expr, r_expr), 3)
    # This is so that when we write the results to a csv, the expressions are readable
    py_reformat = py_expr.replace(",", ", ").replace("&", " & ").replace("==", " == ")
    r_reformat = r_expr.replace(",", ", ").replace("&", " & ").replace("==", " == ")
    # print(py_expr, r_expr)
    results = []
    scores = []
    discarded = []
    for t1, t2 in zip(py_results, r_results):
        # Collect and store comparison scores of py vs r test cases
        py_out, r_out = t1[2], t2[2]
        # print(py_expr, r_expr)
        score = compare(py_out, r_out)
        # The same test case is used for corresponding R snippet, so just get the Python result's argument
        test_case = t1[1]
        if type(score) == tuple:
            overall = score[0]*score[1]*score[2]
            tuple_result = (py_reformat, r_reformat, test_case, py_out, r_out, overall, score[1], score[2], round(score[0], 3), score[3], edit_distance)
        else:
            overall = score
            # For the non-dataframe output case we leave row/col, lca diff blank
            tuple_result = (py_reformat, r_reformat, test_case, py_out, r_out, overall, "", "", overall, "", edit_distance)
                # return results
        scores.append(overall)    
        if KEEP_RESULTS:
            results.append(tuple_result)
    mean_score = sum(scores) / len(scores) if len(scores) > 0 else 0
    mean_score = round(mean_score, 3)
    # print(mean_score)
    # For now, have an overall score for the cluster
    # TODO Restructure the csv to make it more clear in the future
    if KEEP_RESULTS:
        results.insert(0, (py_reformat, r_reformat, "", "", "", mean_score, "", "", "", "", edit_distance))
    else:
        results.insert(0, (py_reformat, r_reformat, mean_score, edit_distance))
    # Only if the mean score satisfies threshold do we return the results
    return results if mean_score >= SIM_T else []

def compare_rpy(py):
    """
    Compares a Python snippet's execution results against all of the R snippets
    and their execution results.

    Ensures that only the Python/R pairs that meets the threshold is kept.
    """
    max_score = 0
    results = []
    scores = []
    for r in rsnips:
        compare_scores = compare_results(py, r)
        for s in compare_scores:
            if len(s) >= 1:
                results.append(s)
    return results if len(results) > 0 else None

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
        results = flatten(result)
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

def store_clusters(clusters):
    """
    Stores clusters which is a list of tuples where each element is a 
    column value
    """ 
    if KEEP_RESULTS:
        df = pd.DataFrame(clusters, columns =['Python', 'R', 'Test Case', 'Python result', \
                'R result', 'Overall', 'Row Diff', 'Col Diff', 'Semantic', 'Largest Common', \
                'Edit Distance'])
    else:
        df = pd.DataFrame(clusters, columns =['Python', 'R', 'Overall', 'Edit Distance'])
    tolerance = round(1-SIM_T, 2)
    df.to_csv(f"{CLUSTERS_PATH}clusters_{tolerance}.csv", index=False)
            
if __name__ == '__main__':
    if len(sys.argv) > 1:
        sim_t = float(sys.argv[1])
        SIM_T = min(1.0, max(0, sim_t)) # lower bound to 0 and upper bound to 1
        KEEP_RESULTS = True if len(sys.argv) == 3 and "keep" in sys.argv[2] else False
        pysnips = pickle.load(open(PY_PICKLE_PATH, "rb")).pairs
        rsnips = pickle.load(open(R_PICKLE_PATH, "rb")).pairs
        clusters = simple_cluster()
        store_clusters(clusters)
    else:
        print("invalid command!")
        print("usage: python cluster.py SIM_T (<= 1.0) [keep]")
        sys.exit(1)

