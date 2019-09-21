from collections import OrderedDict
import time
import multiprocessing
import pickle
from io import StringIO
# import dill

import ast
import astor
import autopep8
import pandas as pd
import numpy as np

# TODO generate random values for df and args
# TODO use eval for a simple testing harness

flatten = lambda l: [item for sublist in l for item in sublist]

def eval_expr(expr):
    """
    Evals an expression given a dataframe.
    Currently, this does not factor in args for expr
    """
    df = "mslacc = pd.read_csv('../train.csv')"
    code_str = f"{df} ; out = {expr}"
    # Need to handle drop() cases since inplace=True returns None
    # Just capture it by reassigning df to out again
    if "drop" in expr:
        code_str = code_str + "; out = mslacc"
    output = None
    try:
        code_obj = compile(code_str, '', 'single')
        eval(code_obj)
        output = locals()['out']
        return expr, output
        # print('out', out)
    except Exception as e:
        # print(e)
        pass

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

class DataframeStore:
    """Just a class to store a dict of <snippet, output> so it can be pickled"""
    pairs = {}
    def __init__(self, pairs):
        self.pairs = pairs
    def get_output(self):
       return self.pairs

def test_pyexec():
    import sys
    # Read python snippets generated by parallelize.py
    snips = pd.read_csv('pysnips.csv')
    snippets = flatten(snips.values)
    # Eval expressions and collect successful ones paired with output: (expr, output)
    start_time = time.time()
    failed, executed = 0, 0
    successful_snips = []
    for i in range(0, len(snippets)):
        result = eval_expr(snippets[i])
        # print(result)
        if result != None:
            successful_snips.append(result)
            executed += 1
        else:
            failed += 1
    end_time = time.time()
    successful_snips = list(filter(None, successful_snips)) # just in case
    print(f"Time taken: {round((end_time - start_time), 2)} secs")
    print(f"Total: {len(snippets)}\tExecuted: {executed}\tFailed: {failed}")
    # Save results
    out_dict = dict(successful_snips)
    df_store = DataframeStore(out_dict)
    pickle.dump(df_store, open("py_dfs.pkl", "wb"))
    # test_dict = pickle.load(open("py_dfs.pkl", "rb")).pairs
    # print(print_full(test_dict["df.drop('Survived', axis=1, inplace=True)"]))

if __name__ == '__main__':
    test_pyexec()