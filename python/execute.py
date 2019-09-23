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

from generate import generate_args

NUM_WORKERS=4
PICKLE_PATH = '/Volumes/TarDisk/snippets/'
PYSNIPS_PATH = 'pysnips.csv'
generated_args = generate_args(100)

flatten = lambda l: [item for sublist in l for item in sublist]

def eval_expr(mslacc, expr):
    """
    Evals an expression given a dataframe.
    Currently, this does not factor in args for expr
    """
    # df = "mslacc = pd.read_csv('../train.csv')"
    code_str = f"out = {expr}"
    # print('mslacc' in locals())
    # Need to handle drop() cases since inplace=True returns None
    # Just capture it by reassigning df to out again
    # TODO return a tuple of snippet, output/error
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
        return e

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
    
def execute_statement(snip):
    test_results = []
    for i, arg in enumerate(generated_args):
        result = eval_expr(arg, snip)
        if type(result) == tuple:
            test_results.append(result[1])
            # executed += 1
        else:
            err = str(result)
            
            if "not defined" in str(result) \
                or "not contained" in err \
                or "no attribute" in err \
                or "does not exist" in err: 
                return snip, ["ERROR: "+str(result)]
            test_results.append("ERROR: "+str(result))
            # failed += 1
    return snip, test_results
    
def execute_statements():
    """Execute python snippets with random dataframes"""
    # Read python snippets generated by filer.py
    snips = pd.read_csv(PYSNIPS_PATH)
    snippets = flatten(snips.values)[0:5]
    # Eval expressions and collect successful ones paired with output: (expr, output)
    start_time = time.time()
    failed, executed = 0, 0
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map_async(execute_statement, snippets)
        results.wait()
        result = dict(results.get())
    end_time = time.time()
    # successful_snips = list(filter(None, successful_snips)) # just in case
    print(f"Time taken: {round((end_time - start_time), 2)} secs")
    return result

def test_pyexec():
    import sys
    # Save results
    executions = execute_statements()
    df_store = DataframeStore(executions)
    pickle.dump(df_store, open(PICKLE_PATH+"py_dfs.pkl", "wb"))
    # testing if we are able to load the pickle in memory: we're good for now
    # test_dict = pickle.load(open(PICKLE_PATH+"py_dfs.pkl", "rb")).pairs
    # print(test_dict["mslacc.drop(['Fare'],1,inplace=True)"][0])

if __name__ == '__main__':
    test_pyexec()
