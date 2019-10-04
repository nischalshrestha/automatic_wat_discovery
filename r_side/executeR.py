import time
import multiprocessing
import pickle
from random import shuffle
import pandas as pd
import numpy as np
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from rpy2.robjects.packages import importr

import sys, os
sys.path.append("../")
sys.stderr = open(os.devnull, 'w') # silences all errors

from generate import generate_args, generate_simple_arg, generate_args_from_df

# importr("base")
# importr("dplyr")

NUM_WORKERS = 4
ARGS_PICKLE_PATH = "../files/args.pkl"
R_PICKLE_PATH = '../files/r_dfs.pkl'
RSNIPS_PATH = "rsnips.csv"
NUM_ARGS = 1 # the default number of arguments (dataframes) to generate as inputs
MAX_ARGS = 256 # the max number of arguments
# this is the type of outputs we want to store when executing snippets so that
# we can control the number of execution results that are stored; None by default
# meaning we accept all types of outputs except for errors.
OUTPUT_TYPE_FILTER = None 
# However, we want these types in general if we aren't specifically filtering for a type
OUTPUT_TYPES = ['DataFrame', 'Series', 'int', 'int64', 'float', 'float64', 'str', 'ndarray', 'list']

flatten = lambda l: [item for sublist in l for item in sublist]

class DataframeStore:
    """Just a class to store a dict of <snippet, output> so it can be pickled"""
    pairs = {}
    def __init__(self, pairs):
        self.pairs = pairs
    def get_output(self):
       return self.pairs
    
def eval_expr(df, expr):
    """
    Evals an R expression given a dataframe.
    Currently, this does not factor in args for expr
    """
    try:
        # Introduce df as mslacc into global env
        robjects.globalenv['mslacc'] = df
        # Rpy2 when evaluating using 'r', returns _something_ no matter what
        # even if expr is just an assignment
        output = robjects.r(f"""library(dplyr); {expr}""")
        # If the output is a NULL it means expression doesn't return anything.
        # For e.g. when setting a column to NULL and deleting it
        # In such a case, simply return the original dataframe as the output
        # Note: need to be careful in analysis since it doesn't mean the expr
        # actually produced a dataframe; TODO a solution is to add another meta data
        # indicating that the expr had returned a NULL.
        if type(output) == rpy2.rinterface.NULLType:
            output = robjects.globalenv['mslacc']
        # print(expr, type(output))
        robjects.r("rm(list = ls())") # clear locals after execution
        return expr, output
    except Exception as e:
        # print(expr, e)
        return e
    
def execute_statement(snip):
    """
    Given a R snippet (one-liner), this executes it against the generated
    argument which are lists of different dataframes.
    """
    test_results = []
    for i, arg in enumerate(generated_args):
        result = eval_expr(arg, snip)
        if type(result) == tuple:
            output = result[1]
            if OUTPUT_TYPE_FILTER != None:
                if type(output) == OUTPUT_TYPE_FILTER:
                    test_results.append(output)
                else:
                    return None
            elif type(output).__name__ in OUTPUT_TYPES:
                test_results.append(output)
            else:
                return None
        elif type(result) == Exception:
            err = str(result)
            test_results.append("ERROR: "+err)
        else:
            return None
    rtn = {'expr': snip, 'test_results': test_results, 'args': generated_args}
    return rtn
    
def execute_statements():
    """Execute R snippets with random dataframes"""
    # Read python snippets generated by filer.py
    snips = pd.read_csv(RSNIPS_PATH)
    snippets = flatten(snips.values)
    # Eval expressions and collect successful ones paired with output: (expr, output)
    start_time = time.time()
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map_async(execute_statement, snippets, chunksize=len(snippets)//4)
        results.wait()
        result = results.get()
    end_time = time.time()
    filtered = list(filter(None, result))
    print(f"Total snips executed: {len(filtered)}")
    print(f"Time taken: {round((end_time - start_time), 2)} secs")
    return filtered

def print_full(x):
    try:
        pd.set_option('display.max_rows', len(x))
        print(x)
        pd.reset_option('display.max_rows')
    except:
        pass

if __name__ == '__main__':
    if len(sys.argv) > 1:
        try:
            # If user specifies, the particular type of outputs to store
            # to reduce number of executions
            filter_for = sys.argv[1]
            if filter_for == "dataframe":
                OUTPUT_TYPE_FILTER = pd.DataFrame
            elif filter_for == "series":
                OUTPUT_TYPE_FILTER = pd.Series
            elif filter_for == "array":
                OUTPUT_TYPE_FILTER = np.ndarray
            elif filter_for == "all":
                pass
            else:
                raise Exception("invalid data type to filter!")
            generated_args = pickle.load(open(ARGS_PICKLE_PATH, "rb"))
            print(generated_args[0])
            executions = execute_statements()
            df_store = DataframeStore(executions)
            pickle.dump(df_store, open(R_PICKLE_PATH, "wb"))
        except Exception as e:
            print("invalid command!", e)
            print("usage: python executeR.py [all | dataframe | series | array]")
            sys.exit(1)
    else:
        print("invalid command!")
        print("usage: python executeR.py [all | dataframe | series | array]")
        sys.exit(1)

