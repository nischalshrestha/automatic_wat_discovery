
import time
import multiprocessing
import pickle
from random import shuffle
import pandas as pd
import numpy as np
import sys
sys.path.append("../")
from generate import generate_args, generate_simple_arg, generate_args_from_df

NUM_WORKERS=12
ARGS_PICKLE_PATH = "../files/args.pkl"
PY_PICKLE_PATH = "../files/py_dfs.pkl"
PYSNIPS_PATH = "pysnips.csv"
NUM_ARGS = 1 # the default number of arguments (dataframes) to generate as inputs
MAX_ARGS = 256 # the max number of arguments
# this is the type of outputs we want to store when executing snippets so that
# we can control the number of execution results that are stored; None by default
# meaning we accept all types of outputs except for errors.
OUTPUT_TYPE_FILTER = None 
# However, we want these types in general if we aren't specifically filtering for a type
OUTPUT_TYPES = ['DataFrame', 'Series', 'int', 'int64', 'float', 'float64', 'str', 'list', 'tuple']

flatten = lambda l: [item for sublist in l for item in sublist]

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def eval_expr(df, expr):
    """
    Evals an expression given a dataframe given a dataframe and an expression.

    Note: The mslacc argument will be available to the Python `eval` when executing
    the expr.

    Currently, this does not factor in args for expr
    """
    code_str = f"out = {expr}"
    try:
        code_obj = compile(code_str, '', 'single')
        eval(code_obj)
        output = locals()['out']
        # If the output is None then grab the mslacc (for now)
        # Note: need to be careful in analysis since it doesn't mean the expr
        # actually produced a dataframe; TODO a solution is to add another meta data
        # indicating that the expr had returned a NULL.
        if output is None:
            output = locals()['df']
        return expr, df, output
        print('out', out)
    except Exception as e:
        # print(e)
        return e

def execute_statement(snip):
    """
    Given a Python snippet (one-liner), this executes it against the generated
    argument which are lists of different dataframes.
    """
    test_results = []
    for i, arg in enumerate(generated_args):
        result = eval_expr(arg, snip)
        if type(result) == tuple:
            output = result[2]
            if OUTPUT_TYPE_FILTER != None:
                if type(output) == OUTPUT_TYPE_FILTER:
                    test_results.append(result)
                else:
                    return None
            elif type(output).__name__ in OUTPUT_TYPES:
                if type(output) == tuple:
                    output = list(output)
                test_results.append(result)
            else:
                return None
        elif type(result) == Exception:
            err = str(result)
            test_results.append("ERROR: "+err)
        else:
            return None
            # For now throwing out the ones where there was an Exception
            # return None
    # print(snip, test_results)
    rtn = {'expr': snip, 'test_results': test_results}
    return rtn

class DataframeStore:
    """Just a class to store a dict of <snippet, output> so it can be pickled"""
    pairs = {}
    def __init__(self, pairs):
        self.pairs = pairs
    def get_output(self):
       return self.pairs

def execute_statements():
    """Execute python snippets with random dataframes"""
    # Read python snippets generated by filer.py
    snips = pd.read_csv(PYSNIPS_PATH)
    snippets = flatten(snips.values)
    # Eval expressions and collect successful ones paired with output: (expr, output)
    start_time = time.time()
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results = pool.map_async(execute_statement, snippets)
        results.wait()
        result = results.get()
    end_time = time.time()
    filtered = list(filter(None, result))
    print(f"Total snips executed: {len(filtered)}")
    print(f"Time taken: {round((end_time - start_time), 2)} secs")
    # For ~6.6K snippets:
    # Time taken: 1.05 secs
    return filtered

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
            # print(generated_args[0])
            executions = execute_statements()
            df_store = DataframeStore(executions)
            pickle.dump(df_store, open(PY_PICKLE_PATH, "wb"))
        except Exception as e:
            print("invalid command!", e)
            print("usage: python execute.py [all | dataframe | series | array]")
            sys.exit(1)
    else:
        print("invalid command!")
        print("usage: python execute.py [all | dataframe | series | array]")
        sys.exit(1)

