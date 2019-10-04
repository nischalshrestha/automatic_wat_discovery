
import time
import multiprocessing
import pickle
from random import shuffle
import pandas as pd
import numpy as np
import sys
sys.path.append("../")
from generate import generate_args, generate_simple_arg, generate_args_from_df

NUM_WORKERS=4
PY_PICKLE_PATH = '../files/py_dfs.pkl'
PYSNIPS_PATH = 'pysnips.csv'
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

def eval_expr(mslacc, expr):
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
            output = locals()['mslacc']
        return expr, output
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
            output = result[1]
            if OUTPUT_TYPE_FILTER != None:
                if type(output) == OUTPUT_TYPE_FILTER:
                    test_results.append(output)
                else:
                    return None
            elif type(output).__name__ in OUTPUT_TYPES:
                if type(output) == tuple:
                    output = list(output)
                test_results.append(output)
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
            NUM_ARGS = int(sys.argv[1]) 
            NUM_ARGS = 1 if NUM_ARGS <= 0 else NUM_ARGS
            if NUM_ARGS > MAX_ARGS: 
                print("beyond max arguments of 256 inputs")
                sys.exit(1)
            # If user specifies, the particular type of outputs to store
            # to reduce number of executions
            if len(sys.argv) > 2:
                if "dataframe" in sys.argv[2]:
                    OUTPUT_TYPE_FILTER = pd.DataFrame
                elif "series" in sys.argv[2]:
                    OUTPUT_TYPE_FILTER = pd.Series
                elif "array" in sys.argv[2]:
                    OUTPUT_TYPE_FILTER = np.ndarray
            # generated_args = generate_args(NUM_ARGS)
            # generated_args = generate_simple_arg()
            ints = [i for i in range(0, 8)]
            ints.extend([8,8])
            sints = [i for i in range(0, 10)]
            # shuffle(sints)
            strs = [f"ID_{i}" for i in range(0, 8)]
            strs.extend(["ID_8", "ID_8"])
            # sstrs = [f"P_{i}" for i in reversed(range(10))]
            df = pd.DataFrame({'col0':sints[::-1], 'col1':ints, 'col2':strs, 'col3':ints})
            # shuffle(df)
            # print(df)
            # TODO: user supplies argument to switch from single to multiple random dfs
            # generated_args = generate_args_from_df(df)
            generated_args = generate_args_from_df(df, n_args=NUM_ARGS, simple=False)
            print(generated_args[0])
            executions = execute_statements()
            # Save results
            df_store = DataframeStore(executions)
            pickle.dump(df_store, open(PY_PICKLE_PATH, "wb"))
            # TODO: maybe move generation functionality to generate module only
            pickle.dump(generated_args, open("../files/args.pkl", "wb"))
        except Exception as e:
            print(e)
            print("invalid option!")
            print("usage: python execute.py [number of inputs to test <= 256] [(dataframe | series | array)]")
            sys.exit(1)
    else:
        print("invalid option!")
        print("usage: python execute.py [number of inputs to test <= 256] [(dataframe | series | array)]")
        sys.exit(1)

