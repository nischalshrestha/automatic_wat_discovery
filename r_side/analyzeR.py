"""
This module is used to analyze the result of the executeR module
"""

import numpy as np
import pandas as pd
import pickle
import rpy2
from rpy2.robjects import pandas2ri
pandas2ri.activate()

from executeR import DataframeStore

PICKLE_PATH = '../files/'

if __name__ == '__main__':
    rsnips = pickle.load(open(PICKLE_PATH+"r_dfs.pkl", "rb")).pairs
    # print(type(rdict))
    print(len(rsnips))
    # TODO check if all ndarrays were actually supposed to be Series; there might
    # have been a lost in translation for vectors (it might need to be explicit)
    uniques = set()
    count_results = 0
    count_errors = 0
    types = []
    for k in rsnips:
        expr = k['expr']
        out = k['test_results']
        # if type(v) == np.ndarray:
        if expr in uniques: continue
        errors = 0
        for t in k['test_results']:
            if type(t) != None:
                types.append(type(t).__name__)
                uniques.add(expr)
                if type(t) == str and "ERROR:" in t:
                        # print(type(t))
                    errors += 1
                    # print(type(out))
        count_errors += errors
        # if type(out) == pd.DataFrame:
        #     print(expr)
        #     uniques.add(expr)
            # if type(v[i]) == pd.Series and v[i].size > 0:
            #     print(k)
            #     uniques.add(k)
    np_types = np.asarray(types)
    unique, counts = np.unique(np_types, return_counts=True)
    adict = dict(zip(unique, counts))
    print(f"snippets: {len(uniques)} Errors: {count_errors}")
    print(adict)
    