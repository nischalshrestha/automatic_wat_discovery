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
    for k in rsnips:
        expr = k['expr']
        out = k['test_results'][0]
        # if type(v) == np.ndarray:
        if expr in uniques: break
        if type(out) != None:
            count_results += len(k['test_results'])
            # print(type(out))
            uniques.add(expr)
        # if type(out) == pd.DataFrame:
        #     print(expr)
        #     uniques.add(expr)
            # if type(v[i]) == pd.Series and v[i].size > 0:
            #     print(k)
            #     uniques.add(k)
    print(f"snippets: {len(uniques)}")
    