"""
This module is used to analyze the result of the execute module
"""

import numpy as np
import pandas as pd
import pickle

from execute import DataframeStore

PICKLE_PATH = '../files/'

if __name__ == '__main__':
    pysnips = pickle.load(open(PICKLE_PATH+"py_dfs.pkl", "rb")).pairs
    print(len(pysnips))
    # print(type(pydict))
    # print(pydict["mslacc.drop(['Fare'],1,inplace=True)"])
    # print(test_dict["mslacc.drop(['Fare'],1,inplace=True)"][0])
    # TODO gather stats on the returned type for each expression's output(s)
    types = []
    uniques = set()
    count_results = 0
    count_errors = 0
    for k in pysnips:
        expr = k['expr']
        out = k['test_results'][0]
        # if type(v) == np.ndarray:
        if expr in uniques: break
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


