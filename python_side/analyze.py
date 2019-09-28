
import numpy as np
import pandas as pd
import pickle

from execute import DataframeStore

PICKLE_PATH = '/Volumes/TarDisk/snippets/'

if __name__ == '__main__':
    pysnips = pickle.load(open(PICKLE_PATH+"py_dfs.pkl", "rb")).pairs
    print(len(pysnips))
    # print(type(pydict))
    # print(pydict["mslacc.drop(['Fare'],1,inplace=True)"])
    # print(test_dict["mslacc.drop(['Fare'],1,inplace=True)"][0])
    # TODO gather stats on the returned type for each expression's output(s)
    uniques = set()
    for k in pysnips:
        expr = k['expr']
        out = k['test_results'][0]
        # if type(v) == np.ndarray:
        if expr in uniques: break
        # if type(out) != None:
            # print(type(out))
            # uniques.add(expr)
        if type(out) == pd.DataFrame:
            print(expr)
            uniques.add(expr)
            # if type(v[i]) == pd.Series and v[i].size > 0:
            #     print(k)
            #     uniques.add(k)
    print(len(uniques))
