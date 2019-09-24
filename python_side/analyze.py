
import numpy as np
import pandas as pd
import pickle

from execute import DataframeStore

PICKLE_PATH = '/Volumes/TarDisk/snippets/'

if __name__ == '__main__':
    pydict = pickle.load(open(PICKLE_PATH+"py_dfs.pkl", "rb")).pairs
    # print(type(pydict))
    # print(pydict["mslacc.drop(['Fare'],1,inplace=True)"])
    # print(test_dict["mslacc.drop(['Fare'],1,inplace=True)"][0])
    uniques = set()
    for k, v in pydict.items():
        # if type(v) == np.ndarray:
        for i in range(len(v)):
            if k in uniques: break
            if type(v[i]) != None:
                print(type(v[i]))
                uniques.add(k)
            # if type(v[i]) == pd.DataFrame and not v[i].empty:
            #     print(k)
            #     uniques.add(k)
            # if type(v[i]) == pd.Series and v[i].size > 0:
            #     print(k)
            #     uniques.add(k)
    print(len(uniques))
