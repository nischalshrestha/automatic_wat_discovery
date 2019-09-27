
import numpy as np
import pandas as pd
import pickle
import rpy2
from rpy2.robjects import pandas2ri
pandas2ri.activate()

from executeR import DataframeStore

PICKLE_PATH = '/Volumes/TarDisk/snippets/'

if __name__ == '__main__':
    rdict = pickle.load(open(PICKLE_PATH+"r_dfs.pkl", "rb")).pairs
    # print(type(rdict))
    # TODO check if all ndarrays were actually supposed to be Series; there might
    # have been a lost in translation for vectors (it might need to be explicit)
    uniques = set()
    for k, v in rdict.items():
        # if type(v) == np.ndarray:
        for i in range(len(v)):
            if k in uniques: break
            # if type(v[i]) != rpy2.rinterface.NULLType:
            #     print(type(v[i]))
            #     uniques.add(k)
            # if type(v[i]) == pd.Series and not v[i].empty:
            #     print(k)
            #     uniques.add(k)
            # if type(v[i]) == pd.DataFrame and not v[i].empty:
            #     print(k)
            #     uniques.add(k)
            if type(v[i]) == np.ndarray and v[i].size > 0:
                print(k)
                uniques.add(k)

    print(len(uniques))
    # print(rdict["mslacc$Sex[which(mslacc$Sex==\"female\")]<-1"])
    