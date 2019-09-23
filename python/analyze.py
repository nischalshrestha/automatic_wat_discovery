
import pandas as pd
import pickle

from execute import DataframeStore

PICKLE_PATH = '/Volumes/TarDisk/snippets/'

if __name__ == '__main__':
    pydict = pickle.load(open(PICKLE_PATH+"py_dfs.pkl", "rb")).pairs
    # print(type(pydict))
    print(pydict["mslacc.drop(['Fare'],1,inplace=True)"])
    # print(test_dict["mslacc.drop(['Fare'],1,inplace=True)"][0])
