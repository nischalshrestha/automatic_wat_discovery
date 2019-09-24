
import pandas as pd
import pickle

from executeR import DataframeStore

PICKLE_PATH = '/Volumes/TarDisk/snippets/'

if __name__ == '__main__':
    pydict = pickle.load(open(PICKLE_PATH+"r_dfs.pkl", "rb")).pairs
    # print(type(pydict))
    print(pydict["mslacc%>%head(10)"])