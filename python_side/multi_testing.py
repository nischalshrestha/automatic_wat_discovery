
import numpy as np
import pandas as pd
import time
import pickle
import multiprocessing
from multiprocessing import Queue
import sys
sys.path.append("../")

from compare import compare

from generate import generate_args_from_df
from execute import DataframeStore

NUM_WORKERS = 4
PICKLE_PATH = '/Volumes/TarDisk/snippets/'
# pysnips = pickle.load(open(PICKLE_PATH+"py_dfs.pkl", "rb")).pairs
rsnips = pickle.load(open(PICKLE_PATH+"r_dfs.pkl", "rb")).pairs

colors = ['red', 'green', 'blue', 'black']

cnt = 1
# instantiating a queue object
queue = Queue()
print('pushing items to queue:')
# for color in colors:
#     print('item no: ', cnt, ' ', color)
#     queue.put(color)
#     cnt += 1
start = time.time()
for r in rsnips:
    # print('item no: ', cnt, ' ', r)
    queue.put(r)
    cnt += 1
print('\npopping items from queue:')
cnt = 0
while not queue.empty():
    print('item no: ', cnt, ' ', queue.get())
    # queue.get()
    cnt += 1
print(time.time()-start)