import numpy as np
import pandas as pd
import time
import pickle
import sys
sys.path.append("../")

from generate import generate_args_from_df
from execute import DataframeStore

# PICKLE_PATH = '/Volumes/TarDisk/snippets/'

# # For now can only focuse on dataframe/series outputs

# # PassengerId      int64
# # Survived         int64 (level) - randomize
# # Pclass           int64 (level) - randomize
# # Name            object 
# # Sex             object (level) - randomize
# # Age            float64
# # SibSp            int64
# # Parch            int64
# # Ticket          object (level) - randomize
# # Fare           float64
# # Cabin           object (level) - randomize
# # Embarked        object (level) - randomize
# df = pd.read_csv("../train.csv")

# # Comparison procedure
# # Input: pydata, rdata
# # Check types
# #   If same type, perform comparison for that data type
# #       Calculate similarity score between the two snippets for that particular execution
# #   If not same type, no need for comparison and similarity is 0
# #   Continually update the snippet's similarity score based on:
# #       mean of all the similarity scores for each execution comparison
# #   Restore the respective python and r data back to the pickle (with similarity score now associated)

# # Input: F - List of Functions with Input and Output 
# # Output: C - List of clusters
# # procedure Cluster(F)
# #     C←φ
# #     for all F ∈ Fdo
# #         for all C ∈ C do
# #             O ← GetRepresentive(C)
# #             if Similarity(O,F) ≥ SIM_T then
# #                 C←C∪F
# #                 break
# #         if ∀C ∈ C, F not in C then
# #             C|C|+1 ←F SetRepresentative(C|C|+1,F)
# #             C←C∪C return C

# # Cases to cover:
# # Not the same type
# a = df['Survived']
# b = 1
# c = False
# if type(a) != type(b):
#     print(f'not same type a={type(a).__name__}, b={type(b).__name__}: similarity is 0')

# # Same type:
# # Scalars 

# # int/float
# # sim score: size diff
# a = 1
# b = 1.0
# b = 1.1 # diff
# # bools
# # a = True
# # b = False
# if (type(a) == int or type(a) == float) and (type(b) == float or type(b) == float)\
#     or (type(a) == bool and type(b) == bool):
#     if type(a) == int:
#         print(f'{a == b}, size diff: {abs(a-b)}')

# # Arrays:
# # sim score: in common / total
# a = [1,2,3]
# b = [1,2,3]
# b = [12,3,4,5]
# # if (type(a) == int or type(a) == float) and (type(b) == float or type(b) == float)\
# #     or (type(a) == bool and type(b) == bool):
# print(f'{a == b}, array similarity: {len(set(a) & set(b))/(len(a)+len(b))}')

# # Dataframes:
# # Either row or col is different

def df_diff(df1, df2):
    """
    This simply uses pandas `eq` to calculate difference between two dataframes.
    """
    print("~~~~")
    print(df1)
    print("----")
    print(df2)
    row_diff = abs(df1.shape[0] - df2.shape[0])
    col_diff = abs(df1.shape[1] - df2.shape[1])
    diff = df1.eq(df2)
    print("----")
    print(f"{df1.eq(df2)}\nrow_diff: {row_diff} col_diff: {col_diff}")
    total_cells = diff.shape[0]*diff.shape[1]
    sim_score = round(sum(diff.sum()) / (diff.shape[0]*diff.shape[1]), 2)
    return sim_score

def compare_df(df1, df2):
    """
    Compares two dataframes and calculates the semantic similartiy score, which
    is defined as the highest similarity score of the largest common area between 
    them. This method is more efficient than the pandas `eq` method.
    """
    # Find the largest common area between the two
    df1_row = df1.shape[0]
    df1_col = df1.shape[1]
    df2_row = df2.shape[0]
    df2_col = df2.shape[1]
    lca = (min(df1_row, df2_row), min(df1_col, df2_col))
    # Note the row and col dimension diff
    row_diff = abs(df1_row - df2_row)
    col_diff = abs(df1_col - df2_col)
    # Convert to ndarrays for a more efficient comparison
    df1_arr = df1.values
    df2_arr = df2.values
    # Make the one with more columns be bottom
    if df2_col > df1_col:
        bottom, top = df2_arr, df1_arr
    elif df2_col < df1_col:
        bottom, top = df1_arr, df2_arr
    else:
        # If tied, make the one with more rows be bottom
        if df1_row > df2_row:
            bottom, top = df1_arr, df2_arr
        else:
            bottom, top = df2_arr, df1_arr
    # print('lca', lca)
    # print('bottom\n', bottom)
    lca_row = lca[0]
    lca_col = lca[1]
    # print('lca_row', lca_row, 'lca_col', lca_col)
    # STore the top and bottom dataframe row/col dimensions
    trow, brow = top.shape[0], bottom.shape[0]
    tcol, bcol = top.shape[1], bottom.shape[1]
    # print(trow, brow, tcol, bcol)
    # i is the current LCA row and j is the current LCA col when sliding
    i, j = 0, 0
    # The sliding window's current position
    wl, wr, wt, wb = 0, 0, 0, 0
    # The top dataframe's current top to bottom 
    ttop, tbot = 0, 0
    # trs is the number of rows processed when the top is taller than the bottom
    trs = 0
    # print('start iteration')
    # Stores the windows with the dimension of both top and bottom and the scores
    windows = []
    while (i + lca_row - 1 < brow) or (trow > brow and trs < trow):  
        # Need to handle the case when top is taller so its window starts from the bottom
        if trow > brow:
            ttop = trow - lca_row - i
            tbot = trow - i
        else: 
            # Normally, we use the full size of the top to superimpose
            ttop, tbot = 0, trow
        # Current top in the LCA window
        curr_top = top[ttop:tbot, :]
        # print('row', i, '\ntop\n', curr_top)
        wt = i
        wb = i + lca_row
        # print(wt, wb)
        # Slide top across until hitting the edge of bottom 
        wl, wr, j = 0, 0, 0
        while wr < bcol:
            wl = j
            wr = j + lca_col
            curr_bottom = bottom[i:, wl:wr]
            # print('wl', wl, 'wr', wr, '\nbot\n', curr_bottom)
            j += 1
        # Update the trs flag if the top is taller than bottom
        if trow > brow:
            trs = lca_row + i
        # print('trs', trs)
        i += 1
        wb += 1
        

if __name__ == '__main__':
    # Let's start by testing smaller random dataframes just based on numbers

    # identical case
    # df1 = pd.DataFrame({'a': [1,2], 'b':[3,4]})
    # df2 = pd.DataFrame({'a': [1,2], 'b':[4,5]})

    # identical bigger case
    # df1 = pd.DataFrame({'a': [1,2,3], 'b':[4,5,6], 'c':[7,8,9]})
    # df2 = pd.DataFrame({'a': [1,2,3], 'b':[4,5,6], 'c':[7,8,9]})

    # row difference when df2 is taller and wider
    df1 = pd.DataFrame({'a': [1,2,3], 'b':[4,5,6]})
    df2 = pd.DataFrame({'a': [3,2,3,2], 'b':[6,5,6,5], 'c':['a','b','c','a'], 'd':[6,5,6,5]})

    # row difference when df2 is shorter and wider
    # df1 = pd.DataFrame({'a': [1,2,3], 'b':[4,5,6]})
    # df2 = pd.DataFrame({'a': [3,2], 'b':[6,5], 'c':['a','b'], 'd':[6,5]})

    start = time.time()
    compare_df(df1, df2)
    # df_diff(df1, df2)
    print(time.time()-start)

    # print(df.dtypes)
    # args = generate_args_from_df(df, n_args=20)
    # half = int(len(args)/2)
    # for a,b in zip(args[:half], args[half:]):
    #     df1 = a.iloc[:np.random.randint(1, a.shape[0] + 1), :np.random.randint(1, a.shape[1] + 1)]
    #     df2 = a.iloc[:np.random.randint(1, b.shape[0] + 1), :np.random.randint(1, b.shape[1] + 1)]
    #     print(df1.shape, df2.shape)
    #     df_diff(df1, df2)

    # pydict = pickle.load(open(PICKLE_PATH+"py_dfs.pkl", "rb")).pairs
    # rdict = pickle.load(open(PICKLE_PATH+"r_dfs.pkl", "rb")).pairs
    # print(len(pydict.items()), len(rdict.items()))
    # 6619 1013

