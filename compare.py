import numpy as np
import pandas as pd
import time, os, sys

def df_diff(df1, df2):
    """
    This simply uses pandas `eq` to calculate difference between two dataframes.
    It's here to compare its performance versus compare_df (below)

    The semantic similarity score is also calculated differently:

    sim_score = diff dataframe / number of elements in diff dataframe

    """
    # print("~~~~")
    # print(df1)
    # print("----")
    # print(df2)
    row_diff = abs(df1.shape[0] - df2.shape[0])
    col_diff = abs(df1.shape[1] - df2.shape[1])
    diff = df1.eq(df2)
    # print("----")
    # print(f"{df1.eq(df2)}\nrow_diff: {row_diff} col_diff: {col_diff}")
    total_cells = diff.shape[0]*diff.shape[1]
    sim_score = sum(diff.sum()) / (diff.shape[0]*diff.shape[1])
    return sim_score

def compare_df(df1, df2):
    """
    Compares two dataframes and calculates the semantic similartiy score, which
    is defined as the highest similarity score of the largest common area (LCA) 
    between them. This method is more efficient than the pandas `eq` method 
    because it uses the dataframe's values directly which are numpy arrays.

    The LCA is found between the df1 and df2 and is used as a window to slide
    the smaller (column-wise) dataframe around on the bigger dataframe, counting 
    common elements for each window. A similarity score is then calculated as: 
    
    sim_score = common cells in window / total cells in window 

    Once all the similarity scores for all windows have been calculated, the average
    is returned as the overall semantic similarity score between df1 and df2.
    """
    if df1.empty or df2.empty:
        return 0
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
    # print(df1_arr, '\n---\n', df2_arr)
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
    # Store the top and bottom dataframe row/col dimensions
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
    # Stores the windows with the dimension of both top and bottom and the scores
    windows = []
    # Slide top across the bottom until hitting the edge of bottom 
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
        wl, wr, j = 0, 0, 0
        # Slide top across the cols of the bottom until hitting the edge
        while wr < bcol:
            wl = j
            wr = j + lca_col
            # If top's row was bigger, then don't need to slice bottom's row
            if trow > brow:
                curr_bottom = bottom[:, wl:wr]
            else:
                curr_bottom = bottom[wt:wb, wl:wr]
            # print('wl', wl, 'wr', wr, '\nbot\n', curr_bottom)
            # Compare current top and current bottom
            cbr, cbc = curr_bottom.shape[0], curr_bottom.shape[1]
            try:
                common = sum([curr_bottom[r][c] == curr_top[r][c] for c in range(cbc) for r in range(cbr)])
            except e:
                common = 0
                
                print("ERROR", e)
            sim_score = common / (cbr*cbc)
            windows.append(sim_score)
            # print("current window's sim_score", sim_score)
            j += 1
        # Update the trs flag if the top is taller than bottom
        if trow > brow:
            trs = lca_row + i
        # print('trs', trs)
        i += 1
        wb += 1
    # print('windows:', windows)
    overall_score = max(windows)
    # overall_score = sum(windows)/len(windows)
    return overall_score

def compare(a, b):
    """
    Given two data a and b, determine and return the semantic similarity score.
    """
    if (type(a) == int or type(a) == float) and (type(b) == int or type(b) == float) \
    or (type(a) == bool and type(b) == bool):
        if type(a) == int or type(a) == float:
            sim_score = int(a == b)
            size_diff = abs(a-b)
            # print(f'sim_score: {sim_score}, size diff: {size_diff}')
        elif type(a) == bool:
            sim_score = int(a == b)
            # print(f'sim_score: {sim_score}')
    elif type(a) == str and type(b) == str:
        s1 = set(a)
        s2 = set(b)
        unioned = len(s1.union(s2))
        sim_score = len(s1.intersection(s2)) / unioned if unioned > 0 else 0
        # print(f'sim_score: {sim_score}')
    # pandas stuff
    # Pandas output for series were saved as Series but output for R was
    # converted to ndarray so need to check for either possibility
    elif (type(a) == np.ndarray or type(a) == list) and (type(b) == np.ndarray or type(b) == list)\
        or (type(a) == pd.Series or type(a) == np.ndarray) and (type(b) == pd.Series or type(b) == np.ndarray):
        if type(a) == pd.Series:
            a = a.values
        if type(b) == pd.Series:
            b = b.values
        # for arrays, use jaccard
        s1 = set(a)
        s2 = set(b)
        unioned = len(s1.union(s2))
        sim_score = len(s1.intersection(s2)) / unioned if unioned > 0 else 0
        # print(f'sim_score: {sim_score}')
    elif type(a) == pd.DataFrame and type(b) == pd.DataFrame:
        sim_score = compare_df(a, b)
        # print(f'sim_score: {sim_score}')
    elif type(a) != type(b):
        sim_score = 0
        # print(f'not same type a={type(a).__name__}, b={type(b).__name__}: similarity is {sim_score}')
    return sim_score

if __name__ == '__main__':
    df = pd.read_csv("../train.csv")
    # Cases to cover:
    # Not the same type

    # Some primitives
    a = df['Survived']
    b = 1
    c = False
    d = 1.0
    e = 1.0
    f = df[0:2]
    g = True
    h = "hello"
    i = "hello"
    j = "hell"
    # compare(h, j)

    # Arrays:
    # sim score: in common / total
    a = [1,2,3]
    b = [1,2,3]
    c = [1,2,3,4]
    d = np.arange(1,5)
    e = df['PassengerId']
    f = e.values
    g = df['Survived']
    h = np.arange(3,8)
    # compare(d, h)

    # Dataframes:
    # Let's start by testing smaller random dataframes just based on numbers

    # identical case
    df1 = pd.DataFrame({'a': [1,2], 'b':[3,4]})
    df2 = pd.DataFrame({'a': [1,2], 'b':[3,4]})
    # compare(df1, df2)

    # identical bigger case
    df1 = pd.DataFrame({'a': [1,2,3], 'b':[4,5,6], 'c':[7,8,9]})
    df2 = pd.DataFrame({'a': [1,2,3], 'b':[4,5,6], 'c':[7,8,9]})
    # compare(df1, df2)

    # same dims but different values
    df1 = pd.DataFrame({'a': [1,2], 'b':[3,4]})
    df2 = pd.DataFrame({'a': [1,2], 'b':[4,5]})
    # compare(df1, df2)

    # row difference and df2 (bottom) is taller and wider
    df1 = pd.DataFrame({'a': [1,2,3], 'b':[4,5,6]})
    df2 = pd.DataFrame({'a': [1,2,3,4], 'b':[5,6,7,8]})
    # compare(df1, df2)

    # col and row difference and df2 (bottom) is taller and wider
    df1 = pd.DataFrame({'a': [1,2,3], 'b':[4,5,6]})
    df2 = pd.DataFrame({'a': [1,2,3,4], 'b':[5,6,7,8], 'c':[9,10,11,12]})
    # compare(df1, df2)

    # col and row difference when the df2 (bottom) is shorter and wider
    df1 = pd.DataFrame({'a': [1,2,3], 'b':[4,5,6]})
    df2 = pd.DataFrame({'a': [3,2], 'b':[6,5], 'c':['a','b'], 'd':[6,5]})
    # compare(df1, df2)

    # Now for real world-y dataframes
    df = pd.read_csv("../train.csv")
    a = df.iloc[:2, :2]
    b = df.iloc[:2, :2] # exact
    c = df.query('Survived == 1').iloc[:2, :2] # same shape but different values
    d = df.iloc[:4, :5] # both rol/col different
    e = df.iloc[:2, :5] # only col different
    # compare(a, c)

    # # Comparing the two methods score and performance
    # start = time.time()
    # print(f"overall score (compare_df): {compare_df(df1, df2)}")
    # # df_diff(df1, df2)
    # print("time taken: %.6fs" % (time.time()-start))

    # start = time.time()
    # # print('overall score:', compare_df(df1, df2))
    # print(f"overall score (pandas.eq): {df_diff(df1, df2)}")
    # print("time taken: %.6fs" % (time.time()-start))

