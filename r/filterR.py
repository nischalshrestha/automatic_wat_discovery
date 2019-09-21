import os, sys
import time
import multiprocessing
import nbformat
import pandas as pd

import ast
from rast import *

NUM_WORKERS = 4
file_list = [f.rstrip() for f in open("../filelist_rnb.txt", "r").readlines()]

def clean(lines, sep):
    """
    Cleans given source code, keeping only snippets and ignoring commented
    lines
    """
    result = []
    for l in lines.split("\n"):
        i = l.find(sep)
        if i >= 0:
            l = l[:i]
        if l != "":
            result.append(l.lstrip())
    cleaned = "\n".join(result)
    return cleaned

def filter_code_lines(fname):
    """
    This function processes each line of a R file (.r) and is much faster?
    than processing .irnb
    """
    # print(fname)
    snippets = []
    with open(fname, 'r') as f:
        for l in f.readlines():
            snippet = clean(l, "#")
            if snippet != "":
                # print(snippet)
                try:
                    valid = check_r(snippet)
                    if valid and snippet not in snippets:
                        snippets.append(snippet)
                        # print(snippet, '\n', valid)
                except:
                    # print(e, snippet)
                    pass
    return snippets

def filter_code_cells(fname):
    """
    This function can be used to filter code cells from .irnb files. It 
    currently cleans up and checks for select expressions (see rast.py)
    """
    print(fname)
    nb = nbformat.read("../"+fname, as_version=nbformat.NO_CONVERT)
    cells = nb.cells
    snippets = []
    failed = 0
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "source" in c:
            # cells will require further cleaning like removing comments
            source = c["source"]
            # if file is using data.table don't consider snippets of this file
            if source != None:
                cleaned = clean(source, "#").splitlines()
                # print(cleaned)
                for snippet in cleaned: # process line by line
                    if "data.table" in snippet: 
                        # print('file is using data.table')
                        return 0, []
                    try:
                        valid = check_r(snippet)
                        if valid and snippet not in snippets:
                            snippets.append(snippet)
                            # print(snippet, '\n', valid)
                        else:
                            failed += 1
                    except:
                        # print('issue parsing code')
                        pass
    return failed, snippets

flatten = lambda l: [item for sublist in l for item in sublist]

start_time = time.time()

with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
    results = pool.map_async(filter_code_cells, file_list)
    results.wait()
    result = results.get()
    result = list(zip(*result))
    failed = sum(result[0])
    all_snippets = flatten(list(result[1]))
 
end_time = time.time()
 
print(f"Time taken: {round((end_time - start_time), 2)} secs")
print(f"Parsed snippets: {len(all_snippets)} Failed snippets: {failed}")

df = pd.DataFrame(list(set(all_snippets)), columns=["snippets"])
df.to_csv("rsnips.csv", index=False)