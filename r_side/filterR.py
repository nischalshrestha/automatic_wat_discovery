"""
This module filters R notebooks or .r files to accept certain expressions and stores 
it in a csv file for later use by the executeR module.
"""

import os, sys
import time
import multiprocessing
import nbformat
import pandas as pd

import ast
from rast import *

NUM_WORKERS = multiprocessing.cpu_count()
R_NOTEBOOK_LIST = "../files/filelist_rnb.txt"
R_LIST = "../files/filelist_r.txt"
EXPERIMENT_LIST = "../experiments/filelist_r.txt"

flatten = lambda l: [item for sublist in l for item in sublist]

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

def filter_code_lines(fname, base="../"):
    """
    This function processes each line of a R file (.r) and is much faster?
    than processing .irnb
    """
    # print(fname)
    snippets = []
    excluded = 0
    with open(base+fname, 'r') as f:
        for l in f.readlines():
            snippet = clean(l, "#")
            if snippet != "":
                # print(snippet)
                if "data.table" in snippet: 
                        # print('file is using data.table')
                        return 0, []
                try:
                    valid = check_r(snippet)
                    if valid and snippet not in snippets:
                        # normalized = normalize(snippet)
                        # snippets.append(normalized)
                        snippets.append(snippet)
                        # print(snippet, '\n', valid)
                    else:
                        excluded += 1
                        # print(snippet, '\n', valid)
                except Exception as e:
                    # print(e, snippet)
                    pass
    return excluded, snippets

def filter_code_cells(fname, base="../"):
    """
    This function can be used to filter code cells from .irnb files. It 
    currently cleans up and checks for select expressions (see rast.py)
    """
    # print(base+fname)
    nb = nbformat.read(base+fname, as_version=nbformat.NO_CONVERT)
    cells = nb.cells
    snippets = []
    excluded = 0
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
                            normalized = normalize(snippet)
                            snippets.append(normalized)
                            # print(snippet, '\n', valid)
                        else:
                            excluded += 1
                            # print(snippet, '\n', valid)
                    except Exception as e:
                        # print('issue parsing code',e)
                        pass
    return excluded, snippets

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        language = sys.argv[1]
        if language == "notebook":
            file_list = [f.rstrip() for f in open(R_NOTEBOOK_LIST, "r").readlines()]
            filter_func = filter_code_cells
        elif language == "script":
            file_list = [f.rstrip() for f in open(R_LIST, "r").readlines()]
            filter_func = filter_code_lines
        elif language == "experiments":
            file_list = [f.rstrip() for f in open(EXPERIMENT_LIST, "r").readlines()]
            filter_func = filter_code_lines
        else:
            print(f"Invalid option {sys.argv[1]}, please enter either 'notebook' or 'script'")
            sys.exit(1)
        # Parellelize the file processing since each one is independent
        # Time taken for filtering scripts: 44.96 secs using 4 processes
        start_time = time.time()
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            filter_func = filter_code_lines
            results = pool.map_async(filter_func, file_list)
            results.wait()
            result = results.get()
            result = list(zip(*result))
            failed = sum(result[0])
            all_snippets = flatten(list(result[1]))
            all_snippets = list(set(all_snippets))
        end_time = time.time()
        print(f"Time taken: {round((end_time - start_time), 2)} secs")
        print(f"Parsed snippets: {len(all_snippets)} Excluded snippets: {failed}") 
        # Using current data and filtering: 
        # Parsed snippets: 1013 Failed snippets: 11991
        df = pd.DataFrame(all_snippets, columns=["snippets"])
        df.to_csv("rsnips.csv", index=False)

        