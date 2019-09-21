"""
This module filters Python notebooks to accept certain expressions and stores 
it in a csv file for later use.
"""

import time
import multiprocessing
import nbformat
import pandas as pd

import ast
from pyast import ASTChecker, Normalizer

NUM_WORKERS = 4
file_list = [f.rstrip() for f in open("../filelist_pynb.txt", "r").readlines()]

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
    This function processes each line of a Python file (.py) and is much faster
    than processing .ipynb
    """
    # print(fname)
    snippets = []
    with open(fname, 'r') as f:
        for l in f.readlines():
            snippet = clean(l, "#")
            if snippet != "":
                # print(snippet)
                try:
                    checker = ASTChecker(ast.parse(snippet))
                    valid = checker.check()
                    if valid and snippet not in snippets:
                        snippets.append(snippet)
                        # print(snippet, '\n', valid)
                except Exception as e:
                    # print(e, snippet)
                    pass
    return snippets

def filter_code_cells(fname):
    """
    This function can be used to filter code cells from .ipynb files. It 
    currently cleans up and checks for select ASTs (see pyast.py)
    """
    # print(fname)
    nb = nbformat.read("../"+fname, as_version=nbformat.NO_CONVERT)
    cells = nb.cells
    snippets = []
    excluded = 0
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "source" in c:
            # cells will require further cleaning like removing comments
            source = c["source"]
            if source != None:
                cleaned = clean(source, "#").splitlines()
                for snippet in cleaned: # process line by line
                    # print(cleaned)
                    try:
                        tree = ast.parse(snippet)
                        checker = ASTChecker(tree)
                        valid = checker.check()
                        if valid and snippet not in snippets:
                            n = Normalizer(tree)
                            normalized = n.normalize().strip()
                            snippets.append(normalized)
                            # print(snippet, '\n', valid)
                        else:
                            excluded += 1
                    except:
                        # print('issue parsing code')
                        pass
    return excluded, snippets

flatten = lambda l: [item for sublist in l for item in sublist]

start_time = time.time()

with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
    results = pool.map_async(filter_code_cells, file_list)
    results.wait()
    result = results.get()
    result = list(zip(*result))
    excluded = sum(result[0])
    all_snippets = list(set(flatten(result[1])))
 
end_time = time.time()
 
print(f"Time taken: {round((end_time - start_time), 2)} secs")
print(f"Parsed snippets: {len(all_snippets)} Excluded snippets: {excluded}")

df = pd.DataFrame(all_snippets, columns=["snippets"])
df.to_csv("pysnips.csv", index=False)