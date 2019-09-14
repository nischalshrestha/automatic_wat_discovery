import time
import multiprocessing
import nbformat
import pandas as pd

import ast
from pyast import ASTChecker

NUM_WORKERS = 4
file_list = [f.rstrip() for f in open("filesnb.txt", "r").readlines()]

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
    nb = nbformat.read(fname, as_version=nbformat.NO_CONVERT)
    cells = nb.cells
    snippets = []
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "source" in c:
            # cells will require further cleaning like removing comments
            source = c["source"]
            if source != None:
                cleaned = clean(source, "#").splitlines()
                for snippet in cleaned: # process line by line
                    # print(cleaned)
                    try:
                        checker = ASTChecker(ast.parse(snippet))
                        valid = checker.check()
                        if valid and snippet not in snippets:
                            snippets.append(snippet)
                            # print(snippet, '\n', valid)
                    except:
                        # print('issue parsing code')
                        pass
    return snippets

flatten = lambda l: [item for sublist in l for item in sublist]

# start_time = time.time()
# all_snippets = [] 
# for file in file_list:
#     all_snippets.append(filter_code_cells(file))

# all_snippets = flatten(all_snippets)
# print(len(all_snippets))

# end_time = time.time()    
 
# print("Time for SerialSquirrel: %ssecs" % (end_time - start_time))

start_time = time.time()

with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
    results = pool.map_async(filter_code_cells, file_list)
    results.wait()
    all_snippets = flatten(results.get())
 
end_time = time.time()
 
print("Time for MultiProcessingSquirrel: %ssecs" % (end_time - start_time))

df = pd.DataFrame(list(set(all_snippets)), columns=["snippets"])
df.to_csv("pythonsnips.csv", index=False)