"""
This module has functions to traverse Notebooks, prettify them or examine code
cells. It can be extended to perform other operations on the code cells.
""" 

from pathlib import Path
import os
import re
import nbformat

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

def print_code_cells(cells):
    """
    This function can be used to print cleaned up code cells
    """
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "source" in c:
            # cells will require further cleaning like removing comments
            source = c["source"]
            if source != None:
                cleaned = clean(c["source"], "#")
                # print(cleaned)
                
def parse_notebooks(lang="r", snippet_type="all"):
    """
    Parses notebooks for r or python on Notebooks directory.
    
    Can be extended to more things.
    """
    counter = 0
    files = []
    for entry in os.scandir("Notebooks/"+lang):
        # go through each user folder
        for file in os.scandir(entry.path):
            # go through the project folder that holds the notebook
            project = entry.path + "/" + file.name
            for file in os.scandir(project):
                if "kernel-metadata.json" not in project and not os.path.isdir(file):
                    with open(file, "r") as f:
                        # only look at files that are notebooks or .py
                        if lang == "r" and ".R" in f.name:
                            print(counter)
                            counter += 1
                            # nb = nbformat.read(f.name, as_version=nbformat.NO_CONVERT)
                            # cells = nb.cells
                            # do something with cells
                            # print_code_cells(cells)
                            files.append(f.name)
                        if lang == "py" and ".py" in f.name:
                            counter += 1
                            print(counter)
                            # nb = nbformat.read(f.name, as_version=nbformat.NO_CONVERT)
                            # cells = nb.cells
                            # do something with cells
                            # print_code_cells(cells)
                            files.append(f.name)
    return files, counter

def prettify_notebook(notebook):
    """Prettify given notebook filename"""
    # print(notebook)
    with open(notebook, "r") as f:
        script = json.load(f)
        pretty = json.dumps(script, indent=4)
        #info = json.loads(js.decode("utf-8"))
        return pretty

def prettify_dirs(lang="r"):
    """Prettify all notebooks for given lang"""
    for entry in os.scandir("Notebooks/"+lang):
        print(entry.path)
        for file in os.scandir(entry.path):
            if "kernel-metadata.json" not in (entry.path + file.name) and (file.name.endswith('.ipynb') or file.name.endswith('.irnb')):
                pretty = prettify_notebook(entry.path + "/" + file.name)
                with open(entry.path + "/" + file.name, "w") as f:
                    f.write(pretty)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        language = sys.argv[1]
        if language == "py" or "r":
            snippet_type = "all"
            # prettify_dirs("python")
            files, total = parse_notebooks(language, snippet_type)
            print(total, len(files))
            with open(f'files/filelist_{language}.txt', 'w') as file:
                for f in files:
                    file.write(f+'\n')

