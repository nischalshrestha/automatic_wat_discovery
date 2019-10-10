# Kaggle Parsing

This repo contains modules to parse Kaggle jupyter notebooks/scripts from competitions like [Titanic](https://www.kaggle.com/c/titanic) in order to find similar code snippets across Python and R.

# Installation:

It's easiest to use [baker](https://docs.getbaker.io/bakerScript/basic/) which makes use of VirtualBox to quickly spin up a VM for the project. Once baker/VirtualBox is installed on your computer, simply execute the following in the root directory:

`baker bake --local .` 

This sets up a VM for the project synced with the local folder and installs Python and R including all dependencies (`requirements.txt`).

## Running with baker:

Then execute the following in the root directory:

`baker run [cmd]` 

where `cmd` runs one of the phases described below and detailed in the default commands within [baker.yml](https://github.com/nischalshrestha/kaggle_parsing/blob/master/baker.yml#L7). For example, to filter all the scripts for Python snippets, one can execute:

`baker run filterPy`

## Running modules directly:

You can also directly run the modules if you need more control or for testing.

**Within root:**

`python parseNotebooks.py [py|r]`

`python generate.py [kaggle|experiments] [-s single_dataframe | -r random_dataframes] [number of inputs to test <= 256]`

where `number of inputs` are how many arguments in total you want to produce when using `-r`. If supplying `-s`, there is 
no need for supplying an additional number; a CSV template or dataframe template is used. 

`python cluster.py SIM_T (<= 1.0) [keep]` 

where `SIM_T` is the similarity score and `keep` is an optional argument to also store test results and their scores.

To query from Python/R snippets with test cases and results, the cluster use the query.py module:

`python query.py cluster_file low_score high_score edit_score` 

where `cluster_file` is the name of the file within the files directory (.csv is not required), `low_score` is the 
lower bound of the overal similarity score, the `high_score` is the upper bound and the `edit_score` is the 
syntactic distance between the snippets. For e.g., running:

`python query.py clusters_0.3 .6 .9 .5`

outputs:

```
~~~~
df.iloc[0:5, 0:3] df[1:5, 1:3]
Row score: 0.6
Column score: 1.0
Overall score: 0.6
Edit distance: 0.794
Test case:

   col0  col1  col2  col3
0     0     6   NaN     5
1     1     3  ID_6     7
2     2     3  ID_4     7
Python output:

   col0  col1  col2
0     0     6   NaN
1     1     3  ID_6
2     2     3  ID_4
R output:

         col0        col1  col2
0           0           6   NaN
1           1           3  ID_6
2           2           3  ID_4
3 -2147483648 -2147483648   NaN
4 -2147483648 -2147483648   NaN
```

**Within `py_side`:**

`python filter.py [notebook|script]`

`python execute.py [number of inputs to test <= 256] [all | dataframe | series | array]`

where the 3rd argument specifies where you want to filter outputs; supply `all` to not filter any particular data type.

**Within `r_side`:**

`python filterR.py [notebook|script]`

`python executeR.py [number of inputs to test <= 256] [all | dataframe | series | array]`

# Execution Phases:

The modules are run in order according to the following phases:

### 1. Preparation phase
First, the Kaggle Notebooks are traversed and the file paths are gathered for both Python and R. These lists will be used in the next phase. Relevant Files:

- [parseNotebooks.py](https://github.com/nischalshrestha/kaggle_parsing/blob/master/parseNotebooks.py) to traverse Notebooks and create file path lists for both Python/R Notebooks/Scripts stored in [files](https://github.com/nischalshrestha/kaggle_parsing/tree/master/files)

### 2. Segmentation + Filter + Normalization phase
Next, both the Python/R notebooks/scripts are segmented, where each line is considered a candidate expression. These candidates are then  filtered for one-liner stand-alone expressions, discarding block expressions that span multiple lines like `if` or `def` in Python, or `function` and `for` in R. In this filtering process, each line must also fit a subset of the the grammar for each language (Python/pandas and R). Once the expressions meet these requirements, they are noramlized: 1) Dataframe variable names are renamed to `df` to standardize the dataframe variables for execution in the Execution phase 2) Whitespace within the expressions are stripped to unbias during the calculation of syntactical edit distances in the Cluster phase.

For Python, the built-in `ast` module is used to parse Python code and filter for certain expressions using the `ast.Visitor` class. The filtered expressions are then normalized using the `ast.Transformer` class. Relevant files:

- [filter.py](https://github.com/nischalshrestha/kaggle_parsing/blob/master/python_side/filter.py) filters and stores a csv file containing snippets in `python_side`
- [pyast.py](https://github.com/nischalshrestha/kaggle_parsing/blob/master/python_side/pyast.py) contains the Visitor/Transformer

For R, the `rpy2` is used in Python code to use R's `getParseData()` function to filter for one-liner expressions and that meet a small subset of grammar. Then, a custom R script is used to normalize the expressions. Relevant files:

- [filterR.py](https://github.com/nischalshrestha/kaggle_parsing/blob/master/r_side/filterR.py) filters and stores a csv file containing snippets in `r_side`
- [rast.py](https://github.com/nischalshrestha/kaggle_parsing/blob/master/r_side/rast.py)
- [varRenamer.r](https://github.com/nischalshrestha/kaggle_parsing/blob/master/r_side/varRenamer.r) used by `rast.py` to normalize

### 3. Input Generation + Execution phase
To execute Python/R snippets, inputs are generated which are dataframes based on a template csv file (for e.g. train.csv for the titanic competition). The generated dataframes are psuedo-random as column labels are preserved as well as column types. Only the values are either randomly generated within bounds or shuffled in the case of levels (for e.g. Sex as 0/1). 

Values for int/float column types are randomly generated within the min/max values of the column; string values are randomly shuffled for str column types; and some NaN values are added if any NaN existed in the template's column. For both Python/R, the [generate.py](https://github.com/nischalshrestha/kaggle_parsing/blob/master/generate.py) is used to generate arguments using `pandas` and `numpy`.

Then the Python/R snippets are executed against these generated input dataframes.

For Python, `compile` is used to compile an expression and `eval` to execute. Relevant files:

- [execute.py](https://github.com/nischalshrestha/kaggle_parsing/blob/master/python_side/execute.py) to generate, execute and store execution results into a py_dfs.pkl file in [files](https://github.com/nischalshrestha/kaggle_parsing/tree/master/files)

For R, `rpy2.robjects.globalenv` is used to introduce the argument into the embedded R's environment and `rpy2.robjects.r` is used to evaluate the expression. Relevant files:

- [executeR.py](https://github.com/nischalshrestha/kaggle_parsing/blob/master/r_side/executeR.py) to generate, execute and store execution results into a r_dfs.pkl file in [files](https://github.com/nischalshrestha/kaggle_parsing/tree/master/files)

### 4. Cluster phase
Finally, the Python/R snippets are then clustered according to output similarity score from 0 to 1. For scalars like ints/floats, a `size_diff` is calculated, for booleans the score is either 0 or 1, and for strings, the jaccard similarity score is calculated. For dataframes, the largest common area (LCA) dimension is first determined. Then the LCA is used as a window to slide the smaller dataframe over the larger to find the region with the highest cell similarity. Various different measures could be included for dataframes such as similarity by columns or rows. The edit distance between the Python and R snippets is also calculated using various measures like levenshtein or jaro. The clustered snippets are then stored in a csv file. Relevant files:

- [cluster.py](https://github.com/nischalshrestha/kaggle_parsing/blob/master/cluster.py) to cluster
- [compare.py](https://github.com/nischalshrestha/kaggle_parsing/blob/master/compare.py) for comparing outputs

## Notes on performance:

Most of the phases (2-4) can be time-consuming. Python's `multiprocessing` module is used to speed up the computations so that time execution is reduced considerably especially when running in a laptop.
