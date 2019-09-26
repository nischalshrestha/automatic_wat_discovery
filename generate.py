import random as random
import pandas as pd
import numpy as np

from rpy2.robjects import pandas2ri
pandas2ri.activate()

from collections import OrderedDict

# PassengerId      int64
# Survived         int64 (level) - randomize
# Pclass           int64 (level) - randomize
# Name            object 
# Sex             object (level) - randomize
# Age            float64
# SibSp            int64
# Parch            int64
# Ticket          object (level) - randomize
# Fare           float64
# Cabin           object (level) - randomize
# Embarked        object (level) - randomize
FILEPATH = "../train.csv"

def construct_df(df_template: pd.DataFrame, max_row_num: int=100, col_num: int=20) -> pd.DataFrame:
    """Construct dataframe based on a template with psuedo-random values"""
    data = OrderedDict()
    n_rows = np.random.randint(1, max_row_num + 1)
    for col_name in df_template.columns.values:
        data[col_name] = generate_series(df_template, col_name, n_rows)
    return pd.DataFrame(data=data)

def generate_series(template: pd.DataFrame, column: str, rows: int) -> pd.Series:
    """
    Construct and return a Series for a specific column of a dataframe using information
    gleaned from the template dataframe.
    """
    if template.dtypes[column] == np.int64 or template.dtypes[column] == np.float64:
        min_val = min(template[column])
        max_val = max(template[column])
        if template.dtypes[column] == np.float64:
            if column == 'Age':
                arr = pd.Series(np.round(np.random.uniform(min_val, max_val, size=(rows,)), 0))
            # elif column == 'Fare':
            else:
                arr = pd.Series(np.random.uniform(min_val, max_val, size=(rows,)))
        else:
            arr = pd.Series(np.random.randint(min_val, max_val + 1, rows))
    elif template.dtypes[column] == np.bool:
        arr = np.random.randint(0, 2, rows, bool)
    elif template.dtypes[column] == np.object:
        unique = set(template[column])
        arr = [str(random.choice(template[column])) for _ in range(rows)]
    # throw in some random NAs if there were any in the original column
    if template[column].isna().sum() > 0:
        for i in range(len(arr)):
            chance = random.random()
            if chance < 0.1:
                # if template.dtypes[column] in {int, float}:
                #     arr[i] = np.NAN
                # else:
                #     print(template.dtypes[column])
                arr[i] = np.NAN
    return np.asarray(arr)

def construct_simple_df(df_template: pd.DataFrame) -> pd.DataFrame:
    """Construct dataframe based on a template with psuedo-random values"""
    data = OrderedDict()
    # n_rows = np.random.randint(1, max_row_num + 1)
    for col_name in df_template.columns.values:
        data[col_name] = generate_simple_series(df_template, col_name, df_template.shape[0])
    return pd.DataFrame(data=data)

def generate_simple_series(template: pd.DataFrame, column: str, rows: int) -> pd.Series:
    """
    Construct and return a Series for a specific column of a dataframe using information
    gleaned from the template dataframe.
    """
    if template.dtypes[column] == np.int64 or template.dtypes[column] == np.float64:
        arr = template[column]
    elif template.dtypes[column] == np.bool:
        arr = template[column]
    elif template.dtypes[column] == np.object:
        arr = [str(template[column][i]) for i in range(rows)]
    return np.asarray(arr)

def generate_args_from_df(df, n_args=256, lang="py"):
    args = []
    for n in range(n_args):
        new_df = construct_simple_df(df, 5, 5)
        if lang == "r":
            new_df = pandas2ri.py2rpy(new_df)
        args.append(new_df)
    return args

def generate_args(n_args=256, lang="py"):
    args = []
    df = pd.read_csv(FILEPATH)
    for n in range(n_args):
        new_df = construct_df(df, df.shape[0])
        if lang == "r":
            new_df = pandas2ri.py2rpy(new_df)
        args.append(new_df)
    return args

def generate_simple_arg(lang="py"):
    df = pd.read_csv(FILEPATH)
    new_df = construct_simple_df(df)
    if lang == "r":
        new_df = pandas2ri.py2rpy(new_df)
    return [new_df]

if __name__ == '__main__':
    # df = pd.read_csv(FILEPATH)
    # new_df = construct_df(df, df.shape[0])
    # print(new_df.shape)
    args = generate_args(10)
    print(args)
