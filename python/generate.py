import random as random
import pandas as pd
import numpy as np
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
    # n_rows = 
    print(n_rows)
    print(df_template.dtypes)
    # series_generator = SeriesGenerator(n_values=n_rows)
    for col_name in df_template.columns.values:
        # print(col_name)
        data[col_name] = generate_series(df_template, col_name, n_rows)
        # data[col_name] = 
    #   data[col_name] = series_generator.generate(column)
    # df = pd.DataFrame()
    return pd.DataFrame(data=data)

def generate_series(template: pd.DataFrame, column: str, rows: int) -> pd.Series:
    """
    Construct and return a Series for a specific column of a dataframe using information
    gleaned from the template dataframe.
    """
    if template.dtypes[column] == np.int64 or template.dtypes[column] == np.float64:
        # print(column, 'int')
        min_val = min(template[column])
        max_val = max(template[column])
        # print(min_val, max_val)
        if template.dtypes[column] == np.float64:
            if column == 'Age':
                # print(pd.Series(np.round(np.random.uniform(min_val, max_val, size=(rows,)), 0)))
                arr = pd.Series(np.round(np.random.uniform(min_val, max_val, size=(rows,)), 0))
            elif column == 'Fare':
                # print(pd.Series(np.random.uniform(min_val, max_val, size=(rows,))))
                arr = pd.Series(np.random.uniform(min_val, max_val, size=(rows,)))
        else:
            # print(pd.Series(np.random.randint(min_val, max_val + 1, rows)))
            arr = pd.Series(np.random.randint(min_val, max_val + 1, rows))
    elif template.dtypes[column] == np.bool:
        # print('bool')
        arr = np.random.randint(0, 2, rows, bool)
    elif template.dtypes[column] == np.object:
        try:
            str(template.dtypes[column])
            unique = set(template[column])
            # print(column, 'str', len(unique), unique)
            arr = [random.choice(template[column]) for _ in range(rows)]
        except:
            pass
    if template[column].isna().sum() > 0:
        # print(column, 'has nas')
        for i in range(len(arr)):
            chance = random.random()
            if chance < 0.1:
                if template.dtypes[column] in {int, float}:
                    arr[i] = np.NAN
                else:
                    arr[i] = ''
    return np.asarray(arr)

if __name__ == '__main__':
    df = pd.read_csv(FILEPATH)
    new_df = construct_df(df)
    print(new_df.shape)
