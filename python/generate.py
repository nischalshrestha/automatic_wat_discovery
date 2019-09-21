import pandas as pd
import numpy as np
from collections import OrderedDict

def construct_df(df_template, row_num=100, col_num=20):
    data = OrderedDict()
    n_rows = np.random.randint(1, df_template.shape[0] + 1)
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

def generate_series(template, column, rows):
    if template.dtypes[column] == np.int64:
        print(column, 'int')
        min_val = min(template[column])
        max_val = max(template[column])
        unique = set(template[column])
        print(min_val, max_val, len(unique))
        # print(pd.Series(np.random.randint(min_val, max_val + 1, rows)))
    elif template.dtypes[column] == np.float64:
        print(column, 'float')
        min_val = min(template[column])
        max_val = max(template[column])
        print(min_val, max_val)
        # if column == 'Age':
        #     print(pd.Series(np.round(np.random.uniform(min_val, max_val, size=(rows,)), 0)))
        # elif column == 'Fare':
        #     print(pd.Series(np.random.uniform(min_val, max_val, size=(rows,))))
    # elif template.dtypes[column] == np.bool:
    #     print('bool')
    # elif template.dtypes[column] == np.object:
    #     try:
    #         str(template.dtypes[column])
    #         unique = set(template[column])
    #         print(column, 'str', len(unique))
    #     except:
    #         pass

# PassengerId      int64
# Survived         int64
# Pclass           int64
# Name            object
# Sex             object
# Age            float64
# SibSp            int64
# Parch            int64
# Ticket          object
# Fare           float64
# Cabin           object
# Embarked        object

df = pd.read_csv("../train.csv")
print(df.head())
construct_df(df)
