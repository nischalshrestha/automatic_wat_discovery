# This is the default df when not randomized.

# df:
#    col0  col1  col2  col3
# 0     9     0  ID_0     0
# 1     8     1  ID_1     1
# 2     7     2  ID_2     2
# 3     6     3  ID_3     3
# 4     5     4  ID_4     4
# 5     4     5  ID_5     5
# 6     3     6  ID_6     6
# 7     2     7  ID_7     7
# 8     1     8  ID_8     8
# 9     0     8  ID_8     8

# shape
train.shape
train[['col1']].shape
# head
train.head()
# basic slicing
train.iloc[:8]
train.iloc[:7]
# filter 
train.query('col1 == 1 & col3 == 1')
# [ subsetting
train.iloc[0:5, 0:3]
train[(train.col1 == 1) & (train.col3 == 1)]
# [[
train[['col1', 'col2']]
# train[['col1', 'col3']]
train[['col1']]
# loc
train.loc[:, 'col1':'col3']
train.loc[1:2, 'col1']
# drop
train.drop(['col1', 'col2'], axis=1)
# drop dups
train[['col1', 'col2']].drop_duplicates()
train[['col1']].drop_duplicates()
# sort
train.sort_values(['col1', 'col2'])
train.sort_values('col1', ascending=False)
# summary
# df.describe()

#### Exploring richer expressions (combinations and chaining)

# subsetting on a particular column with a filter
train['col1'][train['col3'] == 8]
train[train['col3'] == 1]['col1']

# subsetting with a mix of a functions inside [] (notnull, isnull, isin)
train.loc[train.col1.isnull(), :]
train[train.col2.isin(['ID_3', 'ID_4'])]
train[(train['col2'].notnull()) & (train.col2.isin(['ID_0', 'ID_1']))]

# function call on subsetting operation(s)
train[(train.col1 == 1) & (train.col3 == 1)].head()
train.query('col1 < 5').head()


# train = train.loc[1:2, 'a']
# train = train.iloc[:9]
# train[train.col1 == 1]
# train = train[train.col1 == 1]
# train = train.query('col1 == 1 & col2 == 1')
# train = train[(train.col1 == 1) & (train.col2 == 1)]
# train = train[['col1']].drop_duplicates()
# train = train.drop([cols_to_drop], axis=1)


