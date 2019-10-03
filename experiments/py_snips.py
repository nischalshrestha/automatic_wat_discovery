
# shape
train.shape
train[['col1']].shape
# head
train.head()
# basic slicing
train.iloc[:8]
train[:5][:2]
# filter 
train.query('col1 == 1 & col2 == 1')
# [ subsetting
train[(train.col1 == 1) & (train.col2 == 1)]
# [[
train[['col1', 'col2']]
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
df.sort_values(['col1', 'col2'])
df.sort_values('col1', ascending=False)
# summary
df.describe()

# train = train.loc[1:2, 'a']
# train = train.iloc[:9]
# train[train.col1 == 1]
# train = train[train.col1 == 1]
# train = train.query('col1 == 1 & col2 == 1')
# train = train[(train.col1 == 1) & (train.col2 == 1)]
# train = train[['col1']].drop_duplicates()
# train = train.drop([cols_to_drop], axis=1)


