
# shape
dim(train)
dim(train[c('col1')])
# head
head(train)
# basic slicing
slice(train, 1:8)
train[1:8, ]
train[1:7, ]
# filter
filter(train, col1 == 1, col3 == 1)
# [ subsetting
train[1:5, 1:3]
train[train$col1 == 1 & train$col3 == 1, ]
# [[ equivalents
# select(train, col1)
select(train, col1, col2)
# train[c('col1', 'col2')]
# select(train, col1)
train[c('col1')]
train[['col1']]
# loc equivalents
select(train, col1:col3)
train[2:3, 'col1']
# drop
select(train, -c(col1, col2))
# train[ , -c('col1', 'col2')]
# drop duplicates
distinct(select(train, col1, col2))
distinct(select(train, col1))
# sort
arrange(train, col1, col2)
arrange(train, desc(col1))
# summary
# summary(df)

# convert these to R
# train = train.loc[1:2, 'a']
# train = train.iloc[:9]
# train = train[train.col1 == 1]
# train = train.query('col1 == 1 & col2 == 1')
# train = train[(train.col1 == 1) & (train.col2 == 1)]
# train = train[['col1']]
# train = train[['col1']].drop_duplicates()
# train = train.drop([cols_to_drop], axis=1)
