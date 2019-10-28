
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
dim(train)
dim(train[c('col1')])
# head
head(train)
# basic slicing
slice(train, 1:8)
train[1:8, ]
slice(train, 1:7)
train[1:7, ]
# filter
filter(train, col1 == 1, col3 == 1) # safe
# [ subsetting
train[1:5, 1:3]
train[train$col1 == 1 & train$col3 == 1, ] # unsafe
train[which(train$col1 == 1 & train$col3 == 1), ] # safe
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
train[order(train$col1, train$col2), ] # potential gotcha: cant directly supply name of col
arrange(train, desc(col1))
train[order(-train$col1), ]
# summary
# summary(df)

#### Exploring richer expressions (combinations and chaining)

# subsetting on a particular column with a filter
train$col1[train$col3 == 1]
train[train$col3 == 1, ]$col1

# subsetting with a mix of a functions inside [] (notnull, isnull, isin)
train[is.na(train$col1), ]
train[train$col2 %in% c('ID_3', 'ID_4'), ]
train[!is.na(train['col2']) & train$col2 %in% c('ID_3', 'ID_4')]

# function call on subsetting operation(s)
head(train[train$col1 == 1 & train$col3 == 1, ])
head(subset(train, col1 < 5))


# convert these to R
# train = train.loc[1:2, 'a']
# train = train.iloc[:9]
# train = train[train.col1 == 1]
# train = train.query('col1 == 1 & col2 == 1')
# train = train[(train.col1 == 1) & (train.col2 == 1)]
# train = train[['col1']]
# train = train[['col1']].drop_duplicates()
# train = train.drop([cols_to_drop], axis=1)
