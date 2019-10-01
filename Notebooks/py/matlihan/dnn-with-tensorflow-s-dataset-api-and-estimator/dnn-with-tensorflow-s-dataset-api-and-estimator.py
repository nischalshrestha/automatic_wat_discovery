#!/usr/bin/env python
# coding: utf-8

# # DNN Approach using TensorFlow's Dataset API and Predefined Estimators

# With TensorFlow 1.4, the Dataset API is [introduced](https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html). The real advantage of the Dataset API is that a lot of memory management is done for the user when using large file-based datasets. And, in this work, we will be implementing a predefined DNN estimator and feed it with the Dataset API for Kaggle's Titanic dataset.
# 
# ---
# 
# With Dataset API we can use file-based datasets or datasets in the memory. In this work we will read the data from a csv file. In order to have it, you should first do some  feature engineering and then split the train set into train and validation sets.

# ## 1) Feature Engineering
# 
# We will do the same feature engineering as Trevor Stephens did [here](http://trevorstephens.com/kaggle-titanic-tutorial/r-part-4-feature-engineering/) before. The difference is that Trevor Stephens uses R and we will use python pandas library.

# ### Part 1: Label Encoding

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import preprocessing as prep


# In[ ]:


# our dataset is here
get_ipython().magic(u'ls ../input')


# In[ ]:


# read csv files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


test_df.head()


# In[ ]:


# add test df a survived cloumn
test_df['Survived'] = 0


# In[ ]:


test_df.head()


# In[ ]:


# concat train and test sets
concat = train_df.append(test_df, ignore_index=True)
concat.head()


# In[ ]:


# check numbers
print(train_df.shape)
print(test_df.shape)
print(concat.shape)


# In[ ]:


concat.Sex.unique()


# In[ ]:


# label encoder to transform categorical string data to integers
le = prep.LabelEncoder()


# In[ ]:


le.fit(concat.Sex)
le.classes_


# In[ ]:


# encode labels
Sex_le = le.transform(concat.Sex)
Sex_le[0:10]


# In[ ]:


concat_le = concat.copy()
concat_le.head()


# In[ ]:


concat_le.Sex = Sex_le
concat_le.head()


# In[ ]:


# check data types
concat_le.dtypes


# In[ ]:


# print unique values
print(concat.Survived.unique())
print(concat.Pclass.unique())
print(concat.Sex.unique())
print(concat.SibSp.unique())
print(concat.Parch.unique())
print(concat.Embarked.unique())


# In[ ]:


# remove nans and fill with '0's for embarked
embarked = concat['Embarked'].fillna('0')
embarked.unique()


# In[ ]:


# label encode embarked
le.fit(embarked)
embarked = le.transform(embarked)
embarked[:10]


# In[ ]:


concat_le.Embarked = embarked


# In[ ]:


# check
concat_le.head(10)


# In[ ]:


# split train and test sets
train_le = concat_le.iloc[:891].copy()
test_le = concat_le.iloc[891:].copy()


# In[ ]:


# And save
get_ipython().magic(u'mkdir -p data')
train_le.to_csv('./data/train_le.csv', index=False)
test_le.to_csv('./data/test_le.csv', index=False)


# In[ ]:


get_ipython().magic(u'ls data')


# ### Part 2: Further Feature Engineering

# In[ ]:


train = pd.read_csv('./data/train_le.csv')
test = pd.read_csv('./data/test_le.csv')


# In[ ]:


# concat dfs again
concat = train.append(test)


# In[ ]:


# check numbers
concat.shape


# In[ ]:


train.shape[0] + test.shape[0]


# Feature engineer names

# In[ ]:


NameSplit = concat.Name.str.split('[,.]')


# In[ ]:


NameSplit.head()


# In[ ]:


titles = [str.strip(name[1]) for name in NameSplit.values]
titles[:10]


# In[ ]:


# New feature
concat['Title'] = titles


# In[ ]:


concat.Title.unique()


# In[ ]:


# redundancy: combine Mademoiselle and Madame into a single type
concat.Title.values[concat.Title.isin(['Mme', 'Mlle'])] = 'Mlle'


# In[ ]:


# keep reducing the number of factor levels
concat.Title.values[concat.Title.isin(['Capt', 'Don', 'Major', 'Sir'])] = 'Sir'
concat.Title.values[concat.Title.isin(['Dona', 'Lady', 'the Countess', 'Jonkheer'])] = 'Lady'


# In[ ]:


# label encode new feature too
le.fit(concat.Title)
le.classes_


# In[ ]:


concat.Title = le.transform(concat.Title)


# In[ ]:


concat.head(10)


# ### New features family size and family id

# In[ ]:


# new feature family size
concat['FamilySize'] = concat.SibSp.values + concat.Parch.values + 1


# In[ ]:


concat.head(10)


# New feature `FamilyID`, extract family information from surnames and family size information. Members of a family should have both the same surname and family size.

# In[ ]:


surnames = [str.strip(name[0]) for name in NameSplit.values]
surnames[:10]


# In[ ]:


concat['Surname'] = surnames
concat['FamilyID'] = concat.Surname.str.cat(concat.FamilySize.astype(str), sep='')
concat.head(10)


# In[ ]:


# mark any family id as small if family size is less than or equal to 2
concat.FamilyID.values[concat.FamilySize.values <= 2] = 'Small'


# In[ ]:


concat.head(10)


# In[ ]:


# check the frequency of family ids
concat.FamilyID.value_counts()


# Too many family ids with few family members, maybe some families had different last names or something else. Let's clean this too.

# In[ ]:


freq = list(dict(zip(concat.FamilyID.value_counts().index.tolist(), concat.FamilyID.value_counts().values)).items())
type(freq)


# In[ ]:


freq = np.array(freq)
freq[:10]


# In[ ]:


freq.shape


# In[ ]:


# select the family ids with frequency of 2 or less
freq[freq[:,1].astype(int) <= 2].shape


# In[ ]:


freq = freq[freq[:,1].astype(int) <= 2]


# In[ ]:


# assign 'Small' for those
concat.FamilyID.values[concat.FamilyID.isin(freq[:,0])] = 'Small'


# In[ ]:


concat.FamilyID.value_counts()


# In[ ]:


# label encoding for family id
le.fit(concat.FamilyID)
concat.FamilyID = le.transform(concat.FamilyID)
concat.FamilyID.unique()


# In[ ]:


# choose usefull features
concat_reduce = concat[[
    'PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp',
    'Parch', 'Fare', 'Title', 'Embarked', 'FamilySize',
    'FamilyID', 'Survived']]
concat_reduce.head()


# In[ ]:


# split
train_final = concat_reduce.iloc[:891].copy()
test_final = concat_reduce.iloc[891:].copy()


# In[ ]:


# save
train_final.to_csv('./data/train_final.csv', index=False)
test_final.to_csv('./data/test_final.csv', index=False)


# In[ ]:


get_ipython().magic(u'ls data')


# ## Part 3: Split train set  into train and validation sets

# In[ ]:


train = pd.read_csv('./data/train_final.csv')


# In[ ]:


train.shape


# In[ ]:


train.head(10)


# In[ ]:


# shuffle dataframe
train = train.sample(frac=1).reset_index(drop=True)


# In[ ]:


train.head(10)


# In[ ]:


# define number of rows for validation set
n_valid_rows = int(train.shape[0]*0.3)
n_valid_rows


# In[ ]:


# split
valid_split = train.iloc[:n_valid_rows].copy()
train_split = train.iloc[n_valid_rows:].copy()
print(train_split.shape)
print(valid_split.shape)


# In[ ]:


# save
train_split.to_csv('./data/train_split_final.csv', index=False)
valid_split.to_csv('./data/valid_split_final.csv', index=False)

get_ipython().magic(u'ls data')


# ---

# ## 2) Deep Neural Networks with TensorFlow's Dataset API and Estimators

# In[ ]:


import tensorflow as tf


# In[ ]:


train = pd.read_csv('./data/train_split_final.csv')
valid = pd.read_csv('./data/valid_split_final.csv')

train.head()


# `{Pclass, Sex, Age, SibSp, Parch, Fare, Title, Embarked, FamilySize, FamilyID}` we will be using as the features and the `Survived` column will be our labels.

# In order to use the Datasets API and feed the Estimator, we should write an input function like this:
# 
# ```python
# def input_fn():
#     ...<code>...
#     return ({ 'Pclass':[values], ..<etc>.., 'FamilyID':[values] },
#             [Survived])
# ```

# This function takes the `file_path` as input and outputs a two-element tuple. The first element of the tuple is a dictionary containing feature names as keys and features as values. And the second element is a list of labels for the training batch.
# 
# Other two arguments for the input function are `perform_shuffle` and `repeat_count`. If `perform_shuffle` is `True` the order of the examples are shuffled. The `perform_shuffle` argument specifies the number of epochs during training, for instance, if `perform_shuffle=1` all the train set examples are passed only once.

# And the implementation is as follows, we will use this function to feed the estimator later.

# In[ ]:


# define feature names first
feature_names = [
    'Pclass',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Fare',
    'Title',
    'Embarked',
    'FamilySize'
    'FamilyID']


# In[ ]:


def titanic_input_fn(file_path, perform_shuffle=False, repeat_count=1):
    def decode_csv(line):
        # second argument of decode_csv defines the data types for each dataset column!
        # the first argument is passenger ids thus integer
        # the last column is survived or not labels thus integer
        # and the rest are float.
        parsed_line = tf.decode_csv(
            line, [[0], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0]])
        label = parsed_line[-1:] # Last element is the label
        del parsed_line[-1] # Delete last element (it is the labels)
        features = parsed_line[1:] # First element is excluded since it is the id column
        d = dict(zip(feature_names, features)), label
        return d
    
    dataset = (tf.data.TextLineDataset(file_path) # Read text file
        .skip(1) # Skip header row
        .map(decode_csv)) # Transform each elem by applying decode_csv fn
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
    dataset = dataset.batch(32)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


# Memory management is provided here with `TextLineDataset`.

# Let's print and check the first batch:

# In[ ]:


train_path = './data/train_final.csv'


# In[ ]:


next_batch = titanic_input_fn(train_path, False) # Will return first 32 elements

with tf.Session() as sess:
    first_batch = sess.run(next_batch)
print(first_batch)


# Ok, it is working!

# ---
# Now we will define our DNN estimator

# In[ ]:


# remove checkpoints directory if you will rebuild the estimator
# by running this cell again
#%rm -r ./checkpoints

# path to save checkpoints
save_dir = './checkpoints'

# reset default graph if rebuilding the classifier
tf.reset_default_graph()

# Create the feature_columns, which specifies the input to our model.
# All our input features are numeric, so use numeric_column for each one.
feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]

# define the classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns, # The input features to our model
    hidden_units=[2048, 1024, 512, 256, 128], # 5 layers
    n_classes=2, # survived or not {1, 0}
    model_dir=save_dir, # Path to where checkpoints etc are stored
    optimizer=tf.train.RMSPropOptimizer(
        learning_rate=0.00001),
    dropout=0.1)


# Now we will train the model using `titanic_input_fn` and our classifier.

# In[ ]:


train_path = './data/train_split_final.csv'
valid_path = './data/valid_split_final.csv'
test_path = './data/test_final.csv'


# In[ ]:


# the classifier will run for 20 epochs below
classifier.train(input_fn=lambda: titanic_input_fn(train_path, True, 20))


# It is not tuned for a high accuracy, it is possible to get higher accuracies about %85 by tuning hyperparameters.

# In[ ]:


# evaluate
# Return value will contain evaluation_metrics such as: loss & average_loss
evaluate_result = classifier.evaluate(
   input_fn=lambda: titanic_input_fn(valid_path, False, 1))
print('')
print("Evaluation results:")
for key in evaluate_result:
    print("   {}, was: {}".format(key, evaluate_result[key]))


# In[ ]:


# and the prediction is like
predict_results = classifier.predict(
    input_fn=lambda: titanic_input_fn(test_path, False, 1))
print("Predictions on test file")
i=892
for prediction in predict_results:
    # Will print the predicted class: 0 or 1.
    print(prediction["class_ids"][0])
    i += 1
print(i)


# That's all folks!
