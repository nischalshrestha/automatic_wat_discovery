#!/usr/bin/env python
# coding: utf-8

# # Disclaimer:
# For the Titanic dataset, the best performance will likely be achieved by a non-ANN model. But as a student interested in applying artificial neural networks, I thought it'd be a fun/educational challenge! I'm still very very new to machine learning and neural networks so feedback is much appreciated!

# # 0) Libraries

# In[1]:


from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Tensorflow
import tensorflow as tf
from tensorflow.python.data import Dataset


# # 1) Loading in and checking data

# In[2]:


# Okay, let's load in our datasets!
raw_train_df = pd.read_csv("/kaggle/input/train.csv")
raw_test_df = pd.read_csv("/kaggle/input/test.csv")
example_submission_df = pd.read_csv("/kaggle/input/gender_submission.csv")

train_df = raw_train_df.copy(deep=True)
test_df = raw_test_df.copy(deep=True)
train_test_lst = [train_df, test_df]


# ### First, Let's take a look at the train and test data to make sure everything was loaded okay

# In[3]:


# Taking a look at the first few values in the dataframe
display(train_df.head())
# Taking a look at the summary statistics for each feature
display(train_df.describe())


# The minimum fare of 0.00 stuck out to me as a little odd so I decided to take a little deeper look.
# Seems like maybe crew members or staff working on the Titanic? That said, there is a Jonkheer in the list too so there might have been some free tickets involved for special personages...

# In[4]:


train_df[train_df['Fare'] == 0]


# In[5]:


display(test_df.head())
display(test_df.describe())


# # 2) Data preprocessing

# ### Both train and test datasets appear to have NaN values this could cause problems for our model, so let's look at what is missing and how much

# In[6]:


display(train_df.isnull().sum())
print("Total individuals in train set is: {}".format(len(train_df)))


# In[7]:


display(test_df.isnull().sum())
print("Total individuals in test set is: {}".format(len(test_df)))


# ### The huge amount of missing Cabin data is worrying, but let's see if it has any predictive power before figuring out what to do

# In[8]:


# Let's only consider data that has non-NaN Cabin values (Age or Embarked can still be NaN!)
cabin_df = train_df[train_df['Cabin'].notnull()]

# Let's create a new feature 'deck_level' that groups passengers by deck levels
cabin_df = cabin_df.assign(deck_level=pd.Series([entry[:1] for entry in cabin_df['Cabin']]).values)
display(cabin_df.head())

print("Survival chances based on deck level:")
cabin_df.groupby(['deck_level'])['Survived'].mean()


# So it looks like deck level may be a useful feature to learn. The NaNs are troubling though, we can get around them (hopefully) by adding a new option for the deck_level to be 'U' (for unknown).
# 
# Later, we'll use one hot encoding on deck_level before sending it to our neural network.

# In[9]:


def process_deck_level(train_test_lst):
    new = []
    for dataset in train_test_lst:
        dataset = dataset.copy(deep=True)
        # Take the first letter of the Cabin entry if it's not nan. Otherwise, it should be labelled as 'U'.
        dataset = dataset.assign(deck_level=pd.Series([entry[:1] if not pd.isnull(entry) else 'U' for entry in dataset['Cabin']]))
        # Okay, now let's drop the Cabin column from our dataset
        dataset = dataset.drop(['Cabin'], axis = 1)
        new.append(dataset)
    return (new)

train_df, test_df = process_deck_level(train_test_lst)

# Let's check that we did the right thing...
display(train_df.head())
display(test_df.head())
# Let's also recheck what's still missing
display(train_df.isnull().sum())
display(test_df.isnull().sum())


# Okay looking better already! 
# ### Now let's try to address the missing embarked data! First off what are the possible values of embarked?

# In[10]:


display(set(train_df['Embarked']))
print("Survival chances based on embarcation:")
train_df.groupby(['Embarked'])['Survived'].mean()


# It looks like people who embarked from Q had a low survival rate and S had an especially low survival rate...
# 
# For this feature, we'll also fill NaN values with 'N' for 'Not known' since filling with C/Q/S looks like it would make a big difference.

# In[11]:


# Replace NaN values in the 'Embarked' column with 'N'
train_df[['Embarked']] = train_df[['Embarked']].fillna('N')
# Let's check that we filled things correctly!
display(set(train_df['Embarked']))
display(train_df.isnull().sum())


# ### Let's take a quick look at the test data and see what to do about the one fare datapoint that is missing

# In[12]:


test_df[test_df['Fare'].isnull()]


# ### Let's use Pclass to fill our missing value!

# In[13]:


Pclass_Fare_grouping = test_df.groupby(["Pclass"])['Fare']
Pclass_Fare_grouping.plot(kind='hist', bins=15, legend=True, title='Fare histogram grouped by Pclass')
plt.xlabel('Fare')
print("Mean Fare for each Pclass:")
display(Pclass_Fare_grouping.mean())
print("Median Fare for each Pclass:")
display(Pclass_Fare_grouping.median())


# The tail for the Fare for Pclass 3 is a bit long so it's probably safer to fill with the median value for that Pclass.
# 
# All this work for one missing fare is overkill, but it's a good exercise in thinking about how to impute data!

# In[14]:


test_df[['Fare']] = test_df[['Fare']].fillna(Pclass_Fare_grouping.median()[3])
# Let's check that our one fill worked!
display(test_df[test_df['PassengerId'] == 1044])
display(test_df.isnull().sum())


# ### Now to figure out what to do about missing age data. Let's do a quick analysis of age before imputing any values!

# ### Let's first just take a look at the age distribution in our training set.

# In[15]:


ax = train_df[['Age']].plot(kind='hist', bins=20)
plt.xlabel("Age")
_ = plt.title("Age histogram")


# ### Next, let's look at the relationship between Age and survival

# In[16]:


train_df.groupby(['Survived', pd.cut(train_df['Age'], np.arange(0, 100, 5))]).size().unstack(0).plot.bar(stacked=True, alpha=0.75)
_ = plt.title("Age histogram grouped by survival")


# * So an initial analysis shows that younger passengers ( < 6) were much more likely to survive than not.
# 
# * Agewise, the worst outcomes were for folks in their late teens and early 20's. 
# 
# * Bad outcomes also for people between age ~24 and ~32 as well.

# ### What about the effect of gender and age on survival?

# In[17]:


train_df.groupby(['Survived', 'Sex', pd.cut(train_df['Age'], np.arange(0, 100, 5))]).size().unstack(0).plot.bar(stacked=True, alpha=0.75)
_ = plt.title("Age histogram grouped by survival and gender")


# * This plot is a bit messy, but the left side shows pretty clearly that females had a high survival rate.
# 
# * Looking at right side paints the opposite picture (with the exception if you were a male under 6, then youre survival chances were pretty good).
# 
# ### This quick set of observations seem to suggest that getting age right for young children and those between 20 and 30 is important for predicting survival.

# ### One promising strategy to impute age that seems to work well in other kernels is to use the name title

# In[18]:


# All name formats seem to be something like:
# "last_name, title. first_name "nickname" (full_name)"
# To get title, we split the string by comma and select the second half. Then we split that second half by '.' and take the first half
# i.e.
# 1) ["last_name", "title. first_name "nickname" (full_name)"] (select element 1!)
# 2) ["title", "first_name "nickname" (full_name)"] (select element 0!)
train_titles = [name.split(',')[1].lstrip(' ').split('.')[0] for name in train_df['Name']]
# Let's see if the above strategy works
print("Train set titles (and counts):")
print(Counter(train_titles))

print("\nTest set titles (and counts):")
test_titles = [name.split(',')[1].lstrip(' ').split('.')[0] for name in test_df['Name']]
print(Counter(test_titles))

print("\n===============================")

age_missing_train_titles = [name.split(',')[1].lstrip(' ').split('.')[0] for name in train_df[train_df['Age'].isnull()]['Name']]
print("\nTrain set titles (and counts) with missing ages:")
print(Counter(age_missing_train_titles))

age_missing_test_titles = [name.split(',')[1].lstrip(' ').split('.')[0] for name in test_df[test_df['Age'].isnull()]['Name']]
print("\nTest set titles (and counts) with missing ages:")
print(Counter(age_missing_test_titles))


# ### Looks like we have a nice list of titles, let's add them to our dataframe for now

# In[19]:


# Let's add the titles as a new feature for our dataset
def naive_process_title(train_test_lst):
    new = []
    for dataset in train_test_lst:
        dataset = dataset.copy(deep=True)
        titles = [name.split(',')[1].lstrip(' ').split('.')[0] for name in dataset['Name']]
        dataset = dataset.assign(title=pd.Series(titles).values)
        new.append(dataset)
    return (new)

train_df, test_df = naive_process_title([train_df, test_df])

# Taking a look at our dataframes to make sure we did the right thing...
display(train_df.head())
display(test_df.head())


# ### I'm not super well versed with titles from "back in the day" so let's see if we can discover how age (the thing we want to impute) relates to title

# In[20]:


def plot_title_age_hist(title, train_df, bins=20):
    title_ages = train_df[train_df['title'] == title]['Age']
    title_ages.plot(kind='hist', bins=bins, legend=True)
    title_ages.describe()
    plt.xlabel("Age")
    plt.title("Age histogram for '{}' title".format(title))


# In[21]:


title_groups = train_df.groupby(['title'])
display(title_groups['Age'].describe())
plot_title_age_hist("Master", train_df, bins=10)


# * It looks like 'Master' is a reliable signal for young boy.

# In[22]:


plot_title_age_hist('Miss', train_df)


# * Miss looks like the corresponding title, but it can take on a much much larger variation of values...

# ### Let's see if we can get more specific ages to impute for the "miss" title by using the 'Parch' feature

# In[23]:


def title_feature_age_analysis(title, feature, train_df, bins):
    # Let's loop through all values of our feature of interest (in this case "Parch")
    for i in range(max(train_df[train_df['title'] == title][feature]) + 1):
        # Create an age histogram for a given feature level that also has our title of interest
        train_df[(train_df['title'] == title) & (train_df[feature] == i)]['Age'].plot(kind="hist", bins=bins, legend = True, label="{} {}".format(feature, i), alpha = 0.5)
        # Print common descriptive stats for our title and the given level of our feature
        print("Statistics for '{}' title with {} of: {}".format(title, feature, i))
        display(train_df[(train_df['title'] == title) & (train_df[feature] == i)]['Age'].describe())
        print("Median\t{}\n".format(train_df[(train_df['title'] == title) & (train_df[feature] == i)]['Age'].median()))
        print("=========================\n")
        plt.xlabel("Age")
        _ = plt.title("Age histogram for '{}' title grouped by {}".format(title, feature))

title_feature_age_analysis('Miss', 'Parch', train_df, bins=20)


# ### Cool! A parch of 1 or 2 together with the 'Miss' title seems to be quite indicative of younger age! Does our finding in the train dataset hold up in the test dataset?

# In[24]:


title_feature_age_analysis('Miss', 'Parch', test_df, bins=20)


# ### Besides 'Miss' and 'Master' we'll also have to fill many more missing ages with 'Mr' and 'Mrs'. Let's see if we can use Parch to help us out again!

# In[25]:


title_feature_age_analysis('Mrs', "Parch", train_df, bins=20)


# In[26]:


title_feature_age_analysis('Mr', "Parch", train_df, bins=20)


# So it looks like "Parch" is not super helpful for narrowing the age of 'Mr' and 'Mrs' titles. Let's try using the median to fill these titles then...

# ### We've taken a bit of a look at titles and their relation to age, time to fill in our missings age values with the above information

# In[27]:


def age_imputer(train_test_lst):
    new = []
    for dataset in train_test_lst:
        dataset = dataset.copy(deep=True)
        # This is the list of unique titles for individuals with a NaN age
        missing_age_titles = list(set([name.split(',')[1].lstrip(' ').split('.')[0] for name in dataset[dataset['Age'].isnull()]['Name']]))
        print("Titles for individuals with missing age are: {}".format(missing_age_titles))
        for title in missing_age_titles:
            # Fill in missing ages for 'Mr'/'Mrs'/'Master'/'Ms'/'Dr' titles
            if (title in ['Mr', 'Mrs', 'Master', 'Ms', 'Dr']):
                median = dataset[(dataset['title'] == title)]['Age'].median()
                # Treat 'Ms' as 'Mrs'
                if (title == 'Ms'):
                    median = dataset[(dataset['title'] == 'Mrs')]['Age'].median()
                dataset[(dataset['title'] == title) & (dataset['Age'].isnull())] = dataset[(dataset['title'] == title) & (dataset['Age'].isnull())].fillna(median)
            # Fill in missing ages for "Miss" titles
            elif (title == 'Miss'):
                for level in range(max(dataset[dataset['title'] == title]['Parch']) + 1):
                    df = dataset[(dataset['title'] == 'Miss') & (dataset['Age'].isnull()) & (dataset['Parch'] == level)]
                    if (not df.empty):
                        median = dataset[(dataset['title'] == title) & (dataset['Parch'] == level)]['Age'].median()
                        dataset[(dataset['title'] == 'Miss') & (dataset['Age'].isnull()) & (dataset['Parch'] == level)] = dataset[(dataset['title'] == 'Miss') & (dataset['Age'].isnull()) & (dataset['Parch'] == level)].fillna(median)
        new.append(dataset)
    return (new)

train_df, test_df = age_imputer([train_df, test_df])


# In[28]:


display(train_df.isnull().sum())
display(test_df.isnull().sum())


# ### Looks like all the NaNs got filled in. But we should do some sanity checks to verify that things got filled in **correctly**.

# In[29]:


display(raw_train_df[raw_train_df['Age'].isnull()])


# In[30]:


# Select passengers that have NaN ages in our raw_train_df
train_df.loc[train_df['PassengerId'].isin(raw_train_df[raw_train_df['Age'].isnull()]['PassengerId'])]


# ### How does our new distribution look?

# In[31]:


fig, (ax1, ax2) = plt.subplots(1, 2)
# First column plot
train_df[['Age']].plot(kind='hist', bins=20, ax=ax1, legend=False)
ax1.set_xlabel("Age")
ax1.set_title("NaN filled")
ymin, ymax = ax1.get_ylim()
# Second column plot
raw_train_df[['Age']].plot(kind='hist', bins=20, ax=ax2, sharey=True, legend=False)
ax2.set_ylim(ymin, ymax)
ax2.set_xlabel("Age")
_ = ax2.set_title("Original distribution")


# Looks like things got filled in properly. We now have a massive peak at 30 and 35 years of age due to the large number of fills we made (for 'Mr' and 'Mrs' titles) but the rest of the distribution looks to be preserved...

# # 3) Some feature engineering

# While going through the data, I happened to stumble on the unfortunate 'Sage' family which had a 0 survival rate despite having women and children (which normally have a high survival rate).

# In[32]:


raw_train_df[raw_train_df['Name'].str.startswith('Sage,')]


# Maybe, family size and/or the Pclass also played a big role in survivorship. Let's engineer some features and explore!

# In[33]:


# Let's add family_size which is just the sum of 'SibSp' and 'Parch'
train_df['family_size'] = train_df['SibSp'] + train_df['Parch']
test_df['family_size'] = test_df['SibSp'] + test_df['Parch']
# Check that things were added properly
display(train_df.head())
# Plot family size grouped by survival
train_df.groupby(['Survived'])['family_size'].plot(kind='hist', legend=True, alpha=0.5)
plt.xlabel("Family Size")
_ = plt.title("Histogram of family size grouped by survival")


# So yeah... large family is definitely not good for survival...

# In[34]:


_ =train_df.groupby(['Survived', pd.cut(train_df['Pclass'], np.arange(0, 4))]).size().unstack(0).plot.bar(stacked=True)


# Being in Pclass gives much higher chance of survival but let's drill in more and look at gender too

# In[35]:


_ = train_df.groupby(['Survived', 'Sex', pd.cut(train_df['Pclass'], np.arange(0, 4))]).size().unstack(0).plot.bar(stacked=True)


# Looks like if you were a female in class 1 or 2 your chances were pretty great. Pclass 3 females had more of a 50/50 chance.
# 
# As a male things look much more grim. But Pclass 1 and 2 males still fare better than those in 3.
# 
# Since I'm using tensorflow, we can create the 'Pclass' and 'Sex' feature cross in the data pipeline (below).

# # 4) Assembling pipeline for tensorflow

# In[36]:


# To get things to work nicely with tensorflow we'll need to subtract one from 'Pclass' so our classes start at 0
train_df['Pclass'] = train_df['Pclass'] - 1
test_df['Pclass'] = test_df['Pclass'] - 1


# In[37]:


train_df


# In[38]:


# Let's remind ourselves of the data columns we have
train_df.columns


# In[49]:


def build_feature_columns():
    """
    Build our tensorflow feature columns!
    
    For a great overview of the different feature columns in tensorflow and when to use them, see:
    https://www.tensorflow.org/versions/master/get_started/feature_columns
    """
    Pclass = tf.feature_column.categorical_column_with_identity("Pclass", num_buckets = 3)
    Sex = tf.feature_column.categorical_column_with_vocabulary_list("Sex", ["female", "male"])
    Age = tf.feature_column.numeric_column("Age")
    SibSp = tf.feature_column.numeric_column("SibSp")
    Parch = tf.feature_column.numeric_column("Parch")
    Fare = tf.feature_column.numeric_column("Fare")
    # I end up not using the 'Embarked' feature but you can try it if you'd like!
    #Embarked = tf.feature_column.categorical_column_with_vocabulary_list("Embarked", ["C", "N", "Q", "S"])
    # I end up not using the deck_level feature but you can try it if you'd like!
    #deck_level = tf.feature_column.categorical_column_with_vocabulary_list("deck_level", ["A", "B", "C", "D", "E", "F", "G", "T", "U"])
    family_size = tf.feature_column.numeric_column("family_size")
    Pclass_x_Sex = tf.feature_column.crossed_column(keys = [Pclass, Sex], hash_bucket_size = 10)
    # Let's bucket age into 5 year boundaries
    age_buckets = tf.feature_column.bucketized_column(Age, boundaries=list(range(5, 100, 10)))
    fare_buckets = tf.feature_column.bucketized_column(Fare, boundaries=list(range(1, 600, 10)))
    # Wrapping categorical columns in indicator columns
    #Embarked = tf.feature_column.indicator_column(Embarked)
    #deck_level = tf.feature_column.indicator_column(deck_level)
    Pclass = tf.feature_column.indicator_column(Pclass)
    Sex = tf.feature_column.indicator_column(Sex)
    Pclass_x_Sex = tf.feature_column.indicator_column(Pclass_x_Sex)
    # Time to put together all the feature we'll use!
    #feature_columns = set([Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, age_buckets, fare_buckets, family_size, Pclass_x_Sex])
    #feature_columns = set([Pclass, Sex, Age, SibSp, Parch, Fare, age_buckets, fare_buckets, family_size, Pclass_x_Sex])
    #feature_columns = set([Pclass, Sex, Age, SibSp, Parch, Fare, age_buckets, family_size, Pclass_x_Sex])
    #feature_columns = set([Pclass, Sex, Age, SibSp, Parch, fare_buckets, age_buckets, family_size, Pclass_x_Sex])
    feature_columns = set([Pclass, Sex, fare_buckets, age_buckets, family_size, Pclass_x_Sex])
    return(feature_columns)


# In[50]:


def input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """
    This is our input function that will pass data into the tensorflow DNN class we'll create.
    It takes in a pandas dataframe.
    It outputs a tensorflow dataset one_shot_iterator
    """
    # Convert pandas df to dict of numpy arrays
    features = {key:np.array(value) for key, value in dict(features).items()}
    # Put together the tensorflow dataset. Configures batching/repeating.
    dataset = Dataset.from_tensor_slices((features, targets))
    dataset = dataset.batch(batch_size).repeat(num_epochs)
    # Shuffle data
    if (shuffle):
        dataset = dataset.shuffle(buffer_size = 20000)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return (features, labels)


# We'll take our train data and do a 70/30 split so that we can get a sense for the validation performance before trying it on the test set.

# In[51]:


train_ex_df = train_df.sample(frac=0.70)
train_targ_series = train_ex_df['Survived']
#train_ex_df = train_ex_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "family_size", "deck_level"]]
#train_ex_df = train_ex_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "family_size", "deck_level"]]
train_ex_df = train_ex_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "family_size"]]

xval_ex_df = train_df.drop(train_ex_df.index)
xval_targ_series = xval_ex_df['Survived']
#xval_ex_df = xval_ex_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "family_size", "deck_level"]]
#xval_ex_df = xval_ex_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "family_size", "deck_level"]]
xval_ex_df = xval_ex_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "family_size"]]


# # 5) Building our (deep neural network) DNN classifier model with tensorflow

# In[52]:


def plot_acc(train_accs, val_accs):
    fig, ax = plt.subplots(1, 1)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Period")
    ax.set_title("DNN model accuracy vs. Period")
    ax.plot(train_accs, label = "train")
    ax.plot(val_accs, label = "validation")
    ax.legend()
    fig.tight_layout()
    
    print("Final accuracy (train):\t\t{:.3f}".format(train_accs[-1]))
    print("Final accuracy (validation):\t{}".format(val_accs[-1]))

def train_dnn_classifier(periods, learning_rate, steps, batch_size, hidden_units, train_ex, train_targ, val_ex, val_targ):
    #steps per period (spp)
    spp = steps / periods
    # Make our tensorflow DNN classifier (we'll use the ProximalAdagradOptimizer with L1 regularization to punish overly complex models)
    optim = tf.train.ProximalAdagradOptimizer(learning_rate = learning_rate,
                                      l1_regularization_strength=0.1)
    # We'll use the stock DNNClassifier
    # We'll also add dropout at a 20% rate to make our network more robust (hopefully)
    # Finally, we'll make our activation function a leaky_relu
    dnn_classifier = tf.estimator.DNNClassifier(feature_columns = build_feature_columns(),
                                                hidden_units = hidden_units,
                                                optimizer = optim,
                                                dropout = 0.25,
                                                activation_fn = tf.nn.leaky_relu)
    # Input functions
    train_input_fn = lambda: input_fn(train_ex,
                                     train_targ,
                                     batch_size = batch_size)
    pred_train_input_fn = lambda: input_fn(train_ex,
                                          train_targ,
                                          num_epochs = 1,
                                          shuffle = False)
    pred_val_input_fn = lambda: input_fn(val_ex,
                                         val_targ,
                                         num_epochs = 1,
                                         shuffle = False)
    #train and validation accuracy per period
    train_app = []
    val_app = []
    for period in range(periods):
        # Train our classifier
        dnn_classifier.train(input_fn = train_input_fn, steps = spp)
        # Check how our classifier does on training set after one period
        train_pred = dnn_classifier.predict(input_fn = pred_train_input_fn)
        train_pred = np.array([int(pred['classes'][0]) for pred in train_pred])
        # Check how our classifier does on the validation set after one period
        val_pred = dnn_classifier.predict(input_fn = pred_val_input_fn)
        val_pred = np.array([int(pred['classes'][0]) for pred in val_pred])
        # Calculate accuracy metrics
        train_acc = accuracy_score(train_targ, train_pred)
        val_acc = accuracy_score(val_targ, val_pred)
        print("period {} train acc: {:.3f}".format(period, train_acc))
        # Add our accuracies to running list
        train_app.append(train_acc)
        val_app.append(val_acc)
    print("Training done!")
    plot_acc(train_app, val_app)
    return (dnn_classifier)


# # 6) Training our model

# In[53]:


tf.logging.set_verbosity(tf.logging.ERROR)
classifier = train_dnn_classifier(periods = 25,
                                 learning_rate = 0.05,
                                 steps = 4000,
                                 batch_size = 75,
                                 hidden_units = [100, 100, 42],
                                 train_ex = train_ex_df,
                                 train_targ = train_targ_series,
                                 val_ex = xval_ex_df,
                                 val_targ = xval_targ_series)


# # 7) Making predictions for test dataset

# In[54]:


#test_ex_df = test_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "family_size", "deck_level"]]
#test_ex_df = test_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "family_size", "deck_level"]]
test_ex_df = test_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "family_size"]]
# Create a dummy series that will be compatible with our input_fn
test_targ_series = pd.Series(np.zeros(len(test_df), dtype=int))

pred_test_input_fn = lambda: input_fn(test_ex_df,
                                     test_targ_series,
                                     num_epochs = 1,
                                     shuffle = False)

test_pred = classifier.predict(input_fn = pred_test_input_fn)
test_pred = np.array([int(pred['classes'][0]) for pred in test_pred])


# In[ ]:


submission_df = test_df[["PassengerId"]]
submission_df = submission_df.assign(Survived=pd.Series(test_pred).values)
display(submission_df)
submission_df.to_csv('2018-03-31_submission_5.csv', index = False)


# In[ ]:




