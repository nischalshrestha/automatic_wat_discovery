#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This Kernel is my first serious attempt at practicing a classic machine learning workflow. I am a beginner in the field and I use this competition to experiment and practice, thus I hope to receive as many feedback as possible.
# 
# The steps of this workflow will be:
# * Exploratory Analysis: what is missing, how the variables distribute, how are they correlated.
# * Data Cleaning: because it is a dirty world
# * Feature engineering: given what I learned from the exploration, how do I create the best features?
# * Algorithm selection: given what I have, what are the best algorithms to go forward?
# * Model training: time to tune the model and train it
# 
# With this workflow, I reached a public score of 0.80382 (top 12% when) which makes me happy but reveals some crucial gaps in my understanding of the topic.
# 
# Things I have learned by doing what follows:
# * how to better explore the data to get insights
# * how to create new features from those insights
# * feature selection and cross-validation
# * how to test models
# 
# Things I want to learn next:
# * how to improve the model once I reach a certain result
# * error analysis and what to do next
# * stacking
# 
# Once again, thank you for reading and please share any disagreement you have my code.

# In[ ]:


import pandas as pd
import numpy as np

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]


# # Exploratory Analysis
# 
# Just to have a taste of what is in there
# 
# ## Missing Data

# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()
print('_'*40)
test_df.info()


# In[ ]:


train_df.describe()


# In[ ]:


test_df.describe()


# A few things are missing:
# * Age is missing 20% of the times in the train and 35% of the time in the test
# * Embarked feature missing twice in the train
# * Cabin missing 75% of the times in train and 78% of the times in test
# * Fare is missing once in the test
# 
# ## Numeric Features Distribution
# 
# Age, Fare, Parch (number of parents/children), and SibSp (number of siblings/ spouses)

# In[ ]:


fig, axes = plt.subplots(1, 2)

train_df.Age.hist(bins = 30, ax=axes[0])
test_df.Age.hist(bins = 30, ax=axes[1])


# In[ ]:


fig, axes = plt.subplots(1, 2)

train_df.Fare.hist(bins = 30, ax=axes[0])
test_df.Fare.hist(bins = 30, ax=axes[1])


# In[ ]:


fig, axes = plt.subplots(1, 2)

train_df.Parch.hist(bins = 20, ax=axes[0])
test_df.Parch.hist(bins = 20, ax=axes[1])


# In[ ]:


fig, axes = plt.subplots(1, 2)

train_df.SibSp.hist(bins = 20, ax=axes[0])
test_df.SibSp.hist(bins = 20, ax=axes[1])


# Some outliers, nothing crazy tho: the distributions are similar for test and train
# 
# ## Categorical Variables
# 
# Survived, Sex, Pclass, Embarked

# In[ ]:


print("Percentage of survival in the train set: {}%".format(round(sum(train_df.Survived)/train_df.Survived.count(), 2)))


# In[ ]:


print("In train: ")
print(train_df.Sex.value_counts())
print('_'*40)
print("In test: ")
print(test_df.Sex.value_counts())

fig, axes = plt.subplots(1, 2)

train_df.Sex.value_counts().plot(kind = "bar", ax=axes[0])
test_df.Sex.value_counts().plot(kind = "bar", ax=axes[1])


# In[ ]:


print("In train: ")
print(train_df.Pclass.value_counts())
print('_'*40)
print("In test: ")
print(test_df.Pclass.value_counts())

fig, axes = plt.subplots(1, 2)

train_df.Pclass.value_counts().plot(kind = "bar", ax=axes[0])
test_df.Pclass.value_counts().plot(kind = "bar", ax=axes[1])


# In[ ]:


print("In train: ")
print(train_df.Embarked.value_counts())
print('_'*40)
print("In test: ")
print(test_df.Embarked.value_counts())

fig, axes = plt.subplots(1, 2)

train_df.Embarked.value_counts().plot(kind = "bar", ax=axes[0])
test_df.Embarked.value_counts().plot(kind = "bar", ax=axes[1])


# A few issues:
# * The two classes are imbalanced, I have to be careful with that
# * Sex and Pclass are equally distributed
# * Small differences in the distribution of Embarked category between train and test
# 
# ## Correlations

# In[ ]:


train_df.corr()


# In[ ]:


test_df.corr()


# * Similar correlations in train and test.
# * In the training set it seems that class and fare are the most important features.
# * Probably one of the two is redundant since they are also very much correlated.
# 
# Let's have a look at some segmentations with the target variable

# In[ ]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# In[ ]:


train_df[["Sex", "Survived"]].groupby(['Sex'], 
                                      as_index=True).mean().sort_values(by='Survived', 
                                                                         ascending=False)


# In[ ]:


train_df[["SibSp", "Survived"]].groupby(['SibSp'], 
                                        as_index=True).mean().sort_values(by='Survived', 
                                                                           ascending=False)


# In[ ]:


train_df[["Parch", "Survived"]].groupby(['Parch'], 
                                        as_index=True).mean().sort_values(by='Survived', 
                                                                           ascending=False)


# In[ ]:


train_df[["Embarked", "Survived"]].groupby(['Embarked'], 
                                        as_index=True).mean().sort_values(by='Survived', 
                                                                           ascending=False)


# In[ ]:


g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=40)


# * Clearly the class and the gender play a role in determining the survival
# * It looks like a certain number of people traveling with you will help you surviving, but too many will kill you
# * I will not pretend to not know that the feature IsAlone is very famous and useful in this competition
# * Kids seem to survive
# 
# Let's see how these features relate to one another

# In[ ]:


for dataset in combine:
    df = dataset.groupby(['Sex', 'Pclass']).size().unstack(0)
    df['fem_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)


# In[ ]:


train_df[["Sex", "Pclass", "Survived"]].groupby(['Sex', 'Pclass'], 
                                        as_index=True).mean()


# The question may now arise: did they die because they were males or because they were in the third class? This calls for a new feature that I will create later on. 
# 
# There is a small difference in how the genders are distributed among the classes between train and test set but I am confident that the model will be robust enough to not care about that. (Maybe it is more appropriate to say that I hope that the model will be robust enough...)

# In[ ]:


for dataset in combine:
    df = dataset.groupby(['Sex', 'Parch']).size().unstack(0).fillna(0)
    df['male_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]]))
    df['fem_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)


# In[ ]:


train_df[["Sex", "Parch", "Survived"]].groupby(['Sex', 'Parch'], 
                                        as_index=True).mean()


# We have seen that Parch= 0 is most likely to not survive and here we observe that most of them are men. Again: which one is the relevant one?

# In[ ]:


for dataset in combine:
    df = dataset.groupby(['Sex', 'SibSp']).size().unstack(0).fillna(0)
    df['male_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]]))
    df['fem_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)


# In[ ]:


train_df[["Sex", "SibSp", "Survived"]].groupby(['Sex', 'SibSp'], 
                                        as_index=True).mean()


# As before, is it being male, being alone, or both that determines your chances of survival? Getting good ideas for the feature engineering phase.

# In[ ]:


grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid.map(plt.hist, 'Age', alpha=.5, bins=30)
grid.add_legend()


# In[ ]:


grid = sns.FacetGrid(train_df, col='Pclass', row='Sex', hue='Survived')
grid.map(plt.hist, 'Age', alpha=.5, bins=30)
grid.add_legend()


# * Being female seems to guarantee your survival only if you are in the first two classes
# * Being a baby seems to not be important for your survival if you are poor
# * Being a man sucks (only in this case probably, privilege all over the place otherwise) except if you are rich
# 
# Let's see the embarked feature.

# In[ ]:


for dataset in combine:
    df = dataset.groupby(['Sex', 'Embarked']).size().unstack(0)
    df['male_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]]))
    df['fem_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)


# In[ ]:


train_df[["Embarked", "Sex", "Survived"]].groupby(['Sex', 'Embarked'], 
                                        as_index=True).mean()


# In[ ]:


for dataset in combine:
    df = dataset.groupby(['Pclass', 'Embarked']).size().unstack(0)
    df['first_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    df['second_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    df['third_perc'] = (df[df.columns[2]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    print(df)
    print("_"*40)


# In[ ]:


train_df[["Embarked", "Pclass", "Survived"]].groupby(['Embarked', 'Pclass'], 
                                        as_index=True).mean()


# # Data Cleaning
# 
# Cabin is missing too many times to be useful, I will just change it with 1/0 values because maybe the fact that is not missing matter

# In[ ]:


for dataset in combine:
    fil1 = (dataset.Cabin.isnull())
    fil2 = (dataset.Cabin.notnull())
    dataset.loc[fil1, 'Cabin'] = 0
    dataset.loc[fil2, 'Cabin'] = 1
    dataset.Cabin = pd.to_numeric(dataset['Cabin'])

print(train_df.Cabin.value_counts())
print("_"*40)
print(test_df.Cabin.value_counts())


# Now I want to check if it relates with other things

# In[ ]:


train_df[['Cabin', 'Survived']].groupby(['Cabin'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# In[ ]:


for dataset in combine:
    df = dataset.groupby(['Sex', 'Cabin']).size().unstack(0)
    df['fem_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    df['male_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)


# In[ ]:


train_df[["Cabin", "Sex", "Survived"]].groupby(['Sex', 'Cabin'], 
                                        as_index=True).mean()


# In[ ]:


for dataset in combine:
    df = dataset.groupby(['Cabin', 'Pclass']).size().unstack(0)
    df['miss_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)


# In[ ]:


train_df[["Cabin", "Pclass", "Survived"]].groupby(['Pclass', 'Cabin'], 
                                        as_index=True).mean()


# It seems to matter a lot in terms of survival and is very related to the class.
# 
# The Embarked feature is missing only twice in the train and the Fare feature is missing once in the test. Putting them as 'missing' or creating a flag feature would introduce a sparse class in our data and this can lead to overfitting.
# 
# For the Embarked feature, I will just segment the data and impute with the mode of that segment
# 
# For the missing Fare, the same strategy but with a mean value.

# In[ ]:


train_df[train_df.Embarked.isnull()]


# In[ ]:


test_df[test_df.Fare.isnull()]


# I don't want to mix train and test in any way and I don't want to use the target variable to segment. So I will focus on female passengers of first class and traveling alone (also, they have the same ticket number, weird) for the Embarked feature in the train and on male passengers traveling alone of third class embarked in S with missing cabin.

# In[ ]:


fil = ((train_df.Pclass == 1) & (train_df.SibSp == 0) & (train_df.Parch == 0) 
       & (train_df.Sex == 'female'))
mis = train_df[fil].Embarked.mode()
print(mis)
fil = train_df.Embarked.isnull()
train_df.loc[fil, 'Embarked'] = 'C' #I new it was C from a previous run
print("_"*40)
print(train_df.Embarked.value_counts(dropna = False))


# In[ ]:


fil = ((test_df.Pclass == 3) & (test_df.SibSp == 0) & (test_df.Parch == 0) 
       & (test_df.Cabin == 0) & (test_df.Sex == 'male') & (test_df.Embarked == 'S'))
mis = round(test_df[fil].Fare.median(), 4)
print(mis)
fil = test_df.Fare.isnull()
test_df.loc[fil, 'Fare'] = mis
print("_"*40)
print(test_df.Fare.isnull().value_counts(dropna = False))


# Next, the Age feature is missing quite a bit and not enough to drop it (also, I don't want to). 
# 
# My strategy is going to be the following:
# 
# * segment the data according to some criteria
# * calculate median of the age of each segment
# * impute random values in the mean \pm a small variation of each segment
# * create a feature indicating that the age was missing
# * hope for the best
# 
# Let' start with the easy part: flagging the missing ages.

# In[ ]:


for dataset in combine:
    dataset['MisAge'] = 0
    fil = (dataset.Age.isnull())
    dataset.loc[fil, 'MisAge'] = 1

print(train_df.MisAge.value_counts())
print("_"*40)
print(test_df.MisAge.value_counts())
print("_"*40)
print("_"*40)

train_df[['MisAge', 'Survived']].groupby(['MisAge'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# The missing values seem to be missing for a reason: they didn't survive and nobody asked them their age.
# 
# Let me check the usual correlations with sex and class.

# In[ ]:


for dataset in combine:
    df = dataset.groupby(['MisAge', 'Sex']).size().unstack(0)
    df['miss_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]]))
    df['nomiss_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)


# In[ ]:


train_df[["Sex", "MisAge", "Survived"]].groupby(['Sex', 'MisAge'], 
                                        as_index=True).mean()


# In[ ]:


for dataset in combine:
    df = dataset.groupby(['MisAge', 'Pclass']).size().unstack(0)
    df['miss_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]]))
    df['nomiss_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)


# In[ ]:


train_df[["Pclass", "MisAge", "Survived"]].groupby(['Pclass', 'MisAge'], 
                                        as_index=True).mean()


# In[ ]:


train_df[["Cabin", "MisAge", "Survived"]].groupby(['Cabin', 'MisAge'], 
                                        as_index=True).mean()


# Now I want to find who are the people with missing age.

# In[ ]:


fil = (train_df.Age.isnull())
print("By class:")
print(train_df[fil].Pclass.value_counts())
print("_"*40)
print(train_df[train_df.MisAge == 0].Pclass.value_counts())
print("_"*40)
print("_"*40)
print("By sex:")
print(train_df[fil].Sex.value_counts())
print("_"*40)
print(train_df[train_df.MisAge == 0].Sex.value_counts())
print("_"*40)
print("_"*40)
print("By parents and children:")
print(train_df[fil].Parch.value_counts())
print("_"*40)
print(train_df[train_df.MisAge == 0].Parch.value_counts())
print("_"*40)
print("_"*40)
print("By spouse and siblings:")
print(train_df[fil].SibSp.value_counts())
print("_"*40)
print(train_df[train_df.MisAge == 0].SibSp.value_counts())


# It looks that a good segment would be focusing on the Class and Sex categories because the others will be too few to give a meaningful estimate. Let's see in the test.

# In[ ]:


fil = (test_df.Age.isnull())
print("By class:")
print(test_df[fil].Pclass.value_counts())
print("_"*40)
print(test_df[test_df.MisAge == 0].Pclass.value_counts())
print("_"*40)
print("_"*40)
print("By sex:")
print(test_df[fil].Sex.value_counts())
print("_"*40)
print(test_df[test_df.MisAge == 0].Sex.value_counts())
print("_"*40)
print("_"*40)
print("By parents and children:")
print(test_df[fil].Parch.value_counts())
print("_"*40)
print(test_df[test_df.MisAge == 0].Parch.value_counts())
print("_"*40)
print("_"*40)
print("By spouse and siblings:")
print(test_df[fil].SibSp.value_counts())
print("_"*40)
print(test_df[test_df.MisAge == 0].SibSp.value_counts())


# One more thing that could help is to use the title feature, which I have to create. Even if the feature engineering step will happen later, I think it will help to create the right segments for imputing the missing age.

# In[ ]:


# Extract the title from the name feature
for df in combine:
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
pd.crosstab(train_df['Title'], train_df['Sex'])


# In[ ]:


pd.crosstab(test_df['Title'], test_df['Sex'])


# In[ ]:


# handling the rare classes with class
for df in combine:
    df['Title'] = df['Title'].replace(['Mme', 'Countess','Dona'], 'Mrs')
    df['Title'] = df['Title'].replace(['Capt', 'Col','Don', 'Jonkheer', 'Rev', 
                                                 'Major', 'Sir'], 'Mr')
    df['Title'] = df['Title'].replace(['Mlle', 'Lady','Ms'], 'Miss')
    df.loc[(df.Sex == 'male') & (df.Title == 'Dr') , 'Title'] = 'Mr'
    df.loc[(df.Sex == 'female') & (df.Title == 'Dr') , 'Title'] =  'Mrs' 
    
pd.crosstab(train_df['Title'], train_df['Sex'])


# In[ ]:


pd.crosstab(test_df['Title'], test_df['Sex'])


# In[ ]:


fil = (test_df.Age.isnull())
print("By title:")
print(test_df[fil].Title.value_counts())
print("_"*40)
print(test_df[test_df.MisAge == 0].Title.value_counts())
print("_"*40)
print("_"*40)
fil = (train_df.Age.isnull())
print("By class:")
print(train_df[fil].Title.value_counts())
print("_"*40)
print(train_df[train_df.MisAge == 0].Title.value_counts())


# I think Title is the right feature to use to segment the age feature because it catches gender and age section. Moreover, I will also segment per class

# In[ ]:


np.random.seed(452) #reproducibility


# In[ ]:


for df in combine:
    titles = list(set(df.Title))
    classes = list(set(df.Pclass))
    for title in titles:
        for cl in classes:
            fil = (df.Title == title) & (df.Pclass == cl)
            med_age = df[fil].Age.dropna().median()
            var_age = med_age / 5
            mis_age = df[fil].MisAge.sum()
            df.loc[fil & (df.Age.isnull()), 'Age'] = np.random.randint(int(med_age - var_age - 1), 
                                                                       int(med_age + var_age), mis_age)
        
train_df.Age.describe()


# In[ ]:


test_df.Age.describe()


# In[ ]:


train_df.info()
print('_'*40)
test_df.info()


# I am happy with the current state of my data now, I could focus on the outliers in Parch and SpSib but we all know I am going to use IsAlone as a feature, so I don't spend any more time in cleaning the data and move to a funnier step
# 
# # Feature Engineering
# 
# I got some ideas thanks to the data exploration, I will try to produce something useful.
# 
# ### Not really feature engineering
# 
# This is the moment where I just convert category into numbers because it helps.

# In[ ]:


# Convert Sex to numerical
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'male':1 , 'female':2}).astype(int)

train_df.sample(5)


# In[ ]:


# Convert Embarked to numerical
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':1 , 'C':2, 'Q':3}).astype(int)

train_df.sample(5)


# This was easy and it usually helps to deal with numbers rather than strings.
# 
# ### Creating new features
# 
# I have already created the title feature, let's see how it relates to the other variables

# In[ ]:


train_df[['Title', 'Survived']].groupby(['Title'], as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# In[ ]:


train_df.Title.hist()


# In[ ]:


train_df.Title.value_counts()


# In[ ]:


for df in combine:
    df['Title'] = df['Title'].map({'Mr':1 , 'Mrs':2, 'Miss':3, 'Master':4}).astype(int)

train_df.sample(5)


# I don't like this feature very much because it doesn't really say anything more than what I know already (adult males are going to die).
# 
# However, it was useful for imputing the missing ages.
# 
# Another feature I can create is the popular IsAlone

# In[ ]:


for df in combine:
    df['IsAlone'] = 0
    fil = (df.SibSp == 0) & (df.Parch == 0)
    df.loc[fil, 'IsAlone'] = 1
    
print(train_df.IsAlone.value_counts())
print("_"*40)
print(test_df.IsAlone.value_counts())


# In[ ]:


#checking correlation with the target
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# In[ ]:


for dataset in combine:
    df = dataset.groupby(['Sex', 'IsAlone']).size().unstack(0)
    df['fem_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]]))
    df['male_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]]))
    print(df)
    print("_"*40)


# In[ ]:


train_df[["Sex", "IsAlone", "Survived"]].groupby(['IsAlone','Sex'], 
                                        as_index=True).mean()


# In[ ]:


for dataset in combine:
    df = dataset.groupby(['Pclass', 'IsAlone']).size().unstack(0)
    df['first_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    df['second_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    df['third_perc'] = (df[df.columns[2]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    print(df)
    print("_"*40)


# In[ ]:


train_df[["Pclass", "IsAlone", "Survived"]].groupby(['IsAlone', 'Pclass'], 
                                        as_index=True).mean()


# Not as strong as other correlations, but still clear enough to keep this feature for our models.
# 
# Before I have seen that being a kid helps for your survival, so I create a feature for it.

# In[ ]:


for df in combine:
    df['IsKid'] = 0
    fil = (df.Age < 16)
    df.loc[fil, 'IsKid'] = 1
    
print(train_df.IsKid.value_counts())
print("_"*40)
print(test_df.IsKid.value_counts())


# In[ ]:


#checking correlation with the target
train_df[['IsKid', 'Survived']].groupby(['IsKid'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# Again, we see a correlation that might help the models.
# 
# ### Making categories out of continuous variables
# 
# One thing that can help is to have an Age category and a Fare category. 
# 
# Let's start with Age by looking again at the distribution.

# In[ ]:


g = sns.FacetGrid(train_df[train_df.Age > -1], hue='Survived')
g.map(plt.hist, 'Age', bins=30, alpha = 0.6)
g.add_legend()


# I thus see that the cuts can be the following

# In[ ]:


bins = [0, 16, 32, 48, 81] #I just want to avoid the sparse class at 64-80

for df in combine:
    df['AgeBin'] = pd.cut(df['Age'], bins)
    #df['AgeBin'] = pd.to_numeric(df['AgeBin'])
    
print(train_df.AgeBin.value_counts())
print("_"*40)
print(test_df.AgeBin.value_counts())


# In[ ]:


#checking correlation with the target
train_df[['AgeBin', 'Survived']].groupby(['AgeBin'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# In[ ]:


#same thing but with labels
bins = [0, 16, 32, 48, 81]
names = [0, 1, 2, 3]

for df in combine:
    df['AgeBin'] = pd.cut(df['Age'], bins, labels = names)
    df['AgeBin'] = pd.to_numeric(df['AgeBin'])
    
print(train_df.AgeBin.value_counts())
print("_"*40)
print(test_df.AgeBin.value_counts())


# The Fare feature has some outliers and it does not take into account that a passenger might not be alone, I am tempted to categorize the fare per person.
# 
# (This excludes people traveling with non-relatives and we know that Di Caprio was not alone) 
# 
# (but even paid for all that matter)
# 
# (and he died)
# 
# First, I create a feature for the number of family members on board. Then I calculate the fare per person.

# In[ ]:


for df in combine:
    df['NumFam'] = df['SibSp'] + df['Parch'] + 1
    df['FarePP'] = df['Fare'] / df['NumFam']
    
fig, axes = plt.subplots(1, 2)

train_df.NumFam.hist(ax=axes[0])
test_df.NumFam.hist(ax=axes[1])


# In[ ]:


fig, axes = plt.subplots(1, 2)

train_df.FarePP.hist(ax=axes[0], bins=20)
test_df.FarePP.hist(ax=axes[1], bins=20)


# * The NumFam has too many sparse values and I think the IsAlone feature already captures that kind of information
# * The FarePP feature can be put in a category using qcut

# In[ ]:


for df in combine:
    df['FareCat'] = pd.qcut(df.FarePP, 4)
    
print(train_df.FareCat.value_counts())
print("_"*40)
print(test_df.FareCat.value_counts())


# In[ ]:


#same thing with labels
labels = [0, 1, 2, 3]

for df in combine:
    df['FareCat'] = pd.qcut(df.FarePP, 4, labels=labels)
    df['FareCat'] = pd.to_numeric(df['FareCat'])
    
print(train_df.FareCat.value_counts())
print("_"*40)
print(test_df.FareCat.value_counts())


# In[ ]:


train_df[['FareCat', 'Survived']].groupby(['FareCat'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# In[ ]:


for dataset in combine:
    df = dataset.groupby(['Pclass', 'FareCat']).size().unstack(0).fillna(0)
    df['first_perc'] = (df[df.columns[0]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    df['second_perc'] = (df[df.columns[1]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    df['third_perc'] = (df[df.columns[2]]/(df[df.columns[0]] + df[df.columns[1]] + df[df.columns[2]]))
    print(df)
    print("_"*40)


# It seems to me that the FareCat is redundant with the class.
# 
# Next, I want something to distinguish a large family (which before looked like it was going to not survive), from a small one.

# In[ ]:


for df in combine:
    df['FamSize'] = 0 #alone people
    df.loc[(df.NumFam > 1), 'FamSize'] = 1 #small families
    df.loc[(df.NumFam > 3), 'FamSize'] = 2 #medium families
    df.loc[(df.NumFam > 5), 'FamSize'] = 3 #big families

print(train_df.FamSize.value_counts())
print("_"*40)
print(test_df.FamSize.value_counts())


# In[ ]:


train_df[['FamSize', 'Survived']].groupby(['FamSize'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# In[ ]:


train_df[["Pclass", "FamSize", "Survived"]].groupby(['FamSize', 'Pclass'], 
                                        as_index=True).mean()


# In[ ]:


train_df[["Sex", "FamSize", "Survived"]].groupby(['FamSize', 'Sex'], 
                                        as_index=True).mean()


# It seems it gives some more indication. It also include the IsAlone feature, which I will just use in combination with the class.
# 
# ### Indicator features
# 
# I saw before some correlations with the survival rate and gender or class. I also observed that there was no clear answer on which variable was more important. So I now create the following features to express the combinations of different categories:
# 
# * Sex and Pclass
# * PClass and IsAlone
# * Cabin and PClass
# * MisAge and PClass
# * IsKid and PClass
# * Embarked and PClass
# * MisAge and Cabin
# * Sex and IsAlone
# * Cabin and Sex

# In[ ]:


for df in combine:
    df['Se_Cl'] = 0
    df.loc[((df.Sex == 1) & (df.Pclass == 1)) , 'Se_Cl'] = 1 #rich male
    df.loc[((df.Sex == 1) & (df.Pclass == 2)) , 'Se_Cl'] = 2 #avg male
    df.loc[((df.Sex == 1) & (df.Pclass == 3)) , 'Se_Cl'] = 3 #poor male
    df.loc[((df.Sex == 2) & (df.Pclass == 1)) , 'Se_Cl'] = 4 #rich female
    df.loc[((df.Sex == 2) & (df.Pclass == 2)) , 'Se_Cl'] = 5 #avg female
    df.loc[((df.Sex == 2) & (df.Pclass == 3)) , 'Se_Cl'] = 6 #poor female 
    
print(train_df.Se_Cl.value_counts())
print("_"*40)
print(test_df.Se_Cl.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['Se_Cl', 'Survived']].groupby(['Se_Cl'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# In[ ]:


for df in combine:
    df['Cl_IA'] = 0
    df.loc[((df.IsAlone == 1) & (df.Pclass == 1)) , 'Cl_IA'] = 1 #rich alone
    df.loc[((df.IsAlone == 1) & (df.Pclass == 2)) , 'Cl_IA'] = 2 #avg alone
    df.loc[((df.IsAlone == 1) & (df.Pclass == 3)) , 'Cl_IA'] = 3 #poor alone
    df.loc[((df.IsAlone == 0) & (df.Pclass == 1)) , 'Cl_IA'] = 4 #rich with family
    df.loc[((df.IsAlone == 0) & (df.Pclass == 2)) , 'Cl_IA'] = 5 #avg with family
    df.loc[((df.IsAlone == 0) & (df.Pclass == 3)) , 'Cl_IA'] = 6 #poor with family 
    
    
print(train_df.Cl_IA.value_counts())
print("_"*40)
print(test_df.Cl_IA.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['Cl_IA', 'Survived']].groupby(['Cl_IA'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# In[ ]:


for df in combine:
    df['Ca_Cl'] = 0
    df.loc[((df.Cabin == 0) & (df.Pclass == 1)) , 'Ca_Cl'] = 1 #rich no cabin
    df.loc[((df.Cabin == 0) & (df.Pclass == 2)) , 'Ca_Cl'] = 2 #avg no cabin
    df.loc[((df.Cabin == 0) & (df.Pclass == 3)) , 'Ca_Cl'] = 3 #poor no cabin
    df.loc[((df.Cabin == 1) & (df.Pclass == 1)) , 'Ca_Cl'] = 4 #rich with cabin
    df.loc[((df.Cabin == 1) & (df.Pclass == 2)) , 'Ca_Cl'] = 5 #avg with cabin
    df.loc[((df.Cabin == 1) & (df.Pclass == 3)) , 'Ca_Cl'] = 6 #poor with cabin
    
print(train_df.Ca_Cl.value_counts())
print("_"*40)
print(test_df.Ca_Cl.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['Ca_Cl', 'Survived']].groupby(['Ca_Cl'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# In[ ]:


for df in combine:
    df['MA_Cl'] = 0
    df.loc[((df.MisAge == 0) & (df.Pclass == 1)) , 'MA_Cl'] = 1 #rich with age
    df.loc[((df.MisAge == 0) & (df.Pclass == 2)) , 'MA_Cl'] = 2 #avg with age
    df.loc[((df.MisAge == 0) & (df.Pclass == 3)) , 'MA_Cl'] = 3 #poor with age
    df.loc[((df.MisAge == 1) & (df.Pclass == 1)) , 'MA_Cl'] = 4 #rich without age
    df.loc[((df.MisAge == 1) & (df.Pclass == 2)) , 'MA_Cl'] = 5 #avg without age
    df.loc[((df.MisAge == 1) & (df.Pclass == 3)) , 'MA_Cl'] = 6 #poor without age
    
print(train_df.MA_Cl.value_counts())
print("_"*40)
print(test_df.MA_Cl.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['MA_Cl', 'Survived']].groupby(['MA_Cl'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# In[ ]:


for df in combine:
    df['IK_Cl'] = 0
    df.loc[((df.IsKid == 0) & (df.Pclass == 1)) , 'IK_Cl'] = 1 #rich adult
    df.loc[((df.IsKid == 0) & (df.Pclass == 2)) , 'IK_Cl'] = 2 #avg adult
    df.loc[((df.IsKid == 0) & (df.Pclass == 3)) , 'IK_Cl'] = 3 #poor adult
    df.loc[((df.IsKid == 1) & (df.Pclass == 1)) , 'IK_Cl'] = 4 #rich kid
    df.loc[((df.IsKid == 1) & (df.Pclass == 2)) , 'IK_Cl'] = 5 #avg kid
    df.loc[((df.IsKid == 1) & (df.Pclass == 3)) , 'IK_Cl'] = 6 #poor kid
    
print(train_df.IK_Cl.value_counts())
print("_"*40)
print(test_df.IK_Cl.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['IK_Cl', 'Survived']].groupby(['IK_Cl'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# In[ ]:


#for embarked and class I will just multiply them
for df in combine:
    df["Em_Cl"] = df["Embarked"] * df["Pclass"]

print(train_df.Em_Cl.value_counts())
print("_"*40)
print(test_df.Em_Cl.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['Em_Cl', 'Survived']].groupby(['Em_Cl'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# In[ ]:


for df in combine:
    df['Se_Ca'] = 0
    df.loc[((df.Sex == 1) & (df.Cabin == 0)) , 'Se_Ca'] = 1 #male without cabin
    df.loc[((df.Sex == 1) & (df.Cabin == 1)) , 'Se_Ca'] = 2 #male with cabin
    df.loc[((df.Sex == 2) & (df.Cabin == 0)) , 'Se_Ca'] = 3 #female without cabin
    df.loc[((df.Sex == 2) & (df.Cabin == 1)) , 'Se_Ca'] = 4 #female with cabin
    
print(train_df.Se_Ca.value_counts())
print("_"*40)
print(test_df.Se_Ca.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['Se_Ca', 'Survived']].groupby(['Se_Ca'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# In[ ]:


for df in combine:
    df['MA_Ca'] = 0
    df.loc[((df.MisAge == 0) & (df.Cabin == 0)) , 'MA_Ca'] = 1 #Age no Cabin
    df.loc[((df.MisAge == 0) & (df.Cabin == 1)) , 'MA_Ca'] = 2 #Age and Cabin
    df.loc[((df.MisAge == 1) & (df.Cabin == 0)) , 'MA_Ca'] = 3 #No Age no Cabin
    df.loc[((df.MisAge == 1) & (df.Cabin == 1)) , 'MA_Ca'] = 4 #No Age but Cabin
    
print(train_df.MA_Ca.value_counts())
print("_"*40)
print(test_df.MA_Ca.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['MA_Ca', 'Survived']].groupby(['MA_Ca'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# In[ ]:


for df in combine:
    df['Se_IA'] = 0
    df.loc[((df.Sex == 1) & (df.IsAlone == 0)) , 'Se_IA'] = 1 #Male with family
    df.loc[((df.Sex == 1) & (df.IsAlone == 1)) , 'Se_IA'] = 2 #Male without family
    df.loc[((df.Sex == 2) & (df.IsAlone == 0)) , 'Se_IA'] = 3 #Female with family
    df.loc[((df.Sex == 2) & (df.IsAlone == 1)) , 'Se_IA'] = 4 #Female without family
    
print(train_df.Se_IA.value_counts())
print("_"*40)
print(test_df.Se_IA.value_counts())
print("_"*40)
print("_"*40)

#see correlation with target
train_df[['Se_IA', 'Survived']].groupby(['Se_IA'], 
                                         as_index=True).mean().sort_values(by='Survived', 
                                                                            ascending=False)


# I am keeping them excepet for Ca_Cl, MA_Cl, IK_Cl, and Em_Cl either because they don't give new insights as hoped or because of sparse classes.
# 
# 
# # Algorithm Selection
# 
# It is a classification problem and I feel fairly confident in using the following algorithms:
# 
# * Logistic regression
# * Decision tree
# * Support vector machine
# * Naive bayes
# * Ensembles of the previous one
# 
# I will thus import the modules I need and select the features to feed them
# 

# In[ ]:


train_df.describe()


# In[ ]:


from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


train_df.columns


# In[ ]:


features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Title', 'AgeBin', 'MisAge', 'IsKid', 
            'FamSize', 'Se_Cl', 'Cl_IA', 'Se_Ca', 'MA_Ca', 'Se_IA']

y = train_df['Survived'].copy()
X = train_df[features].copy()
test = test_df[features].copy()

X.head()


# In[ ]:


y.head()


# ## Consistency and Robustness
# 
# I have imported a lot of Algorithms and realistically I will not be patient enough to tune all of them. 
# 
# Thus, as a first selection, I want to keep only those that are more consistent. In other words, I want to keep only the algorithms that are more stable when I cross-validate.
# 
# Moreover, I want to find also the more robust: those that perform well across different metrics.

# In[ ]:


# classifier list
clf_list = [DecisionTreeClassifier(), 
            RandomForestClassifier(), 
            AdaBoostClassifier(), 
            GradientBoostingClassifier(), 
            XGBClassifier(),
            Perceptron(),
            LogisticRegression(), 
            SVC(), 
            LinearSVC(), 
            KNeighborsClassifier(), 
            GaussianNB(),
            SGDClassifier()
           ]


# In[ ]:


mdl = []
bias_acc = []
var_acc = []
bias_f1 = []
var_f1 = []
bias_auc = []
var_auc = []

acc_scorer = make_scorer(f1_score)

for clf in clf_list:
    model = clf.__class__.__name__
    res = cross_val_score(clf, X, y, scoring='accuracy', cv = 5)
    score = round(res.mean() * 100, 3)
    var = round(res.std(), 3)
    bias_acc.append(score)
    var_acc.append(var)
    res = cross_val_score(clf, X, y, scoring=acc_scorer, cv = 5)
    score = round(res.mean() * 100, 3)
    var = round(res.std(), 3)
    bias_f1.append(score)
    var_f1.append(var)
    res = cross_val_score(clf, X, y, scoring='roc_auc', cv = 5)
    score = round(res.mean() * 100, 3)
    var = round(res.std(), 3)
    bias_auc.append(score)
    var_auc.append(var)
    mdl.append(model)
    print(model)
    
#create a small df with the scores
robcon = pd.DataFrame({'Model': mdl, 'Bias_acc':bias_acc,'Variance_acc':var_acc, 
                       'Bias_f1':bias_f1,'Variance_f1':var_f1, 'Bias_auc':bias_auc,'Variance_auc':var_auc,
                      })
robcon = robcon[['Model','Bias_acc','Variance_acc', 'Bias_f1','Variance_f1','Bias_auc','Variance_auc' ]]
robcon


# In[ ]:


print("Best for accuracy")
print(robcon[['Model','Bias_acc']].sort_values(by= 'Bias_acc', ascending=False).head(6))
print("_"*40)
print("Best for f1")
print(robcon[['Model','Bias_f1']].sort_values(by= 'Bias_f1', ascending=False).head(6))
print("_"*40)
print("Best for roc_auc")
print(robcon[['Model','Bias_auc']].sort_values(by= 'Bias_auc', ascending=False).head(6))
print("_"*40)
print("Least variance for accuracy")
print(robcon[['Model','Variance_acc']].sort_values(by= 'Variance_acc').head(6))
print("_"*40)
print("Least variance for f1")
print(robcon[['Model','Variance_f1']].sort_values(by= 'Variance_f1').head(6))
print("_"*40)
print("Least variance for roc_auc")
print(robcon[['Model','Variance_auc']].sort_values(by= 'Variance_auc').head(6))


# * I will move forward with SVC, RandomForest, XGB, AdaBoost, and LogisticRegression.
# 
# * LogisticRegression might benefit from the creation of dummy variables.

# ## Hyperparameters tuning with feature selection
# 
# I want to find the best set of parameters for each algorithm. I will use GridSearchCV so that cross-validation is included. This is fairly time-consuming, so I have to be mindful of that.
# 
# Moreover, I am not sure that every algorithm likes having so many features. So will also perform some feature selection.
# 
# I want to test my results on previously unseen data, so I will start by splitting

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=895)


# In[ ]:


from sklearn.feature_selection import RFECV


# In[ ]:


X_train.columns


# ### Logistic Regression
# 
# I will tune the C parameter, which is the inverse of the regularization strength, and the tolerance for stopping.

# In[ ]:


# feature selection
FeatSel_log = RFECV(LogisticRegression(), step = 1, scoring = 'roc_auc', cv = 10)
FeatSel_log.fit(X_train, y_train)

BestFeat_log = X_train.columns.values[FeatSel_log.get_support()]
BestFeat_log


# In[ ]:


# define the parameters grid
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
             'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
             'random_state' : [42]}

# create the grid
grid_log = GridSearchCV(LogisticRegression(), param_grid, cv = 10, scoring= 'roc_auc')

#training
get_ipython().magic(u'time grid_log.fit(X_train[BestFeat_log], y_train)')

#let's see the best estimator
best_log = grid_log.best_estimator_
print(best_log)
print("_"*40)
#with its score
print(np.abs(grid_log.best_score_))
print("_"*40)
#accuracy on test
predictions = best_log.predict(X_test[BestFeat_log])
accuracy_score(y_true = y_test, y_pred = predictions)


# ### SVC
# 
# I will tune the type of kernel, the penalty parameter, and the tolerance for stopping criteria.
# 
# SVC do not have a coef or feature_importance, so I will just use them all.

# In[ ]:


# define the parameters grid with NORMAL
param_grid = {'C': np.arange(1,10),
             'tol': [0.0001, 0.001, 0.01, 0.1, 1],
             'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
             'random_state': [42]}

# create the grid
grid_SVC = GridSearchCV(SVC(), param_grid, cv = 10, scoring= 'roc_auc')

#training
get_ipython().magic(u'time grid_SVC.fit(X_train, y_train)')

#let's see the best estimator
best_SVC = grid_SVC.best_estimator_
print(best_SVC)
print("_"*40)
#with its score
print(np.abs(grid_SVC.best_score_))
#accuracy on test
predictions = best_SVC.predict(X_test)
accuracy_score(y_true = y_test, y_pred = predictions)


# ### AdaBoost 
# 
# This is a boosting algorithm, so it trains constrained learners in sequence and each learner learns from the mistakes of the previous one. Then it combines them into a single unconstrained learner. The base estimator is again a tree.
# 
# I will tune the type of algorithm, the number of estimators, and the learning rate

# In[ ]:


# feature selection
FeatSel_ada = RFECV(AdaBoostClassifier(), step = 1, scoring = 'roc_auc', cv = 10)
FeatSel_ada.fit(X_train, y_train)

BestFeat_ada = X_train.columns.values[FeatSel_ada.get_support()]
BestFeat_ada


# In[ ]:


# define the parameters grid
param_grid = {'n_estimators': np.arange(50, 500, 50),
             'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 2],
             'algorithm': ['SAMME', 'SAMME.R'],
             'random_state': [42]}

# create the grid
grid_ada = GridSearchCV(AdaBoostClassifier(), param_grid, cv = 10, scoring= 'roc_auc')

#training
get_ipython().magic(u'time grid_ada.fit(X_train[BestFeat_ada], y_train)')

#let's see the best estimator
best_ada = grid_ada.best_estimator_
print(best_ada)
print("_"*40)
#with its score
print(np.abs(grid_ada.best_score_))
#accuracy on test
predictions = best_ada.predict(X_test[BestFeat_ada])
accuracy_score(y_true = y_test, y_pred = predictions)


# ### Random Forest
# 
# This is a bagging algorithm, so it trains a lot of unconstrained learners in parallel and then combines them.
# 
# I will tune the number of trees, their depth, and the maximum number of features to determine a split.

# In[ ]:


# feature selection
FeatSel_for = RFECV(RandomForestClassifier(), step = 1, scoring = 'roc_auc', cv = 10)
FeatSel_for.fit(X_train, y_train)

BestFeat_for = X_train.columns.values[FeatSel_for.get_support()]
BestFeat_for


# In[ ]:


# define the parameters grid
param_grid = {'n_estimators': np.arange(10, 100, 10),
             'max_depth': np.arange(2,20),
             'max_features' : ['auto', 'log2', None],
              'criterion' : ['gini', 'entropy'],
             'random_state' : [42]}

# create the grid
grid_forest = GridSearchCV(RandomForestClassifier(), param_grid, cv = 10, scoring= 'roc_auc')

#training
get_ipython().magic(u'time grid_forest.fit(X_train[BestFeat_for], y_train)')

#let's see the best estimator
best_forest = grid_forest.best_estimator_
print(best_forest)
print("_"*40)
#with its score
print(np.abs(grid_forest.best_score_))
#accuracy on test
predictions = best_forest.predict(X_test[BestFeat_for])
accuracy_score(y_true = y_test, y_pred = predictions)


# ### XGBoost
# 
# It is a regularized boosting technique and it is very popular on Kaggle. 
# 
# I admit I don't understand it as much as the other algorithms but I will try to tune the number of trees, the learning rate and the max depth of the trees.

# In[ ]:


# feature selection
FeatSel_XGB = RFECV(XGBClassifier(), step = 1, scoring = 'roc_auc', cv = 10)
FeatSel_XGB.fit(X_train, y_train)

BestFeat_XGB = X_train.columns.values[FeatSel_XGB.get_support()]
BestFeat_XGB


# In[ ]:


# define the parameters grid
param_grid = {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 2],
             'max_depth': np.arange(2,10),
             'n_estimators': np.arange(50, 500, 50),
             'random_state': [42]}

# create the grid
grid_XGB = GridSearchCV(XGBClassifier(), param_grid, cv = 10, scoring= 'roc_auc')

#training
get_ipython().magic(u'time grid_XGB.fit(X_train[BestFeat_XGB], y_train)')

#let's see the best estimator
best_XGB = grid_XGB.best_estimator_
print(best_XGB)
print("_"*40)
#with its score
print(np.abs(grid_XGB.best_score_))
#accuracy on test
predictions = best_XGB.predict(X_test[BestFeat_XGB])
accuracy_score(y_true = y_test, y_pred = predictions)


# In conclusion, the order of scoring is the following:
# 
# * SVC
# * XGBoost
# * RandomForest
# * AdaBoost
# * LogisticRegression
# 
# I can also see that RandomForest and XGB are using fewer features than the others to get their results. I am surprised by the absence of Sex in the XGB features, but it is included in the Title.
# 
# Submitting the results now would give me the following scores:
# 
# * LogisticRegressor: 0.78468 (with dummy variables it goes up to 0.79904)
# * SVC: 0.78468
# * RandomForest: 0.77511
# * AdaBoost: 0.78947
# * XGBoost: 0.80382
# 
# Which is nice (top 12% at the time of submission), but I believe that it can't be too difficult to improve it.

# # Conclusions
# 
# This notebook clarified a few methodologies and finally allowed me to crack my personal goal of 80% in accuracy. A few more considerations before concluding.
# 
# Interestingly, if instead of imputing the missing age I just make categories out of them (included a "missing" category), the result actually improves for some algorithms (RandomForest above all). This makes sense to me because imputing missing values is just reinforcing an existing pattern.
# 
# One quick way of getting a better result is to use both datasets to impute the missing values, which I consider unrealistic since the test dataset is normally not available in a real situation and it gives me the feeling that I am using the solution to find it (if it makes any sense).
# 
# As said in the introduction, I would like to start from this result to better understand how to analyze the misclassified entries (which is not difficult) and, most importantly, what can I do once I get insights from that analysis.
# 
# Moreover, I have seen in other Kernels here on Kaggle that stacking algorithms are very promising and, since I don't know much about it, it is my plan to eventually focus on improving my result with this technique.
# 
# Thank you for reading and for any kind of feedback you might have for me. I hope this can be of any help for beginners like myself out there.

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": best_XGB.predict(test[BestFeat_XGB])
    })
submission.to_csv('submission.csv', index=False)

