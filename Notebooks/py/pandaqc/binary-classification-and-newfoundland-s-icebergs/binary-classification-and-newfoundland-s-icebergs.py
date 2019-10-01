#!/usr/bin/env python
# coding: utf-8

# # This notebook is still a **work in progress**
# 
# Just getting started with Machine Learning. This is my **first submission** on Kaggle.
# Learned a lot from **[Heads or Tails' notebook 'Pytanic`][1]**. Make sure you check his work !
# 
# A **k-nearest neighbors classifier** was used to prepare the prediction.
# It scores **0.78947** on the leaderboard.
# 
# ![Gigantic iceberg in Canada's Newfoundland, aller than the one which sank the Titanic][2]
#   [1]: https://www.kaggle.com/headsortails/pytanic
#   [2]: http://www.telegraph.co.uk/content/dam/video_previews/5/5/55ahf1yte6wfdklz83ylrhvip0f8d1yr-large.jpg

# ## 2017-06-24 update
# First model is **overfitted**. Cross-validated accuracy on the training set is **0.7934**. But submission scored only **0.7655**. It means that our model does not generalize very well.<br\>
# To reduce overfitting we dropped the **'deck' dummy features** and kept only one boolean feature **'deck_known'**.
# Kept KNN classifier but changed hyper parameter n=18 (instead of n=25).
# Cross-validated accuracy improves to **0.7979** (+0.0045). And Submission scores now **0.7751** (+0.0183).
# 
# ## 2017-06-27 update
# To improve generalization we have **cut down** the number of features, focusing on the ones really correlated with the survival rate.<br\>
# Introducing **age groups** did **not** improve the accuracy of the model versus using a rescaled age feature.<br\>
# The '**title**' feature improved the accuracy significantly though.<br\>
# Then multiple submissions have been made with different **hyper parameter** values (n_neighbors=10 to 25).
# All those improvement led to a new score of **0.78947**.
# 
# ## TODO
# - rework layout of the notebook
# - explore other classifiers (logistic regression, SVM)

# In[ ]:


# imports

# data analysis
import pandas as pd
import numpy as np
from scipy import stats, integrate

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error


# In[ ]:


# configuration

# seaborn
sns.set(color_codes=True)


# # 1. Loading Data

# In[ ]:


# load datasets as DataFrames
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# combine datasets to make it easier to run operations on both sets together
test_df['Survived'] = np.NaN
combined_df = pd.concat([train_df, test_df], ignore_index=True)


# In[ ]:


# take a look at the data
combined_df.head()


# In[ ]:


# get information on the datasets
combined_df.info()
print("------------------------")
combined_df.info()


# # 2. Data preprocessing

# ## 2.1 Data cleaning

# 2.1.1 Missing **Age** values
# 
# **Age** is missing for **20% of passengers**.
# 
# The **median age** is **28** years old overall.<br\>
# The **median age** for **females** is **27** years old.<br\>
# The **median age** for **males** is **28** years old.<br\>
# 
# The [0 to 5] age group is **twice more likely to survive** (67%) than the [20 to 25] age group (34%).
# 
# Among the 8 passengers **over 65 years old**, **only 1** survived. Ironically he's the oldest one (80 y.o.).

# In[ ]:


# first create a new feature to track passengers whose age is known (those passengers are probably more likely to have survived)
# might be useful later on
combined_df['age_known'] = combined_df.Age.notnull()

# create 12"x 12" figure
fig = plt.figure(figsize=(12,12))

# create "5-years large" bins (0-5, 5-10, 10-15...)
age_bins = np.arange(0, 90, 5)

# plot age distribution histogram
#-------------------------------------
# filter out missing ages
age_not_null_series = combined_df['Age'][combined_df.Age.notnull()]

plt.subplot(211)
ax1 = sns.distplot(age_not_null_series, bins=age_bins, kde=False,
                  hist_kws=dict(edgecolor="k", linewidth=1))

plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age distribution')
plt.grid(False)

# plot Survived per Age group
#----------------------------------
# filter out missing ages from training set
age_not_null_series_train = train_df['Age'][combined_df.Age.notnull()]

# create a new feature that stores passengers age group (0 to 5 years old, 5 to 10, etc.)
train_df['age_group'] = pd.cut(age_not_null_series_train, bins=age_bins, include_lowest=True, right=False)
# compute mean for each age group
age_group_surv_df = train_df[['age_group', 'Survived']].groupby(by='age_group', as_index=False).mean()
age_group_surv_df.columns = ['age_group', 'surv_pct']

plt.subplot(212)
ax2 = sns.barplot(x='age_group', y='surv_pct', color='salmon', data=age_group_surv_df, linewidth=1, edgecolor='black')

plt.xlabel('Age group')
plt.ylabel('Survived')
plt.title('Survived per Age group')

sns.plt.show()


# In[ ]:


# get details of Survived percentage per age group
age_group_surv_df


# In[ ]:


# take a closer look at passengers over 65 years old
train_df[['Age', 'Survived']][train_df['Age'] > 65]


# In[ ]:


# get median age
combined_df['Age'].median()


# In[ ]:


# get median age per sex
combined_df[['Sex','Age']].groupby(by='Sex').median()


# How should we **fill in** missing age values ?<br\>
# We could use **mean** or **median inputation**.
# 
# But we can probaly do even better if we can find a **correlation** between the age and other features.
# 
# We know that the **sex is irrelevant** since males and females have roughly the **same** median age.<br\>
# Let's look at the following two features:
#  - the **class** (older people tend to be wealthier)
#  - the passenger **title** ('Mr', 'Mrs' and especially 'Miss') that can be **extracted** from his **name**
# 

# In[ ]:


# compute median age, number of observations and Survived % per class
combined_df[['Age', 'Pclass', 'Survived']].groupby(by='Pclass').agg({'Age':['median','count'], 'Survived' : 'mean'})


# In[ ]:


# extract the title ('Mr', 'Mrs', 'Miss', 'Master', ...) from the Name feature
# use a regular expression that gets the first word '(\w+)' finishing by a period '\.
combined_df['title_extracted'] = combined_df['Name'].str.extract('(\w+)\.', expand=False)

# Replacing 'Mme' (Mrs. in french) by 'Mrs'
combined_df['title'] = combined_df['title_extracted'].str.lower()

# Replacing 'Mme' (Mrs. in french) by 'Mrs'
combined_df['title'] = combined_df['title'].replace('mme', 'mrs')

# Replacing 'Mlle' - french for Miss - and 'Ms' (only one in the dataset and she's 28) by 'Miss' 
combined_df['title'] = combined_df['title'].replace(['ms','mlle'], 'miss')

# All other titles are military/religion/honorific titles, we group them under a 'Notable' title
# Note: we do not try to separate male and female notable titles here.
combined_df['title'] = combined_df['title'].replace(['master','rev','dr','col','major','lady','countess','jonkheer','don','capt','dona','sir'], 'notable')

# compute median age, number of observations and Survived % per title
combined_df[['Age', 'title', 'Survived']].groupby(by='title').agg({'Age':['median','count'], 'Survived' : 'mean'})


# In[ ]:


# taking a closer look at our 'Notable' passengers, their median age is only 9 years old ??!
combined_df[combined_df['title'] == 'notable']


# The **Master** title is only held by **young males under 15 years old** in our dataset.
# 
# According to wikipedia:
# "Master – For male children: Young boys were formerly addressed as "Master [first name]." This was the standard form for servants to use in addressing their employer's minor sons. It is also the courtesy title for the eldest son of a Scottish laird."<br\>
# See [https://en.wikipedia.org/wiki/Title]
# 
# Interesting. Let's create a dedicated bucket for our little masters. :)

# In[ ]:


# set title='Master' (instead of 'Notable') for passengers for which we have extracted the title 'Master' previously
combined_df.loc[combined_df['title_extracted'] == 'Master', 'title'] = 'master'

# drop the 'title_extracted' column, we don't need anymore
combined_df.drop('title_extracted', axis=1, inplace=True)

# compute median age, number of observations and Survived % per title
combined_df[['Age', 'title', 'Survived']].groupby(by='title').agg({'Age':['median','count'], 'Survived' : 'mean'})


# Let's compute the **median age** for each **title x class bucket**.

# In[ ]:


combined_df.groupby(by=['Pclass', 'title']).agg({'Age':['median','count'], 'Survived' : 'mean'})


# - **Masters** : their **Age** is not correlated to **Pclass**. It makes sense since this title refers to young children. We will fill in missing ages with the median age = **4 years old**, ignoring **PClass**.
# 
# - **Miss** : there's a correlation with **PClass**. We'll fill in the blanks with their median age per class = **30yo / 20yo / 18yo**
# 
# - **Mrs** : let's use **45yo** for 1st class, **31 yo** otherwise.
# 
# - **Mr** : same as **Miss**. We'll use values **41yo / 30yo / 26yo**.
# 
# - **Notable** : **48yo** for 1st class, **41yo** otherwise.

# In[ ]:


# create a mapping table with 15 rows (3 classes x 5 titles)
map_table_age_df = pd.DataFrame(np.nan, index=range(0,15),columns=['Pclass', 'title', 'age_pred'])

titles = ['master', 'miss', 'mrs', 'mr', 'notable']

# 1st class passengers
map_table_age_df.iloc[0:5, 0] = 1
map_table_age_df.iloc[0:5, 1] = titles
# input our guesses
map_table_age_df.iloc[0:5, 2] = [4, 30, 45, 41, 48]

# 2nd class passengers
map_table_age_df.iloc[5:10, 0] = 2
map_table_age_df.iloc[5:10, 1] = titles
map_table_age_df.iloc[5:10, 2] = [4, 20, 31, 30, 41]

# 3rd class passengers
map_table_age_df.iloc[10:15, 0] = 3
map_table_age_df.iloc[10:15, 1] = titles
map_table_age_df.iloc[10:15, 2] = [4, 18, 31, 26, 41]

map_table_age_df


# In[ ]:


# join mapping table with the dataset, this adds a new 'age_pred' column to the set
combined_df = combined_df.merge(map_table_age_df, on=['Pclass','title'], how='left')

# evaluate mean error for non-null ages
age_not_null_slice = combined_df['Age'].notnull()
mae = mean_absolute_error(combined_df['age_pred'][age_not_null_slice], combined_df['Age'][age_not_null_slice])
print('mean absolute error for non-null ages =', mae)

# what would have been the mean error if we had set missing ages to the overall median value ?
age_pred_median = [combined_df['Age'][age_not_null_slice].median()]*1046
mae_static_median = mean_absolute_error(age_pred_median, combined_df['Age'][age_not_null_slice])
print('mean absolute error for non-null ages with simple overall median =', mae_static_median)


# In[ ]:


# fill in missing ages
combined_df['Age'] = combined_df['Age'].fillna(combined_df['age_pred'])

# drop the 'age_pred' column, we don't need anymore
combined_df.drop('age_pred', axis=1, inplace=True)

combined_df.head()


# 2.1.2 Missing Embarked values
# 
# Two values are missing in the training set.

# In[ ]:


combined_df[combined_df['Embarked'].isnull()]


# Those passengers share the same Ticket # ('113572'), Fare and Cabin. Can we assume they embarked **at the same place** ?

# In[ ]:


# counting the number of distinct values of Embarked per ticket #
combined_df[['Ticket','Embarked']][combined_df['Embarked'].notnull()].groupby(by='Ticket').Embarked.nunique().value_counts()


# The dataset has **only 2 occurences** of tickets **out of 928** shared by multiple passengers which embarked at **different locations**.<br\>
# We can safely assume **our two passengers** sharing ticket # 113572 embarked **at the same port**.<br\>
# But which one ?
# 
# For PassengerId = 62, name is French ('Amélie', 'Icard'). Did she embarked at Cherbourg ?<br\>
# For PassengerId = 829, name is English. Hard to say where those passengers embarked based on their names. Let's find something else.

# Do we have other passengers in the dataset **sharing this ticket #** ?

# In[ ]:


combined_df[combined_df['Ticket'] == '113572']


# Nope. :( <br\>
# There's **no other passenger** in the dataset sharing this ticket #.
# 
# Passengers were in **first class**. Where did first class passengers embark ?

# In[ ]:


# filter 1st class passengers, group them by Embarked values 
combined_df.loc[combined_df['Pclass'] == 1][['Embarked','PassengerId']].groupby(by='Embarked').count()


# Only **3 passengers** (< 1%) of 1st class embarked at 'Q'. **Let's assume our passengers did not embark at 'Q'**.
# 
# Now almost **50%** of 1st class passengers embarked at 'C', **50%** at 'S'. Which one to choose ?
# 
# Maybe the **ticket #** ('113572') can give us a hint.

# In[ ]:


# get number of tickets starting by '113'
print('Number of tickets starting by 113 =', combined_df.loc[combined_df['Ticket'].str.match('^113'), 'Ticket'].count(), '\n')

# get class of tickets starting by '113'
print('Class of tickets starting by 113: \n', combined_df.loc[combined_df['Ticket'].str.match('^113'), ['Pclass', 'PassengerId']].groupby(by='Pclass').count(), '\n')

# get Embarked value of passengers which tiket starts by '113'
print('Embarked values of tickets starting by 113: \n', combined_df.loc[combined_df['Ticket'].str.match('^113') , 'Embarked'].value_counts() / combined_df['Ticket'].str.match('^113').sum(), '\n')


# **66** ticket numbers start by '113'.<br\>
# **All of them** are **first class tickets**.<br\>
# **82%** of passengers embarked at '**S**'.<br\>
# **15%** of passengers embarked at '**C**'.
# 
# We assume passengers embarked at **Southampton**.

# In[ ]:


# replace missing Embarked value with 'S'
combined_df['Embarked'].fillna('S', inplace=True)


# 2.1.3 Missing Fare value
# 
# Only **one passenger** has a missing **Fare** value.

# In[ ]:


combined_df[combined_df['Fare'].isnull()]


# Maybe he was **sharing his ticket**with another passenger ?

# In[ ]:


# get passengers with ticket # = '3701'
combined_df[combined_df['Ticket'] == '3701']


# He's the **only** passenger with this ticket #. :(<br\>
# And we don't have his cabin # either.
# 
# It was a **3rd class** ticket from **Southampton** though.
# Let's fill in the missing value with the **median fare price for 3rd class tickets embarking at Southampton**.

# In[ ]:


# get the median fare for a 3rd class passenger embarking at S (= 8.5)
median_fare_3rd_S = combined_df.loc[(combined_df['Embarked'] == 'S') & (combined_df['Pclass'] == 3) , 'Fare'].median()

# impute fare price
combined_df.loc[combined_df['Fare'].isnull(), 'Fare'] = median_fare_3rd_S


# 2.1.4 Missing Cabin value
# 
# Only 295 non missing values. Meaning the cabin # is **missing 78% of the time**.<br\>
# There are **too many missing values** to try to do some imputation. We'll keep it **as is**. Even though we'll see later on that some new feature could be extrated from the Cabin.

# # Feature Engineering

# Creating new features based on the existing ones. We'll see how they relate to the survival chances.

# ## Sex
# 
# Convert Sex feature to an integer: 'female' -> 0, 'male' -> 1

# In[ ]:


# set Sex to 0 for female and 1 for male
combined_df['Sex'] = combined_df['Sex'].astype("category")
combined_df['Sex'].cat.categories = [0,1]
combined_df['Sex'] = combined_df['Sex'].astype("int")


# ## Age
# 
# To prevent overfitting we would like to create **age groups** ('buckets').<br\>
# 
# We could create a group for "children", because they have a better chancer to survive.
# 
# We have to define at what age we consider a passenger to be a child. The limit should should be **around 14-18 years old**. Let's compute the **survival rate** for the following age groups : [0-14], [0-15], [0-16], [0-17] and [0-18]

# In[ ]:


print('Under 14 y.o. survivded pct:', combined_df[combined_df.Age < 14].Survived.mean())
print('Under 15 y.o. survivded pct:', combined_df[combined_df.Age < 15].Survived.mean())
print('Under 16 y.o. survivded pct:', combined_df[combined_df.Age < 16].Survived.mean())
print('Under 17 y.o. survivded pct:', combined_df[combined_df.Age < 17].Survived.mean())
print('Under 18 y.o. survivded pct:', combined_df[combined_df.Age < 18].Survived.mean())


# We'll take **16 y.o.**as the limit. Then create groups "16 y.o. large".<br\>
# At the same time we **rescale** the data by encoding age groups with values between **0 and 1**.

# In[ ]:


combined_df['age_group'] = np.nan # create new empty column
combined_df.loc[combined_df['Age'] < 16, 'age_group'] = 0
combined_df.loc[(combined_df['Age'] >= 16) & (combined_df['Age'] < 32), 'age_group'] = 0.25
combined_df.loc[(combined_df['Age'] >= 32) & (combined_df['Age'] < 48), 'age_group'] = 0.50
combined_df.loc[(combined_df['Age'] >= 48) & (combined_df['Age'] < 64), 'age_group'] = 0.75
combined_df.loc[combined_df['Age'] >= 64, 'age_group'] = 1


# In[ ]:


combined_df['Pclass'] = combined_df['Pclass'].astype("int")


# ## Cabin
# 
# Cabin # are composed of **a letter** (from 'A' to 'G') + **1 to 3 digits**, e.g. 'E39'<br\>
# Some records only have the letter of the cabin # (e.g. 'D')<br\>
# Some records have **multiple cabin #** in the Cabin feature string, e.g. 'B51 B53 B55', 'E39 E41', 'F E69'
# 
# Google tells us that those letters correspond to **decks**. See https://www.encyclopedia-titanica.org/titanic-deckplans/.
# 
# People from the **upper decks** are probably more **likely to survive** than passengers from the **lower** decks.<br\>
# Let's have a look.

# In[ ]:


# create new feature that contains deck number, by extracting the first character of the Cabin feature
# if the string contains multiple cabin #, we will catch only the first one, assuming the other cabin # of the string are part of the same deck
combined_df['deck'] = combined_df['Cabin'].str.extract('([a-zA-Z])', expand=False)

combined_df[combined_df['deck'].notnull()].head(5)


# In[ ]:


# get survival rate, sex, age and class per deck
deck_notnull_slice = combined_df.loc[combined_df['deck'].notnull(), :]
deck_analysis_df = deck_notnull_slice.groupby(by='deck').agg({'Survived':'mean',                                                               'PassengerId':'count',                                                               'Sex':'mean',                                                               'Age':'mean',                                                                'Pclass':'mean'})

# rename columns
deck_analysis_df.columns = ['pct_surv', 'count', 'pct_male', 'age_mean', 'class_mean']

# moves 'deck' from the index to a column
deck_analysis_df.reset_index(inplace=True)

# compute survival rate, median age, pct male
test_df=combined_df.iloc[0:890, :]
surv_rate_1st = test_df[test_df.Pclass == 1].Survived.mean()
surv_rate_2nd = test_df[test_df.Pclass == 2].Survived.mean()
surv_rate_3rd = test_df[test_df.Pclass == 3].Survived.mean()
o_median_age = combined_df.Age.median()
o_pct_male = combined_df.Sex.mean()
o_class_mean = combined_df.Pclass.mean()

# plot
fig = plt.figure(figsize=(14,14))

plt.subplot(221)
sns.barplot(x='deck', y='pct_surv', data=deck_analysis_df)
surv_rate_1st_line = plt.axhline(y=surv_rate_1st, xmin=0, xmax=1, color='g', label='survival rate 1st class')
surv_rate_2nd_line = plt.axhline(y=surv_rate_2nd, xmin=0, xmax=1, color='b', label='survival rate 2nd class')
surv_rate_3rd_line = plt.axhline(y=surv_rate_3rd, xmin=0, xmax=1, color='r',  label='survival rate 3rd class')
plt.legend(handles=[surv_rate_1st_line, surv_rate_2nd_line, surv_rate_3rd_line])

plt.subplot(222)
sns.barplot(x='deck', y='count', data=deck_analysis_df)

plt.subplot(234)
sns.barplot(x='deck', y='pct_male', data=deck_analysis_df)
pct_male_line = plt.axhline(y=o_pct_male, xmin=0, xmax=1, label='overall pct of males')
plt.legend(handles=[pct_male_line])

plt.subplot(235)
sns.barplot(x='deck', y='age_mean', data=deck_analysis_df)
median_age_line = plt.axhline(y=o_median_age, xmin=0, xmax=1, label='overall median age')
plt.legend(handles=[median_age_line])

plt.subplot(236)
sns.barplot(x='deck', y='class_mean', data=deck_analysis_df)
class_mean_line = plt.axhline(y=o_class_mean, xmin=0, xmax=1, label='overall class mean')
plt.legend(handles=[class_mean_line])

sns.plt.show()

deck_analysis_df


# **What we learn**:<br\>
# - chart #1: the **survival rate** is **significantly higher than the dataset average rate** (40%). This means that passengers are more likely to survive if their Cabin # was populated in the dataset. Especially decks B, D, E (> 70% survival rate).
# - chart #2: very **few observations**. Especially for decks A,F,G,T (< 20). It will be dangerous to conclude anything for those decks.
# - chart #3: **females** are more represented than in the overall dataset. Except for deck A but we have very observations for this deck.
# - chart #4: **age mean** is **higher** than the dataset median. Passengers are older that the dataset median for decks A to E. Decks F and G is under the median age = younger passengers.  
# - chart #5: a large majority of passengers are **in 1st class**. Almost all of them are in 1st class for decks A, B, C, D, E and T. Deck F has almost 50% of passengers in 2nd class / 50% in 3rd class. Deck G has almost 100% of passengers in 3rd class.
# 
# Conclusion: 
# Seems like we can split the decks in **two categories** : **decks A to E** (mostly 1st class, older passengers) and **decks F/G** (2nd and 3rd class, younger passengers).<br\>
# This would explain why decks B,C,D,E seem to have a better survival rate than decks F/G. There's a correlation with Age and Class.<br\>
# With too few observations and a correlation with the class and age, we cannot tell if one deck provides a better chance to survive over another one.<br\>
# The only guess we will make is that - according to chart #1 - **having a deck populated in the dataset (i.e. a Cabin #) improves your chance to have survived**.
# 
# We will use only one **'deck_known' boolean feature** for our model.

# In[ ]:


# create a new feature 'deck_down' which value is True when 'deck' is not null
combined_df['deck_known'] = combined_df['deck'].notnull()


# In[ ]:


# create feature for Age scaled down to [0, 1]
combined_df['age_scaled'] = combined_df.Age - combined_df.Age.min()
combined_df['age_scaled'] = combined_df.age_scaled / combined_df.Age.max()

# create feature for Fare scaled down to [0, 1]
# use logarithm because we have extreme Fare prices
combined_df['fare_scaled'] = np.log10(combined_df['Fare'] + 1)
combined_df['fare_scaled'] = (combined_df.fare_scaled - combined_df.fare_scaled.min()) / combined_df.fare_scaled.max()

# create boolean feature to track passengers whose cabin # is known (those passengers are probably more likely to have survived)
combined_df['cabin_known'] = combined_df.Cabin.notnull()

# create feature for the total number of family members
combined_df['family_size'] = combined_df.SibSp + combined_df.Parch

# create boolean feature for passengers that travelled alone
combined_df['is_alone'] = combined_df['family_size'] == 0

combined_df.head(5)


# ## Title
# 
# Let's tranform titles to integers.

# In[ ]:


title_mapping = {"mr": 0, "notable": 0.2, "master": 0.3, "miss": 0.4, "mrs": 0.5}
combined_df['title_encoded'] = combined_df['title'].map(title_mapping)
combined_df['title_encoded'] = combined_df['title_encoded'].fillna(0)


# In[ ]:


combined_df.head (5)


# # Preparing data for modeling

# Transforming categorial features into integers

# In[ ]:


# create dummy features for the Embarked feature
embarked_dummies_df = pd.get_dummies(combined_df.Embarked, prefix='embarked')
combined_df = pd.concat([combined_df, embarked_dummies_df], axis=1)

# drop the 'Embarked_Q' dummy feature, because if a passenger did not embarked at C or S, he has necessarily embarked at Q.
combined_df.drop('embarked_Q', axis=1, inplace=True)


# **Select features** that we will keep to train our model.<br\>
# This is an **iterative approach**. We will test the model **accuracy** at each iteration, then refine our features list.<br\>
# The details of each iteration is **documented** at the beginning of the notebook.

# In[ ]:


cols = ['Pclass', 'Sex', 'age_scaled', 'fare_scaled', 'title_encoded',          'is_alone', 'embarked_C', 'embarked_S', 'deck_known', 'Survived']

train_df = combined_df.loc[:890, cols]
test_df = combined_df.loc[891:, cols]
test_df.drop('Survived', axis=1, inplace=True)

ax = plt.subplots(figsize =(12, 12))
heat = sns.heatmap(train_df.corr(), vmax=1.0, square=True, annot=True)

sns.plt.show()


# # Modeling

# In[ ]:


y_train = train_df['Survived']
X_train = train_df.drop('Survived', axis=1)
X_test = test_df


# ## K Nearest Neighbours

# In[ ]:


# instantiate KNN classifier
knn = KNeighborsClassifier()

# define the parameters values that should be searched
k_range = range(1, 40)

# create a parameter grid: map the parameter names to the values that should be searched
param_grid = {'n_neighbors' : k_range}

# instantiate the grid
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')


# In[ ]:


# fit the grid
grid.fit(X_train,y_train)


# In[ ]:


# get mean score for each value of params (n=1, n=2, ...)
mean_test_score = grid.cv_results_['mean_test_score']

# plot the results
plt.plot(k_range, mean_test_score)
plt.xlabel('Value of k for KNN')
plt.ylabel('Cross-validated accuracy')
sns.plt.show()

# display best param and corresponding score
result_str = "Best params: {}; score: {}"
print(result_str.format(grid.best_params_, grid.best_score_))


# # Prediction and submission

# In[ ]:


# create an instance of a KNN classifier to make our prediction
# testing with different values of hyper parameter n_neighbors (the [15-30] look like a good range)
knn_sub = KNeighborsClassifier(n_neighbors=25)

# fit our model all the WHOLE training set
knn_sub.fit(X_train, y_train)

# make our prediction
y_pred = knn_sub.predict(X_test)

# insert our prediction in a DataFrame with the PassengerId
submit = pd.DataFrame({'PassengerId' : combined_df.loc[891:,'PassengerId'],
                         'Survived': y_pred.astype(int)})


# In[ ]:


submit.info()


# In[ ]:


submit.head(10)


# In[ ]:


submit.to_csv("prediction-20170627-Q.csv", index=False)

