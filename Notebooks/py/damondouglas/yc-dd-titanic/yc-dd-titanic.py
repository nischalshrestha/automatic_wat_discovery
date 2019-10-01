#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# On April 15, 2912 more than 1,500 souls perished on board the RMS Titanic[1].  This challenge uses machine learning to predict based on passenger data whether an individual would have survived this tragedy.  The titanic dataset comprises information on a passenger's age, boarding class, paid fare, point of embarkment, gender, cabin number, ticket details, and the number of accompanied family members.

# # Imports, Data, Constants and Helpers

# ## Imports

# In[ ]:


# Import libraries
import warnings; warnings.simplefilter('ignore')
import re
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
get_ipython().magic(u'matplotlib inline')


# ## Data

# In[ ]:


# courtesy Names Corpus Version 1.3 by Mark Kantrowitz (c) 1991 (http://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/)
names = pd.read_csv("../input/names/names.csv")
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
train.info()


# ## Globals and Constants

# In[ ]:


"""
Globals
=======
"""
MEDIAN_AGE = train.Age.median()
MEDIAN_FARE = train.Fare.median()
SVM_MODEL = None
CATBOOST_MODEL = None

"""
Columns
=========
"""

# To avoid hardcoding column names:
class columns:
    pass

c = columns()
for col in train.columns.values:
    setattr(c, col, col)

setattr(c, "Female", "Female")
setattr(c, "Title", "Title")
setattr(c, "HasSib", "HasSib")
setattr(c, "HasParch", "HasParch")
setattr(c, "HasFamily", "HasFamily")
setattr(c, "FareCluster", "FareCluster")
setattr(c, "AgeCluster", "AgeCluster")
setattr(c, "CabinLevel", "CabinLevel")
setattr(c, "Sarch", "Sarch")

# Additional Constants
HAS_SIB_THRESHOLD = 0
HAS_PARCH_THRESHOLD = 0
NUM_FARE_CLUSTERS = 3
NUM_AGE_CLUSTERS = 6
MARRIED_TITLES = ["Mr", "Mrs"]
TRUE_LOVE_AGE_CUTOFF = 18
COMMON = "Common"
PRIVILAGED = "Privilaged"
MILITARY = "Military"
PUBLIC_SERVANT = "PublicServant"


# ## Helpers

# In[ ]:


# Pipeline manager
class Pipeline:
    _steps = {}
    
    def step(func):
        df = pd.DataFrame()
        df = func(df)
        try:
            df.info()
        except:
            raise "function does not return pandas.DataFrame"

        Pipeline._steps[func.__name__] = func
        return func


    def process(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for f in Pipeline._steps.values():
            df_copy = f(df_copy)

        return df_copy


# In[ ]:


# Drafts a contingency table between column1 and column2 with pvalues.
# Assums column1 and column2 are categorical variables.
def contingency(df: pd.DataFrame, column1: str, column2: str) -> pd.DataFrame:
    row = sorted(pd.unique(df[column1]))
    col = sorted(pd.unique(df[column2]))
    d = {}
    columns = [(column2, k) for k in col]
    index = [(column1, m) for m in row]
    p = []
    total = []

    for k in col:
        if k not in d:
            d[k] = []
        for m in row:
            d[k].append(
                len(
                    df.query("{} == {} and {} == {}".format(column2, k, column1, m))
                )
            )

    for m in row:
        slice = df.query("{} == {}".format(column1, m))
        (_, pval) = stats.chisquare(slice[column2].value_counts())
        p.append(pval)

        total.append(len(slice))

    cs = pd.DataFrame(d)
    cs.columns = pd.MultiIndex.from_tuples(columns)
    cs.index = pd.MultiIndex.from_tuples(index)
    cs["Total"] = total
    cs["Pvalue"] = p
    
    return cs


# In[ ]:


# Plotting Helpers
def pretty_hist(df: pd.DataFrame, yaxis_title: str, column_name: str, title: str, labels: list):
    values = df.groupby([column_name])[column_name].count().values
    uniq_values = pd.unique(df[column_name].values)
    ticks = np.arange(np.min(uniq_values) - 0.5, np.max(uniq_values)+1, step=1)
    n = len(df)
    percentages = [int(k*100/n) for k in values]
    
    if len(percentages) != len(labels):
        print(len(percentages), len(labels))
    assert len(percentages) == len(labels)
        
    for i in range(len(labels)):
        labels[i] = labels[i] + " ({}%)".format(percentages[i])

    labels.insert(0, "")
    labels.append("")
    plt.hist(df[column_name], np.arange(len(uniq_values)*2), rwidth=0.3)
    plt.title(title)
    plt.xticks(ticks, labels)
    plt.ylabel(yaxis_title)
    


# # Exploratory Data Analysis

# # Univariate

# ## PassengerId
# `PassengerId` is unique to each record and does not provide any meaningful information for the model.  This variable will be dropped.

# In[ ]:


assert len(pd.unique(train.PassengerId.values)) == len(train)


# ## Survived
# `Survived` is the outcome variable to be modeled.  Note that the percent survived (~38%) in the training dataset matches what occured historically, confirming at least with this factor that we've likely sampled appropriately from the population.

# In[ ]:


pretty_hist(train, "Count", c.Survived, "Survived | Titanic train dataset", ["Not\nSurvived", "Survived"])


# ## Pclass
# `Pclass` represents the passenger boarding class.  As expected, most passengers were residing in 3rd class.

# In[ ]:


pretty_hist(train, "Count", c.Pclass, "Pclass | Titanic train dataset", ["1st", "2nd", "3rd"])


# ## Name
# `Name` in this training dataset typically follows the format, `<Last Name>, <Title> <First Name>`.  We can use this property to extract the title.

# In[ ]:


train[[c.Name]].head(10)


# ## Sex
# There were more males aboard the RMS Titanic than females.

# In[ ]:


pretty_hist(train.assign(Sex = [1 if k == 'female' else 0 for k in train.Sex]), "Count", c.Sex, "Sex | Titanic train dataset.", ["Male", "Female"])


# ## Age
# `Age` has significant missing values (~19%) and appears to be multimodel as shown with the kde plot. 

# In[ ]:


pretty_hist(train.assign(Missing = [1 if k else 0 for k in train.Age.isnull()]), "Count", "Missing", "Age | Titanic train dataset.", ["Not\nMissing Value", "Missing Value"])


# In[ ]:


ax = train.Age.plot.kde()
ax.set_title("Age | Titanic train dataset.")
ax.set_ylabel("Density")
ax.set_xbound(0)
_ = ax.set_xlabel("Age (year)")


# ## SibSp
# Most passengers had at most 1 sibling on board as shown in the `SibSp` variable.

# In[ ]:


p = plt.hist(train.SibSp)
plt.title("SibSb | Titanic train dataset.")
plt.ylabel("Count")
_ = plt.xlabel("Number of siblings on board.")


# ## Parch
# `Parch` shows that most passengers boarded alone. 

# In[ ]:


p = plt.hist(train.Parch)
plt.title("Parch | Titanic train dataset.")
plt.ylabel("Count")
_ = plt.xlabel("Number of parent/children accompanied on board.")


# ## Ticket
# `Ticket` most likely correlates with `Pclass` and will therefore be dropped.

# ## Fare
# There are no missing `Fare` values.  Most passengers paid less than $100.

# In[ ]:


assert train.Fare.isnull().sum() == 0 # True
ax = train.Fare.plot.kde()
ax.set_title("Fare | Titanic train dataset.")
ax.set_ylabel("Density")
ax.set_xbound(0)
_ = ax.set_xlabel("Fare ($)")


# ## Cabin
# 
# **We decided to drop this field as it may relate more to `Pclass` which is more of an association with survival.**
# 
# Most `Cabin` assignments start with a number and others a letter.  The letter may correspond to the ship deck [see Titanic cutaway diagram](https://en.wikipedia.org/wiki/First_class_facilities_of_the_RMS_Titanic#/media/File:Titanic_cutaway_diagram.png)[2]. According to the diagram, life boats were near the A deck (top most floor).  However, according to the testimonies of survivors [3], once the ship made contact with the ice berg, there was a significant amount of time to reach the lifeboats.  The issue was not so much the distance to the lifeboats but that there was a lack thereof.  On a large scale, there was no fighting for positions on a lifeboat, according to the testimonies.  Finally, when the ship broke in two pieces, the life boats had already departed.

# In[ ]:


train.assign(CabinLetter = train.Cabin.str.get(0)).CabinLetter.value_counts()


# ## Embarked
# `Embarked` indicates passenger's port of departure where S, C, and Q correspond to **C**herbourg, **S**outhampton, and **Q**ueenstown, respectively.  The order of passenger embarkment was first Southampton, followed by Cherbourg and finally Queenstown before setting out into the Atlantic Ocean [1].  There were only two missing embarkment records of two ladies in first class traveling in the same Cabin who survived.
# On face value, one may consider `Embarked` to lack predictive value.  However, where a passenger boarded may dictate ship cabin filling order which may have influenced access to lifeboats.

# In[ ]:


train.Embarked.value_counts()


# In[ ]:


train[train.Embarked.isnull()]


# # Bivariate Analysis

# ## Embarked versus Survived

# Most people embraced at S, but it has a high rate of not survived
# - source: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.boxplot.html
# - source - http://pandas.pydata.org/pandas-docs/version/0.9.1/visualization.html
# - Source: https://stackoverflow.com/questions/48799718/pandas-pivot-table-to-stacked-bar-chart
# 

# In[ ]:


train['Embarked'].value_counts()
train['Embarked'].value_counts().plot.bar();

df_Survived = train[train['Survived']==1]
df_Not_Survived = train[train['Survived']==0]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
df_Survived['Embarked'].value_counts().plot.box(ax=axes[0]); 
axes[0].set_title('Embarked Survived')
df_Not_Survived['Embarked'].value_counts().plot.box(ax=axes[1]); 
axes[1].set_title('Embarked not Survived')

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 5))
df_Survived['Embarked'].value_counts().plot.bar(ax=axes[0]); 
axes[0].set_title('the total count of Embarked for survived ')
df_Not_Survived['Embarked'].value_counts().plot.bar(ax=axes[1]); 
axes[1].set_title('the total count of Embarked for non-survived ')

train.groupby('Survived')['Embarked'].value_counts().unstack(level=1).plot.bar(stacked=True)


# ## Fare versus Survived
# `Fare` there is a significant difference in that paid by passengers who survived and who didn't survive.  Therefore, it will be valuable to keep this variable in the model.

# In[ ]:


stats.ttest_ind(train[train.Survived == 0].Fare, train[train.Survived == 1].Fare)


# According to the graph, we can know that the fare is critical to the survival.
# - source: https://stackoverflow.com/questions/14885895/color-by-column-values-in-matplotlib
# - source: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.boxplot.html
# - source - http://pandas.pydata.org/pandas-docs/version/0.9.1/visualization.html
# 

# In[ ]:



fg = sns.FacetGrid(data=train, hue='Survived')
fg.map(plt.scatter, 'Age', 'Fare').add_legend()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
df_Survived['Fare'].value_counts().plot.box(ax=axes[0], sharey=True); 
axes[0].set_title('Fare for df_Survived')
df_Not_Survived['Fare'].value_counts().plot.box(ax=axes[1]); 
axes[1].set_title('Fare for df_Not_Survived')
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharey=True)
df_Survived['Fare'].value_counts().plot.hist(ax=axes[0]); 
axes[0].set_title('Fare for df_Survived')
df_Not_Survived['Fare'].value_counts().plot.hist(ax=axes[1]); 
axes[1].set_title('Fare for df_Not_Survived')
plt.show()


# ## Sex

# The total of the male is slightly more than the female but most females survived 
# - Source: https://stackoverflow.com/questions/50319614/count-plot-with-stacked-bars-per-hue
# 

# In[ ]:


print(train['Sex'].value_counts())
train['Sex'].value_counts().plot.bar()

train.groupby('Survived')['Sex'].value_counts().unstack(level=1).plot.bar(stacked=True)


# ## Pclass
# There was a statistically significant survival outcome between boarding classes 1 and 3 only.  2nd class passengers almost equally survived, while 1st and 3rd class passengers found opposite fates.  Roughly 25% 3rd class passengers survived.  Over 50% 1st class passengers enjoyed safety after this tragedy.  Therefore, `Pclass` is worthy to include in the model.

# In[ ]:


contingency(train, c.Pclass, c.Survived)


# In[ ]:


fig, _ = mosaic(train, [c.Pclass, c.Survived], title="Pclass vs Survived | Titanic train dataset.", axes_label=True)
fig.axes[0].set_ylabel(c.Survived)
_ = fig.axes[0].set_xlabel(c.Pclass)


# According to the graph, most people who died were from class 3.****
# - source: https://stackoverflow.com/questions/50319614/count-plot-with-stacked-bars-per-hue

# In[ ]:


train.groupby('Survived')['Pclass'].value_counts().unstack(level=1).plot.bar(stacked=True)
train['Pclass'].value_counts()


# ## Pclass versus Fare

# We want to find out the relationship between class and fare. The people who were class 1 paid more than the other two. This result helps me to identify the relationship between embedded and class.
# - source: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.boxplot.html
# - source:  https://stackoverflow.com/questions/50319614/count-plot-with-stacked-bars-per-hue

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 4), sharey=True)
train[train['Pclass']==1]['Fare'].plot.box(ax=axes[0]);
axes[0].set_title('Class 1')

train[train['Pclass']==2]['Fare'].plot.box(ax=axes[1]); 
axes[1].set_title('Class 1')

train[train['Pclass']==3]['Fare'].plot.box(ax=axes[2]); 
axes[2].set_title('Class 3')
train.groupby('Pclass')['Embarked'].value_counts().unstack(level=1).plot.bar(stacked=True)


# # Preprocessing
# 
# We preprocess variables by one-hot recoding if categorical or kmeans clustering followed by one-hot recoding before providing them to the model.  Using a decorator pattern, flag a method to the pipeline processor to ensure consistency when applying methodology to both train and test datasets.

# ## Sex

# ### Definition
# We simply create a new variable `Female`, assigning `1` where indicated and `0` otherwise.

# In[ ]:


@Pipeline.step
def numerify_sex(df: pd.DataFrame) -> pd.DataFrame:
    if c.Sex not in df.columns.values:
        return df
    
    dfcopy = df.assign(Sex=df.Sex.str.title())
    return dfcopy.join(pd.get_dummies(dfcopy.Sex)).drop(["Male"], axis=1)


# ### Testing
# `Female` seems associated with `Survived`.

# In[ ]:


# Test numerify_sex
train_sex_numerified = numerify_sex(train)
assert train_sex_numerified.query("Sex == 'Male'").query("Female == 0").Female.count() == 577
assert train_sex_numerified.query("Sex == 'Female'").query("Female == 1").Female.count() == 314

contingency(train_sex_numerified, c.Female, c.Survived)


# ## Name

# ### Definition
# The `Name` variable was recoded based on the extracted title assuming a string pattern `<Last Name>, <Title> <First Name>(<Additional Name>...)`.  We employ the Names Corpus to check against a potential error that `Title` may actually map to a `First Name`.  The train dataset is consistent with the string pattern.  We cannot assume this in future test datasets.
# 
# - Names Corpus Version 1.3 Courtesy: Mark Kantrowitz (c) 1991 (http://www.cs.cmu.edu/afs/cs/project/ai-repository/ai/areas/nlp/corpora/names/)

# In[ ]:


def extract_title(name: str) -> str:
    return re.search(r'(?<=\,\s)\w+', name).group(0)

def remap_title(name: str) -> str:
    result = name
    remap = {
        "Mr": COMMON,
        "Mrs": COMMON,
        "Mlle": PRIVILAGED,
        "Mme": PRIVILAGED,
        "Ms": PRIVILAGED,
        "the": PRIVILAGED,
        "Miss": PRIVILAGED,
        "Master": PRIVILAGED,
        "Rev": PUBLIC_SERVANT,
        "Dr": PUBLIC_SERVANT,
        "Don": COMMON,
        "Madame": COMMON,
        "Major": MILITARY,
        "Lady": PRIVILAGED,
        "Sir": PRIVILAGED,
        "Col": MILITARY,
        "Capt": MILITARY,
        "Royalty": PRIVILAGED,
        "Dona": PRIVILAGED,
        "Jonkheer": COMMON
    }
    if is_name(name):
        result = "NONE"

    if name in remap:
        result = remap[name]
    else:
        raise Exception("{} not assignable to title remapping.".format(name))

    return result

def is_name(name: str) -> bool:
    return names.Name.equals(name)

@Pipeline.step
def numerify_title(df: pd.DataFrame) -> pd.DataFrame:
    if c.Name not in df.columns.values:
        return df

    dfcopy = df.copy()
    if dfcopy.Name.isnull().sum() > 0:
        raise Exception("numerify_title is not implemented to handle missing data")

    dfcopy = dfcopy.assign(Title=[remap_title(extract_title(s)) for s in dfcopy.Name])

    dfcopy = dfcopy.join(pd.get_dummies(dfcopy.Title))

    return dfcopy


# ### Testing
# Being `Female` and `Privilaged` ensured greater chances of survival.

# In[ ]:


train_title = numerify_title(train)
f = train_title[train_title.Sex == 'female']
m = train_title[train_title.Sex == 'male']

titleS = pd.unique(train_title.Title.values)
ms = []
fs = []
tot = []

for k in titleS:
    ms.append(m[m.Title == k].Survived.sum())
    fs.append(f[f.Title == k].Survived.sum())
    tot.append(len(train_title[train_title.Title == k]))


df = pd.DataFrame()
df = df.assign(Title = titleS, MaleSurvived = ms, FemaleSurvived = fs, Total = tot)
df = df[["Title", "MaleSurvived", "FemaleSurvived", "Total"]]
df


# ## Pclass

# ### Definition
# We simply hot-recoded `Pclass` into `Pclass_<n>...`

# In[ ]:


@Pipeline.step
def recode_pclass(df: pd.DataFrame) -> pd.DataFrame:
    if c.Pclass not in df.columns.values:
        return df

    dfcopy = df.copy()
    assert dfcopy.Pclass.isnull().sum() == 0

    return dfcopy.join(pd.get_dummies(dfcopy.Pclass, prefix=c.Pclass))


# ### Testing

# In[ ]:


# Test recode_pclass
train_pclass = recode_pclass(train)
assert train_pclass.Pclass_1.isnull().sum() == 0
assert train_pclass.Pclass_2.isnull().sum() == 0
assert train_pclass.Pclass_3.isnull().sum() == 0
assert len(train_pclass.query("Pclass == 1 and Pclass_1 == 1 and Pclass_2 == 0 and Pclass_3 == 0")) == len(train.query("Pclass == 1"))
assert len(train_pclass.query("Pclass == 2 and Pclass_1 == 0 and Pclass_2 == 1 and Pclass_3 == 0")) == len(train.query("Pclass == 2"))
assert len(train_pclass.query("Pclass == 3 and Pclass_1 == 0 and Pclass_2 == 0 and Pclass_3 == 1")) == len(train.query("Pclass == 3"))
train_pclass[[c.Pclass, "Pclass_1", "Pclass_2", "Pclass_3"]].head()


# ## SibSp and Parch

# ### Definition
# We decided to recode `SibSp` and `Parch` into one variable `HasFamily`.  The `SibSp` variable comingles flagging of whether a passenger co-boarded with a spouse or sibling which we attempted to address (see `Sarch` definition).

# In[ ]:


def recode_sibsp(df: pd.DataFrame) -> pd.DataFrame:
    if c.SibSp not in df.columns.values:
        return df

    dfcopy = df.copy()
    assert dfcopy.SibSp.isnull().sum() == 0

    return dfcopy.assign(HasSib = [k > HAS_SIB_THRESHOLD for k in dfcopy.SibSp])

def recode_parch(df: pd.DataFrame) -> pd.DataFrame:
    if c.Parch not in df.columns.values:
        return df

    dfcopy = df.copy()
    assert dfcopy.Parch.isnull().sum() == 0

    return dfcopy.assign(HasParch = [k > HAS_PARCH_THRESHOLD for k in dfcopy.Parch])

@Pipeline.step
def recode_has_family(df: pd.DataFrame) -> pd.DataFrame:
    if c.SibSp not in df or c.Parch not in df:
        return df

    dfcopy = df.copy()
    df_sibsp = recode_sibsp(dfcopy)
    df_parch = recode_parch(dfcopy)
    df_family = df_sibsp.merge(df_parch)
    has_family = [m or n for (m,n) in zip(df_sibsp.HasSib, df_parch.HasParch)]
    return df_family.assign(HasFamily = [1 if k else 0 for k in has_family])


# ### Testing
# Not `HasFamily` seems more associated with not `Survived`.  `HasFamily` seems equivalent in terms of `Survived` (p-value ~ 0.83).

# In[ ]:


train_family = recode_has_family(train)
qry = "(SibSp > {} or Parch > {}) and HasFamily == 1".format(HAS_SIB_THRESHOLD, HAS_PARCH_THRESHOLD)
assert len(train_family.query(qry)) == len(train_family.query("HasFamily == 1"))
contingency(train_family, c.HasFamily, c.Survived)


# ## Fare

# ### Definition
# Using the KMeans algorithm we converted the `Fare` variable into hot-recoded categorical clusters instead of inputing the continuous variable into the model.

# In[ ]:


@Pipeline.step
def recode_fare(df: pd.DataFrame) -> pd.DataFrame:
    if c.Fare not in df.columns.values:
        return df

    dfcopy = df.copy()
    
    median_fare = 0
    
    if not MEDIAN_FARE:
        median_fare = dfcopy.Fare.median()
    
    dfcopy.Fare.fillna(median_fare, inplace=True)
    
    X = dfcopy.Fare.values.reshape(-1,1)
    km = KMeans(n_clusters=NUM_FARE_CLUSTERS, random_state=0)
    results = km.fit(X)
    dfcopy = dfcopy.assign(FareCluster = results.predict(X))
    
    return dfcopy.join(pd.get_dummies(dfcopy.FareCluster, prefix=c.FareCluster))


# ### Testing
# `Fare` clustered into three groups that didn't overlap.  There appears an association with `FareCluster` assignment and `Survived`.

# In[ ]:


train_fare = recode_fare(train)
centers = pd.unique(train_fare.FareCluster.values)
n_centers = len(centers)
fig, axes = plt.subplots(len(centers), sharex=True, sharey=True)
for i, k in zip(range(n_centers+1), centers):
    ax = axes[i]
    train_fare[train_fare.FareCluster == i].Fare.plot.kde(ax=ax)


# In[ ]:


minS = []
maxS = []
clusterS = range(train_fare.FareCluster.min(),train_fare.FareCluster.max()+1)
for k in clusterS:
    g = train_fare[train_fare.FareCluster == k]
    minS.append(g.Fare.min())
    maxS.append(g.Fare.max())
    
df = pd.DataFrame()
df.assign(FareCluster = clusterS, MinFare=minS, MaxFare=maxS).sort_values(by="MinFare")


# In[ ]:


contingency(train_fare, c.FareCluster, c.Survived)


# ## Age

# ### Definition
# Using the KMeans algorithm we converted the `Age` variable into hot-recoded categorical clusters instead of inputing the continuous variable into the model.  Missing values are imputed using the median.  We assign a global variable `MEDIAN_AGE` from the `train` dataset.  We will impute missing values in the `test` dataset using the median from the `train` dataset.

# In[ ]:


@Pipeline.step
def recode_age(df: pd.DataFrame) -> pd.DataFrame:
    if c.Age not in df.columns.values:
        return df

    dfcopy = df.copy()
    
    median_age = 0
    
    if not MEDIAN_AGE:
        median_age = dfcopy.Age.median()
    
    dfcopy.Age.fillna(median_age, inplace=True)
    X = dfcopy.Age.values.reshape(-1,1)
    km = KMeans(n_clusters=NUM_AGE_CLUSTERS, random_state=0)
    results = km.fit(X)
    dfcopy = dfcopy.assign(AgeCluster = results.predict(X))
    return dfcopy.join(pd.get_dummies(dfcopy.AgeCluster, prefix=c.AgeCluster))


# ### Testing
# The KMeans algorithm custered the `Age` variable into five groups that do not overlap.  `AgeCluster` assignment seems significantly associated with `Survived`.

# In[ ]:


train_age = recode_age(train)
assert train_age.AgeCluster.isnull().sum() == 0
centers = sorted(pd.unique(train_age.AgeCluster.values))
n_centers = len(centers)
fig, axes = plt.subplots(len(centers)-1, sharex=True, sharey=True)
for i, k in zip(range(0, n_centers+1), centers[1:]):
    ax = axes[i]
    train_age[train_age.AgeCluster == i].Age.plot.kde(ax=ax)


# In[ ]:


minS = []
maxS = []
clusterS = range(train_age.AgeCluster.min(),train_age.AgeCluster.max()+1)
for k in clusterS:
    g = train_age[train_age.AgeCluster == k]
    minS.append(g.Age.min())
    maxS.append(g.Age.max())
    
df = pd.DataFrame()
df.assign(AgeCluster = clusterS, MinAge=minS, MaxAge=maxS).sort_values(by="MinAge")


# In[ ]:


contingency(train_age, c.AgeCluster, c.Survived)


# ## Embarked

# ### Definition
# We simply hot recode `Embarked` into `Embarked_<n>...`.  

# In[ ]:


@Pipeline.step
def recode_embarked(df: pd.DataFrame) -> pd.DataFrame:
    if c.Embarked not in df.columns.values:
        return df
    
    dfcopy = df.copy()
    
    dfcopy.Embarked.fillna(dfcopy.Embarked.mode()[0], inplace=True)
    return dfcopy.join(pd.get_dummies(dfcopy.Embarked, prefix=c.Embarked))


# ### Testing

# In[ ]:


train_embarked = recode_embarked(train)
train_embarked[[c.Embarked, "Embarked_C", "Embarked_Q", "Embarked_S"]].head()


# # Sarch

# ### Definition
# Survivor testimonies state that wives coboarded with their husbands remained to perish with them [3].  Therefore, we create a new variable `Sarch` indicating whether or not an individual's spouse is on board with the following criteria:
# * `SibSp` == 1
# * `Parch` == 0
# * Age > 18 (we defined as `TRUE_LOVE_AGE_CUTOFF`)
# * `Title` remapped from `Name` using `remap_title(Name)` is `Mr` or `Mrs`

# In[ ]:


def code_sarch(df: pd.DataFrame) -> pd.DataFrame:
    if c.SibSp not in df.columns.values:
        return df
    
    if c.Parch not in df.columns.values:
        return df
    
    if c.Age not in df.columns.values:
        return df
    
    dfcopy = df.copy()
    sibsp1 = dfcopy.SibSp == 1
    parch0 = dfcopy.Parch == 0
    titleS = [extract_title(k) for k in dfcopy.Name]
    title_mr_or_mrs = [k in MARRIED_TITLES for k in titleS]
    
    median_age = 0
    
    if not MEDIAN_AGE:
        median_age = dfcopy.Age.median()
    
    dfcopy.Age.fillna(median_age, inplace=True)
    
        
    age_gt_cutoff = [k > TRUE_LOVE_AGE_CUTOFF for k in dfcopy.Age]
    return dfcopy.assign(Sarch = 1 * (sibsp1 & parch0 & title_mr_or_mrs & age_gt_cutoff))


# ### Testing
# While there is a significant association with not `Sarch` and not `Survived`, having a spouse co-passenger does not seem significantly associated with `Survived` (p-value ~ 0.57).  Also when applied to the model it reduced the score.
# **Unfortunately this new variable did not improve the model and was not added to the preprocessing pipeline.  We kept the variable definition for interest to the reader.**

# In[ ]:


train_sarch = code_sarch(train)
contingency(train_sarch, c.Sarch, c.Survived)


# # Model Build

# ## Preprocess Data

# In[ ]:


train_recoded = Pipeline.process(train)
columns_to_drop = [c.PassengerId, c.Pclass, c.Name, c.Sex, c.Age, c.SibSp, c.Parch, c.Ticket, c.Fare, c.Cabin, c.Embarked, c.HasSib, c.HasParch, c.Title, c.AgeCluster, c.FareCluster]
train_recoded.drop(columns_to_drop, axis=1, inplace=True)
train_recoded.info()


# ## Split training dataset into its own train and test subsets.

# In[ ]:


X = train_recoded.drop([c.Survived], axis=1)
y = train_recoded.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ## Support Vector Machine

# In[ ]:


def getSVM(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> GridSearchCV:
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svc = svm.SVC(gamma="scale")
    clf = GridSearchCV(svc, parameters, cv=5)
    clf.fit(X_train, y_train)
    print("score %s" % (clf.score(X_test, y_test)))
    y_pred = clf.predict(X_test)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
    return clf

SVM_MODEL = getSVM(X_train, X_test, y_train, y_test)


# ## KNeighborsClassifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
def getKN():
    scores = []
    bestScore = 0
    bestModel = None
    for n in range(1, 10):
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(X_train, y_train)
        s = clf.score(X_test, y_test)
        if bestScore < s:
            bestScore = s
            bestModel = clf
        print("%s score %s" % (n, s))
        scores.append(s)
    plt.plot(range(1, 10), scores, 'ro')
    plt.show()
    y_pred = bestModel.predict(X_test)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
    return bestModel
getKN()


# ## RandomForestClassifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
def getRFT():
    gnb = RandomForestClassifier(n_estimators=100, random_state=20)
    gnb.fit(X_train, y_train)
    print("score %s" % (gnb.score(X_test, y_test)))
    y_pred = gnb.predict(X_test)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
    return gnb
getRFT()


# ## Model Optimization

# GridSearchCV for Random Forest Classifier
# - source: https://stackoverflow.com/questions/30102973/how-to-get-best-estimator-on-gridsearchcv-random-forest-classifier-scikit
# - source: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
# 

# In[ ]:



def getRFTGridSearch():
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True, random_state=20) 

    param_grid = { 
        'n_estimators': [200, 700],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    CV_rfc.fit(X_train, y_train)
    print("score %s" % (CV_rfc.score(X_test, y_test)))
    forest = RandomForestClassifier(n_estimators = CV_rfc.best_params_['n_estimators'], max_features= CV_rfc.best_params_['max_features'], random_state=20)
    forest.fit(X_train, y_train)
    
    # get feature importance
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    indcesOfFeatureNames = [ X_train.columns[f] for f in indices ]
    for f in range(X.shape[1]):
        print("%d. feature %s (index %d )(%f)" % (f + 1, X_train.columns[f], indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indcesOfFeatureNames, rotation=70)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    
    return forest
getRFTGridSearch()


# ## CatBoost
# > Training and applying models for the classification problems. When using the applying methods only the probability that the object belongs to the class is returned. Provides compatibility with the scikit-learn tools.
# 
# - source: https://tech.yandex.com/catboost/doc/dg/concepts/python-usages-examples-docpage/

# In[ ]:


def getCatBoost():
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(iterations=10, learning_rate=1, depth=4, loss_function='Logloss', random_state=20)
    model.fit(X_train, y_train)
    print("score %s" % (model.score(X_test, y_test)))
    y_pred = model.predict(X_test)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
    return model
CATBOOST_MODEL = getCatBoost()


# # Apply Preprocessing and Model to Test Dataset.

# In[ ]:


test_recoded = Pipeline.process(test)
columns_to_drop = [c.PassengerId, c.Pclass, c.Name, c.Sex, c.Age, c.SibSp, c.Parch, c.Ticket, c.Fare, c.Cabin, c.Embarked, c.HasSib, c.HasParch, c.Title, c.AgeCluster, c.FareCluster]
test_recoded.drop(columns_to_drop, axis=1, inplace=True)
test_recoded.info()


# In[ ]:


y_pred = SVM_MODEL.predict(test_recoded)
submission = pd.DataFrame()
submission = submission.assign(PassengerId = test.PassengerId, Survived = y_pred)
submission.to_csv("titanic_survival_prediction.csv", index=False)
submission.head()


# In[ ]:


y_pred = CATBOOST_MODEL.predict(test_recoded)
submission = pd.DataFrame()
submission = submission.assign(PassengerId = test.PassengerId, Survived = y_pred)
submission.to_csv("titanic_survival_prediction_CATBOOST_MODEL.csv", index=False)
submission.head()


# # Summary and Conclusion
# 
# - We used SVM, KNN, and Random Forest in the beginning. RandomForestClassifier has a better accuracy score.  Therefore, we applied `GridSearchCV` for Random Forest in order to find a best hyper parameters. 
# - We also tried `CatBoostClassifier`, and we get slightly better performance
# - Not surprised, the feature of `Female` is most important. 
# 

# # References
# 1. RMS Titanic. (n.d.). In Wikipedia. Retrieved Oct 31, 2018, from [https://en.wikipedia.org/wiki/RMS_Titanic](https://en.wikipedia.org/wiki/RMS_Titanic).
# 2. First class facilities of the RMS Titanic (n.d.). In Wikipedia. Retrieved Oct 31, 2018, from [https://en.wikipedia.org/wiki/First_class_facilities_of_the_RMS_Titanic](https://en.wikipedia.org/wiki/First_class_facilities_of_the_RMS_Titanic).
# 3. Titanic Archive - 1957 Interviews (n.d.). From YouTube. Retrieved Oct 31, 2018, from [https://www.youtube.com/watch?v=FVLiZo6Pkak](https://www.youtube.com/watch?v=FVLiZo6Pkak&t=250s).

# In[ ]:




