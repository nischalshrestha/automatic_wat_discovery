#!/usr/bin/env python
# coding: utf-8

# # Titanic Top 4% with ensemble modeling
# ### **Yassine Ghouzam, PhD**
# #### 13/07/2017
# 
# * **1 Introduction**
# * **2 Load and check data**
#     * 2.1 load data
#     * 2.2 Outlier detection
#     * 2.3 joining train and test set
#     * 2.4 check for null and missing values
# * **3 Feature analysis**
#     * 3.1 Numerical values
#     * 3.2 Categorical values
# * **4 Filling missing Values**
#     * 4.1 Age
# * **5 Feature engineering**
#     * 5.1 Name/Title
#     * 5.2 Family Size
#     * 5.3 Cabin
#     * 5.4 Ticket
# * **6 Modeling**
#     * 6.1 Simple modeling
#         * 6.1.1 Cross validate models
#         * 6.1.2 Hyperparamater tunning for best models
#         * 6.1.3 Plot learning curves
#         * 6.1.4 Feature importance of the tree based classifiers
#     * 6.2 Ensemble modeling
#         * 6.2.1 Combining models
#     * 6.3 Prediction
#         * 6.3.1 Predict and Submit results
#     

# ## 1. Introduction
# 
# This is my first kernel at Kaggle. I choosed the Titanic competition which is a good way to introduce feature engineering and ensemble modeling. Firstly, I will display some feature analyses then ill focus on the feature engineering. Last part concerns modeling and predicting the survival on the Titanic using an voting procedure. 
# 
# This script follows three main parts:
# 
# * **Feature analysis**
# * **Feature engineering**
# * **Modeling**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

from collections import Counter

from lime import lime_tabular, explanation
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, cross_validate

sns.set(style='white', context='notebook', palette='deep')


# ## 2. Load and check data
# ### 2.1 Load data

# In[ ]:


# Load data
##### Load train and Test set

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
IDtest = test["PassengerId"]


# ### 2.2 Outlier detection

# In[ ]:


# Outlier detection 

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])


# Since outliers can have a dramatic effect on the prediction (espacially for regression problems), i choosed to manage them. 
# 
# I used the Tukey method (Tukey JW., 1977) to detect ouliers which defines an interquartile range comprised between the 1st and 3rd quartile of the distribution values (IQR). An outlier is a row that have a feature value outside the (IQR +- an outlier step).
# 
# 
# I decided to detect outliers from the numerical values features (Age, SibSp, Sarch and Fare). Then, i considered outliers as rows that have at least two outlied numerical values.

# In[ ]:


train.loc[Outliers_to_drop] # Show the outliers rows


# We detect 10 outliers. The 28, 89 and 342 passenger have an high Ticket Fare 
# 
# The 7 others have very high values of SibSP.

# In[ ]:


# Drop outliers
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


# ### 2.3 joining train and test set

# In[ ]:


## Join train and test datasets in order to obtain the same number of features during categorical conversion
train_len = len(train)
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)


# I join train and test datasets to obtain the same number of features during categorical conversion (See feature engineering).

# ### 2.4 check for null and missing values

# In[ ]:


# Fill empty and NaNs values with NaN
dataset = dataset.fillna(np.nan)

# Check for Null values
dataset.isnull().sum()


# Age and Cabin features have an important part of missing values.
# 
# **Survived missing values correspond to the join testing dataset (Survived column doesn't exist in test set and has been replace by NaN values when concatenating the train and test set)**

# In[ ]:


# Infos
train.info()
train.isnull().sum()


# In[ ]:


train.head()


# In[ ]:


train.dtypes


# In[ ]:


### Summarize data
# Summarie and statistics
train.describe()


# ## 3. Feature analysis
# ### 3.1 Numerical values

# In[ ]:


# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# Only Fare feature seems to have a significative correlation with the survival probability.
# 
# It doesn't mean that the other features are not usefull. Subpopulations in these features can be correlated with the survival. To determine this, we need to explore in detail these features

# #### SibSP

# In[ ]:


# Explore SibSp feature vs Survived
g = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# It seems that passengers having a lot of siblings/spouses have less chance to survive
# 
# Single passengers (0 SibSP) or with two other persons (SibSP 1 or 2) have more chance to survive
# 
# This observation is quite interesting, we can consider a new feature describing these categories (See feature engineering)

# #### Parch

# In[ ]:


# Explore Parch feature vs Survived
g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# Small families have more chance to survive, more than single (Parch 0), medium (Parch 3,4) and large families (Parch 5,6 ).
# 
# Be carefull there is an important standard deviation in the survival of passengers with 3 parents/children 

# #### Age

# In[ ]:


# Explore Age vs Survived
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")



# Age distribution seems to be a tailed distribution, maybe a gaussian distribution.
# 
# We notice that age distributions are not the same in the survived and not survived subpopulations. Indeed, there is a peak corresponding to young passengers, that have survived. We also see that passengers between 60-80 have less survived. 
# 
# So, even if "Age" is not correlated with "Survived", we can see that there is age categories of passengers that of have more or less chance to survive.
# 
# It seems that very young passengers have more chance to survive.

# In[ ]:


# Explore Age distibution 
g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])


# When we superimpose the two densities , we cleary see a peak correponsing (between 0 and 5) to babies and very young childrens.

# #### Fare

# In[ ]:


dataset["Fare"].isnull().sum()


# In[ ]:


#Fill Fare missing values with the median value
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())


# Since we have one missing value , i decided to fill it with the median value which will not have an important effect on the prediction.

# In[ ]:


# Explore Fare distribution 
g = sns.distplot(dataset["Fare"], color="m", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")


# As we can see, Fare distribution is very skewed. This can lead to overweigth very high values in the model, even if it is scaled. 
# 
# In this case, it is better to transform it with the log function to reduce this skew. 

# In[ ]:


# Apply log to Fare to reduce skewness distribution
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)


# In[ ]:


g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))
g = g.legend(loc="best")


# Skewness is clearly reduced after the log transformation

# ### 3.2 Categorical values
# #### Sex

# In[ ]:


g = sns.barplot(x="Sex",y="Survived",data=train)
g = g.set_ylabel("Survival Probability")


# In[ ]:


train[["Sex","Survived"]].groupby('Sex').mean()


# It is clearly obvious that Male have less chance to survive than Female.
# 
# So Sex, might play an important role in the prediction of the survival.
# 
# For those who have seen the Titanic movie (1997), I am sure, we all remember this sentence during the evacuation : "Women and children first". 

# #### Pclass

# In[ ]:


# Explore Pclass vs Survived
g = sns.factorplot(x="Pclass",y="Survived",data=train,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# In[ ]:


# Explore Pclass vs Survived by Sex
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# The passenger survival is not the same in the 3 classes. First class passengers have more chance to survive than second class and third class passengers.
# 
# This trend is conserved when we look at both male and female passengers.

# #### Embarked

# In[ ]:


dataset["Embarked"].isnull().sum()


# In[ ]:


#Fill Embarked nan values of dataset set with 'S' most frequent value
dataset["Embarked"] = dataset["Embarked"].fillna("S")


# Since we have two missing values , i decided to fill them with the most fequent value of "Embarked" (S).

# In[ ]:


# Explore Embarked vs Survived 
g = sns.factorplot(x="Embarked", y="Survived",  data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# It seems that passenger coming from Cherbourg (C) have more chance to survive.
# 
# My hypothesis is that the proportion of first class passengers is higher for those who came from Cherbourg than Queenstown (Q), Southampton (S).
# 
# Let's see the Pclass distribution vs Embarked

# In[ ]:


# Explore Pclass vs Embarked 
g = sns.factorplot("Pclass", col="Embarked",  data=train,
                   size=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")


# Indeed, the third class is the most frequent for passenger coming from Southampton (S) and Queenstown (Q), whereas Cherbourg passengers are mostly in first class which have the highest survival rate.
# 
# At this point, i can't explain why first class has an higher survival rate. My hypothesis is that first class passengers were prioritised during the evacuation due to their influence.

# ## 4. Filling missing Values
# ### 4.1 Age
# 
# As we see, Age column contains 256 missing values in the whole dataset.
# 
# Since there is subpopulations that have more chance to survive (children for example), it is preferable to keep the age feature and to impute the missing values. 
# 
# To adress this problem, i looked at the most correlated features with Age (Sex, Parch , Pclass and SibSP).

# In[ ]:


# Explore Age vs Sex, Parch , Pclass and SibSP
g = sns.factorplot(y="Age",x="Sex",data=dataset,kind="box")
g = sns.factorplot(y="Age",x="Sex",hue="Pclass", data=dataset,kind="box")
g = sns.factorplot(y="Age",x="Parch", data=dataset,kind="box")
g = sns.factorplot(y="Age",x="SibSp", data=dataset,kind="box")


# Age distribution seems to be the same in Male and Female subpopulations, so Sex is not informative to predict Age.
# 
# However, 1rst class passengers are older than 2nd class passengers who are also older than 3rd class passengers.
# 
# Moreover, the more a passenger has parents/children the older he is and the more a passenger has siblings/spouses the younger he is.

# In[ ]:


# convert Sex into categorical value 0 for male and 1 for female
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})


# In[ ]:


g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)


# The correlation map confirms the factorplots observations except for Parch. Age is not correlated with Sex, but is negatively correlated with Pclass, Parch and SibSp.
# 
# In the plot of Age in function of Parch, Age is growing with the number of parents / children. But the general correlation is negative.
# 
# So, i decided to use SibSP, Parch and Pclass in order to impute the missing ages.
# 
# The strategy is to fill Age with the median age of similar rows according to Pclass, Parch and SibSp.

# In[ ]:


# Filling missing value of Age 

## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        dataset['Age'].iloc[i] = age_pred
    else :
        dataset['Age'].iloc[i] = age_med



# In[ ]:


g = sns.factorplot(x="Survived", y = "Age",data = train, kind="box")
g = sns.factorplot(x="Survived", y = "Age",data = train, kind="violin")


# No difference between median value of age in survived and not survived subpopulation. 
# 
# But in the violin plot of survived passengers, we still notice that very young passengers have higher survival rate.

# ## 5. Feature engineering
# ### 5.1 Name/Title

# In[ ]:


dataset["Name"].head()


# The Name feature contains information on passenger's title.
# 
# Since some passenger with distingused title may be preferred during the evacuation, it is interesting to add them to the model.

# In[ ]:


# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"].head()


# In[ ]:


g = sns.countplot(x="Title",data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45) 


# There is 17 titles in the dataset, most of them are very rare and we can group them in 4 categories.

# In[ ]:


# Convert to categorical values Title 
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)


# In[ ]:


g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])


# In[ ]:


g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")
g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")


# "Women and children first" 
# 
# It is interesting to note that passengers with rare title have more chance to survive.

# In[ ]:


# Drop Name variable
dataset.drop(labels = ["Name"], axis = 1, inplace = True)


# ### 5.2 Family size
# 
# We can imagine that large families will have more difficulties to evacuate, looking for theirs sisters/brothers/parents during the evacuation. So, i choosed to create a "Fize" (family size) feature which is the sum of SibSp , Parch and 1 (including the passenger).

# In[ ]:


# Create a family size descriptor from SibSp and Parch
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1


# In[ ]:


g = sns.factorplot(x="Fsize",y="Survived",data = dataset)
g = g.set_ylabels("Survival Probability")


# The family size seems to play an important role, survival probability is worst for large families.
# 
# Additionally, i decided to created 4 categories of family size.

# In[ ]:


# Create new feature of family size
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)


# In[ ]:


g = sns.factorplot(x="Single",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="SmallF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="MedF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="LargeF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")


# Factorplots of family size categories show that Small and Medium families have more chance to survive than single passenger and large families.

# In[ ]:


# convert to indicator values Title and Embarked 
dataset = pd.get_dummies(dataset, columns = ["Title"])
dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")


# In[ ]:


dataset.head()


# At this stage, we have 22 features.

# ### 5.3 Cabin

# In[ ]:


dataset["Cabin"].head()


# In[ ]:


dataset["Cabin"].describe()


# In[ ]:


dataset["Cabin"].isnull().sum()


# The Cabin feature column contains 292 values and 1007 missing values.
# 
# I supposed that passengers without a cabin have a missing value displayed instead of the cabin number.

# In[ ]:


dataset["Cabin"][dataset["Cabin"].notnull()].head()


# In[ ]:


# Replace the Cabin number by the type of cabin 'X' if not
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])


# The first letter of the cabin indicates the Desk, i choosed to keep this information only, since it indicates the probable location of the passenger in the Titanic.

# In[ ]:


g = sns.countplot(dataset["Cabin"],order=['A','B','C','D','E','F','G','T','X'])


# In[ ]:


g = sns.factorplot(y="Survived",x="Cabin",data=dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])
g = g.set_ylabels("Survival Probability")


# Because of the low number of passenger that have a cabin, survival probabilities have an important standard deviation and we can't distinguish between survival probability of passengers in the different desks. 
# 
# But we can see that passengers with a cabin have generally more chance to survive than passengers without (X).
# 
# It is particularly true for cabin B, C, D, E and F.

# In[ ]:


dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")


# ### 5.4 Ticket

# In[ ]:


dataset["Ticket"].head()


# It could mean that tickets sharing the same prefixes could be booked for cabins placed together. It could therefore lead to the actual placement of the cabins within the ship.
# 
# Tickets with same prefixes may have a similar class and survival.
# 
# So i decided to replace the Ticket feature column by the ticket prefixe. Which may be more informative.

# In[ ]:


## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 

Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
        
dataset["Ticket"] = Ticket
dataset["Ticket"].head()


# In[ ]:


dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")


# In[ ]:


# Create categorical values for Pclass
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")


# In[ ]:


# Drop useless variables 
dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)


# In[ ]:


dataset.head()


# ## 6. MODELING

# In[ ]:


## Separate train dataset and test dataset

train = dataset[:train_len].copy()
test = dataset[train_len:].copy()
test.drop(labels=["Survived"],axis = 1,inplace=True)


# In[ ]:


train.columns


# In[ ]:


## Separate train features and label 

train["Survived"] = train["Survived"].astype(int)

Y_train = train["Survived"]

X_train = train.drop(labels = ["Survived"],axis = 1)


# In[ ]:


mlp = MLPClassifier(hidden_layer_sizes=(50,50,50,50,50,50), random_state=71, max_iter=2000)


# In[ ]:


scorers = {
    'acc':make_scorer(accuracy_score),
    'f1':make_scorer(f1_score),
    'roc_auc':make_scorer(roc_auc_score)
}


# In[ ]:


val_scores = cross_validate(mlp,X_train,Y_train, cv = 10, scoring=scorers, n_jobs=-1, verbose=True)


# In[ ]:


pd.DataFrame(val_scores).mean()


# In[ ]:


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=71)


# In[ ]:


train_idx, val_idx = next(skf.split(X_train, Y_train))
train, target = X_train.iloc[train_idx], Y_train.iloc[train_idx]
X_val, target_val   = X_train.iloc[val_idx], Y_train.iloc[val_idx]


# In[ ]:


mlp.fit(train,target)
explainer = lime_tabular.LimeTabularExplainer(train.values,discretize_continuous=True, discretizer='decile',
                                              feature_names=list(train.columns),
                                              class_names=['died', 'lived'])


# In[ ]:


preds = mlp.predict(X_val)


# In[ ]:


f1_score(target_val,preds)


# In[ ]:


i = X_val.sample(1).index[0]
sample_pred = mlp.predict_proba(X_val.loc[i].values.reshape(1,-1))
exp = explainer.explain_instance(X_val.loc[i], mlp.predict_proba, num_samples=5000)


# In[ ]:


sample_pred


# In[ ]:


fig = exp.as_pyplot_figure()

