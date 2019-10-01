#!/usr/bin/env python
# coding: utf-8

# # **Titanic Survival Prediction** #
# 
# Siddharth Sharma
# 
# ***Introduction***
# 
# The premise of the problem is to categorise passengers into 'Survived', and 'Not survived' categories, who were onboard the Titanic before it sank.
# The Titanic classification problem provides ample opportunity in implementing and learning key aspects of Data cleansing, Feature analysis and Feature engineering.
# 
# ****The notebook is divided into these 4 key parts:***
# 
# 1.  Data Loading
# 2. Data Visualisation and primary cleansing
# 3. Feature Analysis
# 4. Feature Engineering
# 5. Data Compilation
# 5. Modeling
# 

# ## 1. Data Loading

# In[ ]:


### Ignore Deprecation and Future Warnings
import warnings
warnings.filterwarnings('ignore', category = DeprecationWarning) 
warnings.filterwarnings('ignore', category = FutureWarning) 

### Standard Inputs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')
plt.style.use('bmh')                    # Use bmh's style for plotting

from collections import Counter

### Sklearn Imports

# Standards

from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.grid_search import GridSearchCV


### Load Data

train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
IDtest = test["PassengerId"]

train.shape


#    ## 2. Data Visualisation and data cleansing

# In[ ]:


# Visualising Train and Test data

train.head(5)


# In[ ]:


test.head(5) 


# 
# ### Preliminary Data Assessment
# 
# The Train dataset has the following data categories:
# 
#     1. Numerical Data   : PassengerId,Pclass, Age, SibSp, Parch, Fare
#     2. Categorical Data : Name, Sex, Ticket, Cabin, Embarked
# 
# Of the aforementioned data categories in the Train set:
# *         **PassengerId** is a unique ID corresponding to each passenger.
# *         **Survived** is the flag indicating whether the passenger Survived(=1) or died (=0)
# 
# 
# ### Primary Data Cleansing:
# 
# The raw data provided has the following issues which need to be tacked:
# 
#     1. Numerical data may have outliers present, these outlier tend to skew the data which is detrimental for the classification algorithms.
#     2. Both numerical and categorical data may have missing or NaN values for certain passengers. These need to be appropriately rectified before using it
# 

# **Outlier Detection:**
# Outliers in the numerical data is detected using the Inter-quartile Range Method.
# 
# The function reads a dataframe,and returns a list of indices of the dataframe having more
# than 'n' outliers according to IQR method

# In[ ]:


# Outlier Detection (Interquartile Range)

def IQR_outlier(df,n,features):
  
    outlier_indices=[]
    
    #Iterating over features(columns)
    
    for col in features:
        Q1=np.percentile(df[col],25)
        Q3=np.percentile(df[col],75)
        
        # Interquartile range
        IQR=Q3-Q1
        
        # outlier step
        outlier_step=1.5*IQR
        
        outlier_list_col=df[(df[col]<Q1-outlier_step)|(df[col]>Q3+outlier_step)].index
        print('Total Number Outliers of',col,' : ',len(outlier_list_col))
        print('Percentage of Outliers of',col,' : ',np.round(len(outlier_list_col)/len(df[col])*100),'%')
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices=Counter(outlier_indices)
    multiple_outliers=list(k for k, v in outlier_indices.items() if v>n)
    
    return multiple_outliers

num_features=['Age','SibSp','Parch','Fare']

Outliers_to_drop=IQR_outlier(train,2,num_features)

print('Total number of Outlier indices : ', len(Outliers_to_drop))


# In[ ]:


train.loc[Outliers_to_drop,num_features]


# Overall there there 10 passenger data having more than 2 of its feaures as outliers.
# Another key observation is that Age in the Train dataset has no outlier according to the IQR method.
# 

# In[ ]:


# Visualing Age, SibSp, Parch and Fare data with and without outliers

num_features=['Age','SibSp','Parch','Fare']

for feature in num_features:
    plt.figure()
    g=sns.boxplot(x=feature,data=train)

print('Skewness :')
print(train[num_features].skew())


# As previously observed the Skewness is the least in Age, and most in Fare.
# A Skewness value between [-0.5 : 0.5] corresponds to the data being normally distributed. 
# A value greater than 1 is symptomatic of highly skewed data.
# 
# We'll drop outliers and check how the skewness is affected.

# In[ ]:


### Dropping outliers
train_temp=train.copy()
train_temp=train_temp.drop(Outliers_to_drop,axis=0).reset_index(drop=True)

print('Skewness in Data without outliers :')
print(train_temp[num_features].skew())

for feature in num_features:
    plt.figure()
    g=sns.boxplot(x=feature,data=train_temp)


# In[ ]:


print('Skewness in Data with outliers :')
print(train[num_features].skew())

print('Skewness in Data without outliers :')
print(train_temp[num_features].skew())


# 
# Upon dropping the outliers the skewness of SibSp has improved, but is still positively skewed. Moreso, the skewness for Parch and Fare has considerably increased.
# 
# **Therefore we won't drop the outliers in the data**.

# **Null Value Detection**
# 
# We will now identify the various features that have missing or NaN values as entries.
# 
# These need to be logically imputed for both the Train and Test set.
# 
# So we will combine the two sets. There can be a case made for possible data leakage as imputation would be done using data from both the sets. 
# But it would be wiser to use all available data for imputing the data.
# Also the model will be eventually evaluated with the hold-out set, so it will generalise better. 

# In[ ]:


## Joining Train and Test set to get same number of features during categorical conversion

train_len = len(train)
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
 
print('Dataset shape :',dataset.shape)

# Fill empty and NaNs with NaN        

dataset=dataset.fillna(np.nan)

dataset.isnull().sum()


# Survived missing values are because of the Test set, where we eventually have to predict the Survived values

# In[ ]:


# Information

train.info()
train.dtypes


# In[ ]:


# Data Summary

train.describe()


# ## 3. Feature Analysis
# 
# In this section we will analyse the different numerical and categorical features of the combined dataset.
# The multiple features would be analysed to gauge their distribution in the dataset, and their correlation with each other as well.
# 
# We will start by calculating the correlation between the numerical variables with respect to each other, and also with our target variable, 'Survived' 

# In[ ]:


### Correlation matrix in numerical values

num_features.insert(0,'Survived')
g=sns.heatmap(train[num_features].corr(),annot=True, fmt = ".2f")


# Key takeaways from the correlation plot:
# 
# 1.  Of the 4 numerical features, Fare correlates the best with Survived, our Target variable.
# 2.  Age has  a strong correlation with SibSp
#             This makes sense because Age would have a role to play with the number of Siblings or the fact that he/she has a spouse.
# 3.  SibSp and Parch are also strongly correlated. Also the correlation is positive, which means that both increase and decrease together. 

# >  ### **Numerical Features**
# 
# #### ** 1.  Age **
# 
#     Age is a numerical variable have float type data input. Therefore we would study its distribution in the dataset 

# In[ ]:


print('Skew:',train.Age.skew())
train.Age.describe()


# In[ ]:


# Explore Age vs Survived

# Plotting the distribution of Age amongst passengers who survived and wthose who didn't.
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")

# Overlapping the two plots
plt.figure()
g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])


# As we can see from the distribution plot, Age is almost normally distributed in the Train dataset.
# But the distribution is somewhat different in the two subpopulations.
# 
#  **Very young passenger** (0-10) seem to have a high probability of survival.
# **Old passengers** (60-80) seem to have lower probability of survival.
# 
# Therefore, even though Age doesn't correlate well with Survived, Age categories may. 
# 

# #### ** 2.  SibSp ** 
#         
#         SibSp would be analysed in terms of its frequency distribution in the Train dataset

# In[ ]:


# Explore SibSp feature vs Survived

g = sns.catplot(x="SibSp",y="Survived",data=train,kind="bar", height = 6 , palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# Passengers having low SibSp( 1 or 2)  values have a higher chance of surviving as compared to those without siblings/spouses (SibSp=0) and those having many siblings (SibSp>=3)

# #### ** 3. Parch ** 
#         
#        Parch would be analysed in terms of its frequency distribution in the Train dataset

# In[ ]:


# Explore Parch feature vs Survived
g=sns.catplot(x='Parch',y='Survived',data=train,kind='bar',height=6,palette='muted')
g.despine(left=True)
g = g.set_ylabels("survival probability")


# Different family sizes have varying survival probablilities.
# Small families(Parch = 1,2) have higher survival probabilities than Single (Parch= 0), Medium(Parch= 3,4), and Large(Parch= 5,6) families.
# 
# The std deviation for Parch=3 is pretty high. 
# This was previously observed when we were identifying outliers in the data (24% of the Parch data in the Train dataset was identified as outlier)

# #### ** 4. Fare** 
#         
#        Fare is a numerical variable have float type data input. Therefore we would study its distribution in the dataset 

# In[ ]:


dataset["Fare"].isnull().sum()


# In[ ]:


# Imputing the 1 missing value with median value of the combined dataset
dataset["Fare"]=dataset["Fare"].fillna(dataset["Fare"].median())


# In[ ]:


# Explore Fare distribution
g=sns.distplot(dataset['Fare'],label='Skewness: %2f'%(dataset['Fare'].skew()))
g.legend(loc='best')


# The data is highly skewed towards the left. This will affect the overall distribution if fed like this in the classification models.
# We would thus take the natural log of the data, thereby reducing the skewness. 

# In[ ]:


dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)


# In[ ]:


g=sns.distplot(dataset['Fare'],label='Skewness: %2f'%(dataset['Fare'].skew()))
g.legend(loc='best')


# The data seems normally distributed now. This can now be used for further analysis

# >  ### **Categorical Features**
# 
# 
# #### **5. Sex**

# In[ ]:


### Sex

g=sns.catplot(x='Sex',y='Survived',data=train,kind='bar')
train[['Sex','Survived']].groupby('Sex').mean()


# The probability of survival is a lot higher for Females than Males.

# In[ ]:


# Converting sex to categorical values 0: male, 1:female

dataset['Sex']=dataset['Sex'].map({'male':0,'female':1})


# #### **6. Pclass**

# In[ ]:


### Pclass

g=sns.catplot('Pclass','Survived',hue='Sex',data=train,kind='bar')
g=sns.catplot('Pclass','Survived',data=train,kind='bar')


# The probability of survival is highest for the members of the Pclass 1, followed by Pclass 2. This confirms the fact that the evacuation was prioritised with respect to class.
# Also,  it can be observed that more women survived than men in all classes.

# #### **7. Embarked**

# In[ ]:


print('Number of Null entries: ',dataset['Embarked'].isnull().sum())
print('Most common dock: ',dataset.Embarked.mode()[0])


# In[ ]:


# Filling Embarked with most common dock 

dataset['Embarked']=dataset['Embarked'].fillna('S')

# Exploring Embarked

g=sns.catplot(x='Embarked',y='Survived',data=train,kind='bar')

# Exploring embarked with Pclass

g=sns.catplot(x='Embarked',y='Survived',hue='Pclass',data=train,kind='bar')
g=sns.catplot(x='Pclass',col='Embarked',data=train,kind='count')


# * Of the three ports,  C has the highest survival rate, followed by S.
# * From the count plot, it can be further deduced that most of the passengers boarding from C were in 1st class. 
# * Maximum number of passengers boarded from S
# * Most of passengers from S, and Q were from the 3rd class.

# #### **7. Name**

# In[ ]:


dataset.Name.head(5)


# The variable Name on it's own may not be a useful addition for our classification problem. We would try and learn Titles of the passengers aboard. (We will do this in the Feature Engineering section)

# #### **8. Cabin**

# In[ ]:


# Visualising Cabin data
dataset.Cabin.head(5)


# In[ ]:


display(dataset.Cabin.shape)
print('Number of null values : ', dataset.Cabin.isnull().sum())
print('Percentage of null values : ', round(dataset.Cabin.isnull().sum()/dataset.Cabin.shape[0]*100),'%')


# A major chunk of Cabin values are missing in the combined dataset.
# We would thus impute the missing data in the Feature engineering section.

# #### **9. Ticket**

# In[ ]:


dataset.Ticket.head(10)


# There are Ticket entries having Prefixes that may be of significance, as it may indicate the level of the ship. Different class tickets would also have different prefixes. Therefore a lot can be learned from the Tickets as well.

# ## 3. Feature Engineering
# 
# Now that we have analysed the various features, there are a few modifications that we need to make to the existng dataset before it becomes usable for model training and evaluation.
# 
# Key aspects of Feature engineering are:
# 1.  Data imputation
# 2. Modifying existing features
# 3. Identifying new features

# ### Data Imputation

# In[ ]:


# Identifying missing values in the dataset

dataset.isnull().sum()


# #### 1. Age
# 
# There are 263 missing values in the Age category.
# We will identify the different features that may be correlated with Age and use them while imputing missing data

# In[ ]:


# Features most correlated with age
g=sns.heatmap(dataset[['Age','Fare','Parch','Pclass','Sex','SibSp']].corr(),annot=True,fmt='.2f')


# * Of the 4 aforementioned features, Sex is the least correlated with Age.  
# * Pclass is the most,  with a value of -0.41. This means that the Age and Pclass are inversely related. A lower Pclass in value(1,2, or 3) has passengers which are older.
# * A similar negative correlation is there with SibSp, and Parch as well. More the number of siblings or children a passenger has, his/her age would be relatively less.
# * Age and Fare have a positive correlation. This means that Older passengers are more likely to pay higher than younger ones. This may well be because Fare and Pclass are highly correlated (-0.69), and since class 1 passengers are more likely to be older, they would be paying a higher fare than the younger ones.
# 
# 
# We will further look at factor plots comparing Age with SibSp, Sex, Parch, Pclass,

# In[ ]:


g=sns.catplot(y='Age',x='SibSp',data=dataset,kind='box')

g=sns.catplot(y='Age',x='Sex',data=dataset,kind='box')
g=sns.catplot(y='Age',x='Sex',hue='Pclass',data=dataset,kind='box')

g=sns.catplot(y='Age',x='Parch',data=dataset,kind='box')

g=sns.catplot(y='Age',x='Pclass',data=dataset,kind='box')


# * Age vs SibSp :  An increase in the number of simblings indicates a passenger who's relatively young. Matches with the previous correlation matrix
# * Age vs Sex :  The distribution is fairly similar for the two sexes.  Sex is thus not a good indicator of Age. Also, the higher class passengers are older than lower class passengers for both sex sub populations.
# * Age vs Parch : It indicates a trend slightly different that what was inferred from the correlation matrix. More the number of childer, older is the person expected to be.
# * Age vs Pclass :  As previously observed, The Class 1 passengers are older than Class 2 , and Class 2 are older than Class 3.
# 

# In[ ]:


plt.figure()
g=sns.lineplot(x='Pclass',y='Fare',data=dataset)
plt.figure()
g=sns.lineplot(x='Age',y='Fare',data=dataset)


# Fare seems fairly monotonic with Pclass. We can this simply use Pclass for Age imputation.
# 
# 
# 
# Of the above mentioned features we would use **Parch, SibSb, Pclass** for filling missing Age values.
# 
# ***Imputation methodology :***
# 
# We would identify Age rows having similar Parch, Pclass, and SibSp values to that of the passenger not having an Age value, and use their median Age value for imputation. In a condition where no other passenger has the same value of Parch, Pclass and SibSp, we would impute the missing Age with the global median.

# In[ ]:


## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp

# Index of NaN age rows
index_NaN_age = dataset["Age"][dataset.Age.isnull()].index

for i in index_NaN_age :
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset['SibSp'][i]) & (dataset['Parch'] == dataset['Parch'][i]) & (dataset['Pclass'] == dataset['Pclass'][i]))].median()
    if not np.isnan(age_pred) :
        dataset['Age'].iloc[i] = age_pred
    else :
        dataset['Age'].iloc[i] = age_med
        


# In[ ]:


dataset.isnull().sum()


# Now only Cabin has missing values, these will be imputed as we further modify the existing dataset.

# ### 2. Modifying existing features
# 
# 
# 
# 

# #### 1. Cabin

# In[ ]:


dataset.Cabin[dataset.Cabin.notna()].head(10)


# The first letter of the Cabin indicates the level at which it is situated. During the time of evacuation, it is highly likely that the Cabin value may have a bearing on the probability of survival.

# In[ ]:


## Replacing each Cabin entry with the first letter, and replacing the missing cabin entries with 'X'

dataset.Cabin=pd.Series(['X' if pd.isnull(i) else i[0] for i in dataset.Cabin])                        


# In[ ]:


dataset.head()


# In[ ]:


print('Mean : ', dataset[['Cabin','Survived']].groupby('Cabin').mean())
print('Count : ', dataset[['Cabin','Survived']].groupby('Cabin').count())


# In[ ]:


g=sns.catplot(x='Cabin',y='Survived',data=dataset,kind='bar', order=['A','B','C','D','E','F','G','T','X']) 


# As it can be seen from the count values, most of the Cabin entries were missing. 
# Also due to very less number of entries in the  known Cabins (A to T), the distribution has high variance.
# 
# One key inference that can be drawn, is that the Survival probability is a lot lower for passengers who have their cabin entries missing.

# #### 2. Name
# 
# As discussed previously, Titles can be extracted from the Name of the passengers, these may hold useful information regarding the survival probability of the passenger

# In[ ]:


dataset.Name.head()


# In[ ]:


# The title is mentioned as the 2 word in the string

dataset['Title']=pd.Series([i.split(',')[1].split('.')[0].strip() for i in dataset.Name])


# In[ ]:


dataset.head()


# In[ ]:


g=sns.countplot(x='Title',data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45)


# There are several titles with the very least frequency. So, it makes sense to put them in fewer buckets. We would replace Mlle and Ms with Miss and Mme by Mrs as these are French titles. Also we will club all the other titles under 'Rare'.
# 
# 
# **We will create categories from these titles:**
# 
#     0: Master
#     1: Miss(Miss, Ms, Mlle)
#     2: Mrs(Mrs,Mme)
#     3: Mr
#     4: Rare (Dr, Rev, Col, Major, Capt,Dona, Jonkheer, Countess, Sir, Lady, Don)
# 
# 

# In[ ]:


# Convert to categorical values Title 
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
dataset["Title"] = dataset["Title"].replace(['Ms', 'Mlle'], 'Miss')
dataset["Title"] = dataset["Title"].replace(['Mme'], 'Mrs')


# In[ ]:


dataset[['Name','Title']].head(5)


# #### 3. Ticket
# 
# As discussed previously, there are Ticket entries having Prefixes that may be of significance, as it may indicate the level of the ship. 

# In[ ]:


dataset['Ticket'].head(10)


# Some Ticket entries have just numerical data, while some have alphanumeric prefixes.
# we would try and club ticket entries ahving the same prefixes together, also all entries having no prefixes would be clubbed separately.

# In[ ]:


## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X.

Ticket=[]
for i in list(dataset['Ticket']):
    if i.isdigit():
        Ticket.append('X')
        
    else:
        Ticket.append(i.split(' ')[0])

dataset['Ticket']=Ticket


# In[ ]:


dataset.Ticket.head(10)


# In[ ]:


dataset.Ticket.unique()


# It can be observed that a lot of prefixes have character values( " / "  ,  '" . "  ) 
# We would try and remove these characters, and use only the reduced alphanumeric prefixes as categories.
# 

# In[ ]:


Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip()) #Take prefix
    else:
        Ticket.append(i)

dataset['Ticket']=Ticket


# In[ ]:


dataset.Ticket.unique()


# ### 3. Identifying new features

# #### 1. Family size
# 
# The size of the family may be an important factor since a larger family size would make it difficult during evacuation. 
# Family size will be calculated using SibSp and Parch

# In[ ]:


# Create a family size descriptor from SibSp and Parch

dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1

g = sns.factorplot(x="Fsize",y="Survived",data = dataset)
g = g.set_ylabels("Survival Probability")


# It can be infered from the above plot that a small to medium family (2-4) has a higher survival probability than individuals ( Fsize=1) and large families (5 and above) 
# 
# We would further convert Fsize into categorical bins

# In[ ]:


dataset["Fsize"].replace(to_replace = [1], value = 'Single', inplace = True)
dataset["Fsize"].replace(to_replace = [2], value = 'Small', inplace = True)
dataset["Fsize"].replace(to_replace = [3,4], value = 'Medium', inplace = True)
dataset["Fsize"].replace(to_replace = [5,6,7,8,11], value = 'Large', inplace = True)

g=sns.catplot(x='Fsize',y='Survived',data=dataset,kind='bar',order=['Single','Small','Medium','Large'] ) 


# Small and Medium size families have the highest survival probability, followed by Single. Large families have the poorest survival probability. 
# This may be because of difficulty arising in evacuating large families.

# ## 4. Data Compilation

# We would now compile all the features necessary for Survival prediction in the test set.
# Subsequently we will remove the unnecessary ones.
# 

# In[ ]:


dataset.head(5)


# In[ ]:


# Converting categorical data to usable form

dataset = pd.get_dummies(dataset, columns = ["Cabin"], prefix="Cab")
dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
dataset = pd.get_dummies(dataset, columns = ["Fsize"], prefix="Fam") 
dataset = pd.get_dummies(dataset, columns = ["Pclass"], prefix="Pc")
dataset = pd.get_dummies(dataset, columns = ["Title"], prefix="Title") 
dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="Tick")


# In[ ]:


# Dropping the unnecessary variables
# labels : Name, Passenger Id
dataset.drop(labels = ['Name','PassengerId'], axis = 1, inplace = True)


# In[ ]:


dataset.head(2)


# In[ ]:


dataset.shape


# In[ ]:


from sklearn.decomposition import PCA
dataset1=dataset.copy()

dataset1.drop(labels='Survived',axis=1,inplace=True)
pca=PCA(0.999,whiten=True).fit(dataset1)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
data=pca.transform(dataset1)

data.shape


# #### The data can now be used to train and test classification models

# ## 5. Modeling

# In[ ]:


#train=dataset[:train_len]
#test=dataset[train_len:]

#test.drop(labels='Survived',axis=1,inplace=True)

#train['Survived']=train['Survived'].astype(int)

#Y_train=train['Survived']

#X_train=train.drop(labels='Survived',axis=1)

Y_train=dataset[:train_len]['Survived']
X_train=data[:train_len]
test=data[train_len:]




# In[ ]:


#train=dataset[:train_len]
#test=dataset[train_len:]

#test.drop(labels='Survived',axis=1,inplace=True)

#train['Survived']=train['Survived'].astype(int)

#Y_train=train['Survived']

#X_train=train.drop(labels='Survived',axis=1)


# In[ ]:


#X_train.head()


# In[ ]:


#Y_train.head()


# ### 1. Simple Modeling
# 
# 
# Comparing 5 popular classifiers and evaluate the mean accuracy of each of them by a stratified kfold cross validation procedure.
# * SVC
# * Decision Tree
# * Random Forest
# * KNN
# * Logistic regression
# 

# In[ ]:


### Model Imports

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


# In[ ]:


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10) 


# In[ ]:


# Modeling step to test differents algorithms 

random_state = 42
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))

cv_results = [] 
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","RandomForest","KNeighbours","LogisticRegression"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")



# In[ ]:


cv_res.head()


# Of the 5 classifiers, SVC, Random Forest and Logistic Regression have the best cross validation performance.
# 
# We would now try and tune the hyper parameters for the above classifiers, so as to maximise their accuracy.

# ### 2. Hyperparameter tuning
# 
# We would use Grid Search Cross validation approach to fine tune the parameters fro different classifiers 

# #### 1. SVC Classifier

# In[ ]:


SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 'gamma': [ 0.001, 0.01, 0.1, 1],'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(X_train,Y_train)

SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_


# #### 2. Decision Tree Classifier

# In[ ]:


DTC = DecisionTreeClassifier()

dt_param_grid = {'max_features': ['auto', 'sqrt', 'log2'],'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
                 'min_samples_leaf':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],'random_state':[42]}

gsDTC = GridSearchCV(DTC,param_grid = dt_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsDTC.fit(X_train,Y_train)

DTC_best = gsDTC.best_estimator_

# Best score
gsDTC.best_score_


# #### 3. Random Forest Classifier

# In[ ]:


RFC = RandomForestClassifier()

rf_param_grid = {"max_depth": [None],"max_features": [1, 3, 10],"min_samples_split": [2, 3, 10],"min_samples_leaf": [1, 3, 10],
                 "bootstrap": [False],"n_estimators" :[100,300],"criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_


# #### 4. Logistic Regression Classifier

# In[ ]:


LRC = LogisticRegression() 

lr_param_grid = {'penalty':['l1', 'l2'],'C': np.logspace(0, 4, 10)}

gsLRC=GridSearchCV(LRC,param_grid=lr_param_grid,cv=kfold,scoring='accuracy', n_jobs= 4, verbose = 1)

gsLRC.fit(X_train,Y_train)

LRC_best = gsLRC.best_estimator_

# Best score
gsLRC.best_score_


# #### 5. KNN Classifier

# In[ ]:


KNNC = KNeighborsClassifier()
knn_param_grid = {'n_neighbors':[3, 4, 5, 6, 7, 8],'leaf_size':[1, 2, 3, 5],
              'weights':['uniform', 'distance'],'algorithm':['auto', 'ball_tree','kd_tree','brute']}

gsKNNC=GridSearchCV(KNNC,param_grid=knn_param_grid,cv=kfold,scoring='accuracy',n_jobs=4,verbose=1)

gsKNNC.fit(X_train,Y_train)

KNN_best=gsKNNC.best_estimator_

# Best score
gsKNNC.best_score_


# After hyper parameter tuning, SVC is the best performing classifier, followed by RF and LR. 
# KNN is still the least accurate of the 5.

# ###  3. Plot learning curves
# 
# Learning curves are a good way to see the overfitting effect on the training set and the effect of the training size on the accuracy.

# In[ ]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsDTC.best_estimator_,"DT learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsRFC.best_estimator_,"RF learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsLRC.best_estimator_,"LR learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsKNNC.best_estimator_,"KNN learning curves",X_train,Y_train,cv=kfold)


# Decision Tree and KNN tend to overfit the training set. Their performance can be improved with increasing the training set size
# SVC and LR are better generalised and have their Training score and CV score converging to the same value

# ### 4. Feature importance of tree based classifiers
# 
# In order to see the most informative features for the prediction of passengers survival, the feature importance for the 2 tree based classifiers is displayed.

# In[ ]:


'''
names_classifiers = [("RandomForest",RFC_best),("Desicion Tree",DTC_best)]

for nclassifier in range(2):
    name = names_classifiers[nclassifier][0]
    classifier = names_classifiers[nclassifier][1] 
    indices = np.argsort(classifier.feature_importances_)[::-1][:40]
    plt.figure(figsize=(8,8))
    g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h')
    g.set_xlabel("Relative importance",fontsize=12)
    g.set_ylabel("Features",fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title(name + " : feature importance")
'''    


# We note that the two classifiers have different top features according to the relative importance. 
# 
# It means that their predictions are not based on the same features. Nevertheless, they share some common important features for the classification , for example 'Fare', 'Title_Mr', 'Title_Mr's' and 'Age'.
# 
# **According to the feature importance of this 2 classifiers, the prediction of the survival seems to be more associated with the Age, the Sex, the family size and the social standing of the passengers than the location in the boat.**

# ### 5. Test Data prediction
# 
# 
# We would now predict the survival probability of the passengers in the Test dataset.
#  
#  We would drop KNN as its performance has been the least accurate of the 5 classifiers.

# In[ ]:


test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVC")
test_Survived_DTC = pd.Series(DTC_best.predict(test), name="DTC")
test_Survived_RFC = pd.Series(RFC_best.predict(test), name="RFC")
test_Survived_LRC = pd.Series(LRC_best.predict(test), name="LRC")


# Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_SVMC,test_Survived_DTC,test_Survived_RFC,test_Survived_LRC],axis=1)


g= sns.heatmap(ensemble_results.corr(),annot=True)


# The prediction seems to be quite similar for the 4 classifiers. Although the prediction is fairly similar there are a few differences in the output. 
# We would thus consider an ensembling vote. 

# ### 5. Ensemble Modeling using Voting Classifier
# 
# Argument "soft" would be passed to the voting parameter to take into account the probability of each vote.

# In[ ]:


votingC = VotingClassifier(estimators=[('svc', SVMC_best),('rfc', RFC_best), ('lrc', LRC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)


# ### 6. Prediction and Submission

# In[ ]:


'''
test_Survived = pd.Series(votingC.predict(test), name="Survived")

results = pd.concat([IDtest,test_Survived],axis=1)

results.to_csv("Titanic_test_set_prediction.csv",index=False)

'''


# In[ ]:


test_Survived = votingC.predict(test).astype(int)
submission = pd.DataFrame({
        "PassengerId": IDtest,
        "Survived": test_Survived
    })
submission.to_csv('Titanic_test_prediction_V9.csv', index=False)


# In[ ]:


accuracy_score(Y_train,votingC.predict(X_train))

