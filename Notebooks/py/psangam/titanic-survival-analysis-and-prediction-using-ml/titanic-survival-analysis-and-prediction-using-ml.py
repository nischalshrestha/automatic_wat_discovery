#!/usr/bin/env python
# coding: utf-8

# 
# After looking around for inspiration from fellow kagglers, i gained some great insights about the Titanic dataset. I noticed Random Forest Trees to be the go-to technique for prediction in this competition. Having python at my disposal, i figured i am interested in attempting multiple ML techniques and hope to analyze how they fare. 
# 
# ## Project Plan: 
# ![img](https://github.com/PrathyushaSangam/DataScienceUsingPython/blob/master/Images/DataScience%20Proj%20Lifecycle.png?raw=true)
# 
# ***Reference: Zumel, N. and Mount, J. (2014). Practical data science with R. Shelter Island, NY: Manning Publications Co., p.6.***
# 
# This image I saw on the first day school, showing how a Data Science Project Lifecycle works, really stuck around. It shows what questions we need to address, how we may go back and forth in various stages. It's a good starting point for anyone, to lay out a project plan.
# 
# ** How do we go ahead? **  
# * Define the objective
# * Set up helpful functions
# * Load data
# * Perform preliminary analysis on what we have
# * Try digging deeper - finding correlations, importance of variables
# * Data Cleaning - dealing with missing values
# * Feature Engineering / Extraction - creation and convertion
# * Dropping unfixable/unimportant features
# * Modelling using multiple ML techniques
# * Evaluation
# 
# ## Objective:
# 1.	Analysing the Titanic dataset to obtain useful and interesting insights.
# 2.	Predicting survival on the Titanic using multiple ML techniques.
# 
# ## Setting up libraries and functions

# In[ ]:


# Data Analysis
import numpy as np 
import pandas as pd

# Data Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# allowing multiple/scrollable outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#machine Learning

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from fancyimpute import KNN


# In[ ]:


#Setting up functions for visualization

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( plt.hist , var , alpha=0.5)
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()
    
# Function inspired by Helge Bjorland: An Interactive Data Science Tutorial


# ## Load Data
# We create a combined list of dataframes - df_full, to allow us perform operations both datasets require at once, using a simple for loop.

# In[ ]:


# Load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

df_full = [df_train, df_test]


# ## Preliminary/Descriptive Analysis 
# Now that we have our data ready, we can look at the variables, shape of the data, missing values and some standard descriptive stats. Further, we can figure out sensible data cleaning and possible feature extraction/engineering opportunities.
# 

# In[ ]:


# preliminary analysis
print('Shape of the Data -> Train:', df_train.shape, 'Test:', df_test.shape)
pd.crosstab(index = df_train['Survived'], columns = "count")


# In[ ]:


# preliminary analysis
df_train.describe()
df_train.isnull().sum()
df_test.isnull().sum()

#for understanding datatypes of columns
#df_train.info()
#df_test.info()


# ### What we know so far:
# * Train has 891 records, Test has 418 records, amounting to a total of 1309. The original passenger count on Titanic was 2208. So we can say we have 59.3% of data for this competition.
# 
# *  From our dataset, we see 38.3% chance of survival (Positive samples of Survival: 342 , negative samples: 549). On the Titanic, 705/2208 passengers survived i.e., 32% survived.
# 
# * Missing values - both datasets: Age - 263 (~20%), Cabin - 1014 (~77%) , Embarked- 2, Fare- 1. The missing values of Age, Fare and Embarked might be important in deciding the survival of a passenger, hence have to be filled. Though we can assume that Cabin might have been an important variable, defining the closeness to a lifeboat; with almost 77% missingness, this feature can be dropped.
# Note: NaN in Survived column of df_full (in last 418 records) are added due to coercion, since df_test has no Survived column
# 
# *  Mean age of the passengers is 29.68 and ranges between 0.42 - 80 years
# 
# * There is a large variation in fare, someone (or some people) travelled for free (\$0), the mean of fare is \$32.20, ranging upto a costly \$512.33.
# 
# * Names have titles, which describe their social status, sometimes age, marital status in women and even occupation! (Royalty, Mr, Dr, Master, Mrs and so on.) which might have some correlation with survival rates. However, names can be dropped while retaining the title.

# ## Digging Deeper - Finding correlations and variable importance
# 
# * Using Cross-tabulation/ frequency tables and visualizations, we can look at correlations among the features and also guage their importance in predicting survival. 
# 
# * High School Math Rocks: Since survival is binary (0/1), the mean of survived directly gives probability of survival in any given category! We can use this simple calculation to check the chance of survival given sex, pclass, embarkation etc.
# 
# 
# ### Categorical Variables : Pclass, Family Size (SibSp & Parch), Embarked vs Survival

# In[ ]:


table1 = pd.pivot_table(df_train, values='Survived', index=['Pclass'])
table1


# In[ ]:


for dataset in df_full:
    family_size = dataset.SibSp + dataset.Parch +1 #including themselves
    dataset['FamilySize'] = family_size


table2 = pd.pivot_table(df_train, values = 'Survived', index= ['FamilySize'])
table2


# In[ ]:


sns.barplot(x=df_train['Embarked'],y=df_train['Survived'] )


# ### Findings:
# * Women and passengers from higher classes had a better chance of survival.
# * Women in Pclass 3 had lower survival rates than Pclass 1 and 2, which are above 90%. Men from Pclass 1 had slightly better survival rates than men in other classes.
# * Having family members improves survival than those travelling alone. On the contrary bigger familes sink together!
# * Passengers who embarked in C=Cherbourg had more chance of survival, followed by Q=Queenstown
# 
# ### Continuous Variables: Age, Fare vs Survival

# In[ ]:


plot_distribution( df_train , var = 'Age' , target = 'Survived', row = 'Sex')


# In[ ]:


plot_distribution( df_train , var = 'Age' , target = 'Survived')


# In[ ]:


child = df_train[(df_train['Age']<=10)]
pd.pivot_table(child, index = ['Sex'], values = 'Survived')


# In[ ]:


plot_distribution( df_train , var = 'Fare' , target = 'Survived')


# ### Findings
# * Among males,  57.6% children age <=10 survived. The survival rate got lower as age increased, with an exception of age 80.
# * Among females, 61.2% children age <=10 survived, whereas, women in general had a good chance of survival.
# * Overall, children had better survival rates than adults
# *  Passengers who paid more fare had better chance of survival.
# 
# ### Decisions
# Age and Fare are continuous variables. These variables can be converted into categorical / ordinals by diving them into bands and assigning each band an ordinal value.
# 

# ## Feature Extraction and Data Cleaning 
# 
# **Missingness**
# * Filling missing Embarked in df_train: with most common port of embarkation
# * Filling missing fare df_test: with the median fare
# * Filling Missing Age : using KNN. lets push this task to post feature extraction, title could help with filling missing age.
# 
# **Features to numeric**
# * Converting Sex to numeric (male:0 female:1)
# * Converting Embarked to numeric (S:0, C:1, Q:2)
# 
# **Binning FamilySize**
#  * FamilySize =1 | alone|  ordinal =0
#  * FamilySize >1 and < 4 | small family | ordinal =1
#  * FamilySize >4 | Large Family | ordinal = 2
#  
# ** Binning: Age bands and Fare bands**
# * Age: Ordinal age bands - By dividing age into 5 age groups
# * Fare: Ordinal fare bands - By dividing into 4 groups based on quantiles
# 
# ** Extracting new features**
#  * **Title** counting number of titles in entire dataset (both test and train, since both need cleaning) , replacing rare titles and synonymous titles, giving category number to each title
# 
# **Dropping columns**
# * Ticket
# * PassengerId
# * Name
# * Cabin

# In[ ]:


#missing Embarked
port_mode = df_train.Embarked.mode()[0]
#port_mode
df_train['Embarked'] = df_train['Embarked'].fillna(port_mode)

#missing Fare
fare_median = df_test.Fare.median()
#fare_median
df_test['Fare'] = df_test['Fare'].fillna(fare_median)


# In[ ]:


#numeric values for Sex
for dataset in df_full:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[ ]:


#numeric values to Embarked
for dataset in df_full:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0 , 'C':1 , 'Q':2}).astype(int)


# In[ ]:


# Grouping Family Size to Ordinals
for dataset in df_full:
    dataset['FamilySize'] = dataset['FamilySize'].replace([1], 0)
    dataset['FamilySize'] = dataset['FamilySize'].replace([2,3,4], 1)
    dataset['FamilySize'] = dataset['FamilySize'].replace([5,6,7,8,9,10,11], 2)

pd.pivot_table(df_train, index = 'FamilySize' , values= 'Survived')


# In[ ]:


#quantiles for fare attribute
pd.qcut(df_train['Fare'],4, retbins=True)[1]


# In[ ]:


#creating same bins for fare bands in train and test based on quantiles in train
#giving ordinal labels 0-3

bins = [0,7.91,14.454,31.0,513.0]
labels = [0,1,2,3]

for dataset in df_full:
    dataset['Fareband'] = pd.cut(dataset['Fare'], bins=bins, labels=labels, include_lowest = True)
    dataset['Fareband'] = dataset['Fareband'].astype(int)

pd.pivot_table(df_train, index = df_train['Fareband'],values = 'Survived' )


# In[ ]:


for dataset in df_full:
 dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


# to get an idea of all titles in both datasets (to make cleaning easier)
all_titles = df_test['Title'].append(df_train['Title'])
pd.crosstab(all_titles,'count')


# In[ ]:


for dataset in df_full:
    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'],'Miss')
    dataset['Title'] = dataset['Title'].replace(['Mme'], 'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Don','Jonkheer','Major','Sir','Rev','Dr'],'Raremale')
    dataset['Title'] = dataset['Title'].replace(['Countess','Dona','Lady'],'Rarefemale')


# In[ ]:


pd.pivot_table(df_train, index = df_train['Title'], values = 'Survived')


# In[ ]:


title_map = {"Master":1, "Miss":2, "Mr":3, "Mrs":4, "Rarefemale":5, "Raremale":6}
for dataset in df_full:
    dataset['Title'] = dataset['Title'].map(title_map)
    


# In[ ]:


df_train.head()
df_test.head()


# ### Fill in missing age, creating age bands, mapping age bands to ordinals
# 

# In[ ]:


#dropping columns which we may not need / use
for dataset in df_full:
    dataset.drop(['Name','SibSp','Parch','Ticket','Cabin','Fare'], axis= 1, inplace = True)


# In[ ]:


#dataframes we are left with : age still has missing values
df_train.head()
df_test.head()


# In[ ]:


#impute age in df_train and df_test
for dataset in df_full:
    new_df = dataset[['PassengerId','Pclass','Sex','Age','Embarked','FamilySize','Fareband','Title']]
    filled = KNN(k=3).complete(new_df)
    filled = pd.DataFrame(filled, columns =['PassengerId','Pclass','Sex','Age','Embarked','FamilySize','Fareband','Title'])
#separate modifying original dataframe, add histograms for comparison
    dataset['Age'] = filled['Age']
    dataset.head()
    dataset.isnull().sum()


# In[ ]:


#hist before and after imputation
#plt.hist(filled['Age'],bins=10, alpha=0.5)
#plt.hist(new_df.Age[~np.isnan(df_train.Age)], bins =10, alpha = 0.5)


# In[ ]:


#plt.hist(filled_df['Age'],bins=10, alpha=0.5)
#plt.hist(df_train.Age[~np.isnan(df_train.Age)], bins =10, alpha = 0.5)
#plt.hist(df_test.Age[~np.isnan(df_test.Age)], bins =10, alpha = 0.5)


# In[ ]:


#Discretize age into 5 equal groups and assign ordinal agebands
pd.cut(df_train['Age'],5).unique()
bins = [0,16,32,48,64,80]
labels = [0,1,2,3,4]

for dataset in df_full:
    dataset['Ageband'] = pd.cut(dataset['Age'],bins = bins,labels = labels, include_lowest=True)
    dataset['Ageband'] = dataset['Ageband'].astype(int)

pd.pivot_table(df_train, index = ['Ageband'],values = 'Survived',columns=['Sex'])


# In[ ]:


for dataset in df_full:
    dataset.drop("Age", axis= 1, inplace = True)
    
df_train.head()
df_test.head()


# ## Model Building and Prediction
# * Creating variables for building our models
# * Training the models
# * Calculating accuracy/score (during training)

# In[ ]:


# Variables needed for building prediction model
X_train = df_train.drop(["Survived","PassengerId"], axis=1)
Y_train = df_train["Survived"]
X_test  = df_test.drop(["PassengerId"], axis=1).copy()


# In[ ]:


#logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
#Y_pred
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


#KNN k=3

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[ ]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[ ]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[ ]:


#X_train['Fareband'] = X_train['Fareband'].astype('int')
#X_train.apply(pd.to_numeric)
X_train.info()

xgb = XGBClassifier()
xgb.fit(X_train,Y_train)
y_pred = xgb.predict(X_test)

acc_xgb = round(sgd.score(X_train, Y_train) * 100, 2)
acc_xgb


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)

Y_pred

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# ## Model Evaluation
# * Comparing the scores of various ML algorithms

# In[ ]:


# Model Evaluation

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'XGBoost', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_xgb, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# ## Submission

# In[ ]:


# #competition submission: Random Forest Trees
# submission = pd.DataFrame({
#         "PassengerId": df_test["PassengerId"],
#         "Survived": Y_pred
#     })
# submission.to_csv('submission.csv', index=False)

