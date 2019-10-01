#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Being a data scientist, I finally decide it's time to explore Kaggle competitions. Titanic seems to be the prime option for new commers to Kaggle. So I am here. Ensemble methods are the most popular and powerful tools which I also want to focus in this kernel.
# 
# After quickly browsing several popular kernels, I fell in love with the clean and comprehensive exporatory data analysis in   [EDA To Prediction(DieTanic)](https://www.kaggle.com/ash316/eda-to-prediction-dietanic) by *I,Coder* and decided to fork it and start my own coding based on it. Many codes in this kernel are re-invented wheels of codes written by Kagglers althrough I wrote line by line my self.
# 
# The other script I referenced a lot is [Introduction to Ensembling/Stacking in Python](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python) by *Anisotropic*.
# 
# Before start, let me show my sincere condolences for the passengeners not able to survive in the disaster.

# # Table Contents
# 
# #### Part1: Feature Exploration, Engineering and Cleaning
# #### Part2: Ensemble modelling

# ## Part1: Feature Exploration, Engineering and Cleaning

# ### Import libraries

# In[ ]:


import numpy as np 
import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import plotly.plotly as py
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tools
init_notebook_mode(connected=True)
import cufflinks as cf
cf.set_config_file(offline=True, theme='ggplot')
from IPython.display import Image, display

import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')


# ### Load data

# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')

# Store our passenger ID for easy access
PassengerId_test = test['PassengerId']

print('The shape of the training data:', train.shape)
print('The shape of the testing data:', test.shape)
print('The features in the data:',train.columns.values)
train.head()


# The training data contains 11 independent features while **Survived** is the dependent feature to be predict. Some of the independent features are nominal features (i.e., Sex, Embarked), some are ordinal features (i.e., pClass) and some are continuous features (i.e., Age, Fare). Most of the features need to be converted to numeric values so that they can be processed by machine learning models. We also find there are features containing NaN values.

# In[ ]:


train.isnull().sum().sort_values(ascending=False)


# In[ ]:


test.isnull().sum().sort_values(ascending=False)


# **Cabin** and **Age** contain a lot of NaN values and **Embarked**and **Fare**  only has 2 and 1 NaN values. For features with many NaN values, we may want to discard it cause it does not carry enough information. The other method is interpolating with  mean or median value of other samples. 

# ### Survived

# In[ ]:


survive_counts = train['Survived'].value_counts().reset_index().replace([0,1],['Dead','Survived'])
survive_counts.iplot(kind='pie',labels='index',values='Survived',pull=.02,title='Survived Vs Die')


# Only **38.4%** of the total passengers in training set surveved. What a disaster!
# 
# Now Let's explore the relationship between those independent features and **Survived** outcome.

# ### Sex

# In[ ]:


dft = pd.crosstab(train.Sex,train.Survived,margins=True).rename(columns={0:'Dead',1:'Survived'})
dft.iloc[:-1,:].iplot(kind='bar',title='Sex: Survived Vs Dead')


# Most women survived while most men did not. Men protects women in emergency. I am proud of you, gentmen!

# ### Pclass

# In[ ]:


dft = pd.crosstab(train.Pclass,train.Survived,margins=True).rename(columns={0:'Dead',1:'Survived'})
dft.iloc[:-1,:].iplot(kind='bar',title='Pclass: Survived Vs Dead')


# 136 out of 216 people of Pclass 1 survived (accounting **63%**), 87 out of 184 of passengers of Pclass 2 survived (accounting **48%**), and only 119 out of 491 of Pclass 3 escaped from death (**25%**). Money can't buy everything, but can improve the chance of living in emergency!

# Let's combine **Sex** and **Pclass** and have a look.

# In[ ]:


dft = pd.crosstab([train.Sex,train.Pclass],train.Survived).rename(columns={0:'Dead',1:'Survived'})
dft.iplot(kind='bar',title='Pclass: Survived Vs Dead',barmode='stack')


# Female always has priority across all classes of cabins. Only 3 out of 94 women in Pclass 1 did not make it. However, the women in Pclass 3 are not that lucky although they are are already lucky compared with men companies. The following factor plot gives a more clear idea.

# In[ ]:


sns.factorplot('Pclass','Survived',hue='Sex',data=train)
plt.show()


# ### Age
# 

# In[ ]:


print('Oldest Passenger was of:',train.Age.max(),'Years')
print('Youngest Passenger was of:',train.Age.min(),'Years')
print('Average Age on the ship:',train.Age.mean(),'Years')


# In[ ]:


Age_female_survived = train[(train.Sex=='female') & (train.Survived==1)].Age
Age_female_dead = train[(train.Sex=='female') & (train.Survived==0)].Age
Age_male_survived = train[(train.Sex=='male') & (train.Survived==1)].Age
Age_male_dead = train[(train.Sex=='male') & (train.Survived==0)].Age

fig = tools.make_subplots(rows=1, cols=2,subplot_titles=('Female', 'Male'))

survived_female = go.Histogram(
    name='Survived_female',
    x=Age_female_survived
)
fig.append_trace(survived_female, 1, 1)
dead_female = go.Histogram(
    name='Dead_female',
    x=Age_female_dead
)
fig.append_trace(dead_female, 1, 1)
fig.layout.xaxis1.update({'title':'Age'})

survived_male = go.Histogram(
    name='Survived_male',
    x=Age_male_survived
)
dead_male = go.Histogram(
    name='Dead_male',
    x=Age_male_dead
)
fig.append_trace(survived_male,1,2)
fig.append_trace(dead_male,1,2)
fig.layout.xaxis2.update({'title':'Age'})
fig.layout.update({'barmode':'stack'})
iplot(fig)


# Men got higher chance of survival when they are between 25 and 40 years old, while the oldest man aged 80 was able tot survive. The elder women were even luckier cause the percentage of surviveal for those above 50 years is quite high. Children have a good survival rate irrespective of sex.

# We have already known that the **Age** feature contains 177 null values. The above plot demonstrates **Age** should be a useful feature, therefore we do not want to discard it but use interpolation method to fill null values. The age falls in a wide range, we want to divide it into multiple bins and assign the appropriate age to passengers in certain range of age. The feature engineering method of [EDA To Prediction(DieTanic)](https://www.kaggle.com/ash316/eda-to-prediction-dietanic) is brilliant, which I use here.

# The **Name** feature contains salutions which indicate the age in some way. The feature engineering method referenced Sina's well-thought work [Titanic Best Working Classfier](https://www.kaggle.com/sinakhorami/titanic/titanic-best-working-classifier)

# In[ ]:


full_data = [train, test]  # Both training and testing data need processing
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Intial, containing the titles of passenger names
for dataset in full_data:
    dataset['Initial'] = dataset.Name.apply(get_title)
# The initials need processing
for dataset in full_data:
    dataset['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess',
                             'Jonkheer','Col','Rev','Capt','Sir','Don','Dona'],
                            ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs',
                             'Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)


# Let's check whether the processing is reasonable. The following crosstab demonstrates the above processing is pretty good.

# In[ ]:


pd.crosstab(train.Sex,train.Initial).style.background_gradient(cmap='summer_r')


# Now we are ready to fill NaN ages!

# In[ ]:


train.groupby('Initial')['Age'].mean() #lets check the average age by Initials


# In[ ]:


for dataset in full_data:
    dataset.loc[(dataset.Age.isnull())&(train.Initial=='Master'),'Age']=5
    dataset.loc[(dataset.Age.isnull())&(train.Initial=='Miss'),'Age']=22
    dataset.loc[(dataset.Age.isnull())&(train.Initial=='Mr'),'Age']=33
    dataset.loc[(dataset.Age.isnull())&(train.Initial=='Mrs'),'Age']=36
    dataset.loc[(dataset.Age.isnull())&(train.Initial=='Other'),'Age']=46


# In[ ]:


train.Age.isnull().any() #So no null values left finally 


# Let's have a look at how age distribution looks like after interpolating NaN values.

# In[ ]:


Age_female_survived = train[(train.Sex=='female') & (train.Survived==1)].Age
Age_female_dead = train[(train.Sex=='female') & (train.Survived==0)].Age
Age_male_survived = train[(train.Sex=='male') & (train.Survived==1)].Age
Age_male_dead = train[(train.Sex=='male') & (train.Survived==0)].Age

age_data = [Age_female_survived, Age_female_dead,Age_male_survived,Age_male_dead]
age_groups = ['Survived_female', 'Dead_female','Survived_male','Dead_male']
fig = ff.create_distplot(age_data, age_groups, bin_size=3)
iplot(fig)


# **Observations**
# - Female with age between 21 and 25 has the highest sruvival rate while the largest group of dead female also lies in similar age rage. 
# - Concerning male, survived children are more than dead ones, especially for those under 4. The 24-27 young men and 33-37 strong men also have a bigger than 50% survival rate.
# 

# We can also use factor map to investigate the impact of age on survival rate of people from different Pclasses.

# In[ ]:


sns.factorplot('Pclass','Survived',hue='Initial',data=train)
plt.show()


# Note that **Master** refers to children acoording the mean age. The Women and Child first policy thus holds true irrespective of the class.

# ### Embarked

# In[ ]:


dft = pd.crosstab(train.Embarked,train.Survived,margins=True).rename(columns={0:'Dead',1:'Survived'})
dft.iloc[:-1,:].iplot(kind='bar',title='Embarked: Survived Vs Dead')


# In[ ]:


sns.factorplot('Embarked','Survived',data=train)
sns.factorplot('Embarked','Survived',hue='Sex',col='Pclass',data=train)
plt.show()


# ** Observations**:
# - Port S has the largest number of boarded passengeers.
# - The chances for survival for Port C is highest around 0.55 while it is lowest for S. 
# - The survival chances are almost 1 for women for Pclass1 and Pclass2.
# - Male passengers boarded from port Q has a very low survival rate
# - Both men and women boarded from port S had no fortune if they belonged to Pcalss 3
# 

# ### Filling Embarked NaN
# 
# **Embarked** also contains NaN values, but only 2. We can just replace them with the port with largest passengeers boarded, a.k.a Port S.

# In[ ]:


for dataset in full_data:
    dataset['Embarked'].fillna('S',inplace=True)


# In[ ]:


train.Embarked.isnull().any()# Finally No NaN values


# ### SibSp
# This feature represents whether a person is alone or with his family members.  
# Sibling = brother, sister, stepbrother, stepsister  
# Spouse = husband, wife 

# In[ ]:


dft = pd.crosstab(train.SibSp,train.Survived,margins=True).rename(columns={0:'Dead',1:'Survived'})
dft.iloc[:-1,:].iplot(kind='bar',title='SibSp: Survived Vs Dead')


# In[ ]:


sns.factorplot('SibSp','Survived',data=train)
sns.factorplot('SibSp','Survived',hue='Sex',col='Pclass',data=train)
plt.show()


# **Observations**:
# - Most passengeers are traveling alone. They have around **34.5%** survival rate.
# - The survival rate is highest for those with 1 sibling or spous. The survival rate dereaces if the number of siblings increases. People save family but are not able to save a big family.
# - The largest family size of people in Pclass 1 and Pclass 2 are 4 while Pclass 3 people can have up to 8 siblings or spouses.
# - The suvival rate for famillies with 5-8 memebrs is **0%**. 

# ### Parch
# This feature is similar with SibSp, representing whether a person is alone or with his family members.  
# Par = parents 
# ch = children

# In[ ]:


dft = pd.crosstab(train.Parch,train.Survived,margins=True).rename(columns={0:'Dead',1:'Survived'})
dft.iloc[:-1,:].iplot(kind='bar',title='Parch: Survived Vs Dead')


# In[ ]:


sns.factorplot('Parch','Survived',data=train)
sns.factorplot('Parch','Survived',hue='Sex',col='Pclass',data=train)
plt.show()


# **Observations**:
# The results are similar to that of SibSp. 
# - Passengers with parents have greater chance of survival but the number reduces as the size of family goes up.
# - Being alone has lower survival rate compared with those in family of 2-4, but higher rate than those in large family.

# ### Fare

# In[ ]:


print('Highest Fare was:',train['Fare'].max())
print('Lowest Fare was:',train['Fare'].min())
print('Average Fare was:',train['Fare'].mean())


# In[ ]:


fare_pc1 = train[(train.Pclass==1)].Fare
fare_pc2 = train[(train.Pclass==2)].Fare
fare_pc3 = train[(train.Pclass==3)].Fare

fig = tools.make_subplots(rows=1, cols=3,subplot_titles=('Pclass 1', 'Pclass 2', 'Pclass 3'))

p1_fare = ff.create_distplot([fare_pc1], ['Pclass 1'], bin_size=30)
fig.append_trace(p1_fare.data[0], 1, 1)
fig.append_trace(p1_fare.data[1], 1, 1)

p2_fare = ff.create_distplot([fare_pc2], ['Pclass 2'], bin_size=5)
fig.append_trace(p2_fare.data[0], 1, 2)
fig.append_trace(p2_fare.data[1], 1, 2)

p3_fare = ff.create_distplot([fare_pc3], ['Pclass 3'], bin_size=5)
fig.append_trace(p3_fare.data[0], 1, 3)
fig.append_trace(p3_fare.data[1], 1, 3)
fig.layout.update({'showlegend':False})

iplot(fig)


# **Observations**：
# The highest fare in pclass 1 can reach to over 500 while highest fares in pcalss 2 and 3 are below 100

# In[ ]:


fare_survived = train[train.Survived==1].Fare
fare_dead = train[train.Survived==0].Fare

survived_fare = go.Histogram(
    name='Survived',
    x=fare_survived
)

dead_fare = go.Histogram(
    name='Dead',
    x=fare_dead
)
layout = go.Layout(title='Fare: Survived Vs Dead',barmode='stack')
fig = go.Figure(data=[survived_fare, dead_fare],layout=layout)
iplot(fig)


# **Observations**:
# - Most passengers are among the low-fare groups
# - The survival rate of the lowest fare group is quite low, while the rich men were more lucky. **Money** matters!

# Now let's have a conclusion of the above features. I directly used the nutshell in  [EDA To Prediction(DieTanic)](https://www.kaggle.com/ash316/eda-to-prediction-dietanic), cause the analysis results are pretty much the same

# ### Observations in a Nutshell for all features:
# **Sex:** The chance of survival for women is high as compared to men.
# 
# **Pclass:**There is a visible trend that being a **1st class passenger** gives you better chances of survival. The survival rate for **Pclass3 is very low**. For **women**, the chance of survival from **Pclass1** is almost 1 and is high too for those from **Pclass2**.   **Money Wins!!!**. 
# 
# **Age:** Children less than 5-10 years do have a high chance of survival. Passengers between age group 15 to 35 died a lot.
# 
# **Embarked:** This is a very interesting feature. **The chances of survival at C looks to be better than even though the majority of Pclass1 passengers got up at S.** Passengers at Q were all from **Pclass3**. 
# 
# **Parch+SibSp:** Having 1-2 siblings,spouse on board or 1-3 Parents shows a greater chance of probablity rather than being alone or having a large family travelling with you.

# Some feature engineering has been done during the feature exploration procedure (filling NaN values). The other feature engineering steps are concluded in the following nutshell. Thanks to Sina's well-thought work [Titanic Best Working Classfier](https://www.kaggle.com/sinakhorami/titanic/titanic-best-working-classifier)

# In[ ]:


# Feature engineering steps taken from Sina
# build fare band
for dataset in full_data:
    # Remove all NULLS in the Fare column and create a new feature CategoricalFare
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['Fare_Range']=pd.qcut(train['Fare'],4) # to check the value for dividing fare
for dataset in full_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare']  = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Create new feature FamilySize as a combination of SibSp and Parch
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    # Create new feature IsAlone from FamilySize
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
    # Build age band
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
    dataset['Age'] = dataset['Age'].astype(int)
    
    # Mapping character values to integers    
    dataset['Sex'].replace(['male','female'],[0,1],inplace=True)
    dataset['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
    dataset['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)


# ### Dropping UnNeeded Features
# 
# **Name**--> We don't need name feature as it cannot be converted into any categorical value.  
# **Ticket**--> It is any random string that cannot be categorised.  
# **Cabin**--> A lot of NaN values and also many passengers have multiple cabins. So this is a useless feature.  
# **PassengerId**--> It is not meaningful.  

# In[ ]:


# Feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin']
for dataset in full_data:
    dataset.drop(drop_elements,axis=1,inplace=True)
train.drop(['Fare_Range'],axis=1,inplace=True)


# Let's have a look at the new **FamilySize**, **Fare** and **Age** features.

# In[ ]:


sns.factorplot('Age','Survived',data=train,col='Pclass')
plt.show()


# The survived rate decreases with the increase of **Age** irrespective of pclass.

# In[ ]:


sns.factorplot('Fare','Survived',data=train,hue='Sex')
plt.show()


# The survived rate increases with the increase of **Fare** irrespective of sex.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,6))
sns.factorplot('FamilySize','Survived',data=train,ax=ax[0])
ax[0].set_title('FamilySize vs Survived')
sns.factorplot('IsAlone','Survived',data=train,ax=ax[1])
ax[1].set_title('IsAlone vs Survived')
plt.close(2)
plt.close(3)
plt.show()


# Being alone and being with a large family relate to low survival rate.

# ## Correlation Between The Features

# In[ ]:


sns.heatmap(train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# ### Interpreting The Heatmap
# 
# **POSITIVE CORRELATION:** If an **increase in feature A leads to increase in feature B, then they are positively correlated**. A value **1 means perfect positive correlation**.
# 
# **NEGATIVE CORRELATION:** If an **increase in feature A leads to decrease in feature B, then they are negatively correlated**. A value **-1 means perfect negative correlation**.
# 
# If the correlation between two features approcah **1** or **-1**, they are carrying highly similar information. Investigatin the table that **FamilySize** and **SibSp** have the highest correlation (0.89), the second highest is **FamilySize** and **Parch** (0.78). This makes sense cause **FamilySize** is the sum of **SibSip** and **Parch**. As we find the correlation between **SibSip** and **Parch** is only 0.41, I decide to remove **SibSp** and keep **Parch**

# In[ ]:


for dataset in full_data:
    dataset.drop(['SibSp'],axis=1,inplace=True)


# ## Part2: Ensemble Modelling
# 
# ##  Preditive Modelling
# 
# Now it's time to build machine learnng model to do predictive job. **sklearn** is a mature python package hosting various machine leanring models, like, **Logistic Regression**, **Support Vector Machines**, **Random Forest**,  **K-Nearest Neighbours**, **Naive Bayes**, **AdaBoost Classifier**, **Neural Network**, etc.

# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Gaussian Naive Bayes
from sklearn.svm import SVC #support vector classifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
import xgboost as xgb  # xgboost is not a classifier in sklean, therefore needs better attention


# No test data should be seen during the training procedure. The train data can be split into training and validation set so that the model trained on training set can be tested on valication set.

# In[ ]:


SEED = 0 # for reproducibility
## Prepares training and testing input and target data
training,val=train_test_split(train,test_size=0.3,random_state=SEED,stratify=train['Survived'])
Xtrain=training[training.columns[1:]]
ytrain=training[training.columns[:1]]
Xval=val[val.columns[1:]]
yval=val[val.columns[:1]]

# Some useful parameters which will come in handy later on
ntrain = Xtrain.shape[0]
nval = Xval.shape[0]


# We first train some base models on the data.

# In[ ]:


# build a set of base learners
def base_learners():
    """Construct a list of base learners"""
    lr = LogisticRegression(random_state=SEED)
    svc = SVC(kernel='linear', C=0.1, gamma=0.1, probability=True,random_state=SEED)
    knn = KNeighborsClassifier()
    nb = GaussianNB()
    nn = MLPClassifier((80,10),random_state=SEED)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    et = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    ab = AdaBoostClassifier(n_estimators=100, random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    
    models = {
        'Logistic Regression': lr,
        'SVM': svc,
        'KNN': knn,
        'Naive Bayes': nb,
        'Neural Network': nn,
        'Random Forest': rf,
        'Extra Trees': et,
        'AdaBoosting': ab,
        'GradientBoosting': gb
    }
    
    return models

def train_predict(models, Xtrain, ytrain, Xtest):
    """Fit models"""
    P = np.zeros((nval,len(models)))
    P = pd.DataFrame(P)
    
    print("Fitting models")
    for i, (name, model) in enumerate(models.items()):
        print("%s..."%name, end=' ', flush=False)
        model.fit(Xtrain,ytrain)
        P.iloc[:,i]=model.predict(Xtest)
        print('done')
    print('Fitting done!')   
    P.columns = models.keys()

    return P 

def score_models(P, y):
    "Obtain accuracy scores of models"
    acc = []
    print("Scoring models")
    for m in P.columns:
        acc.append(metrics.accuracy_score(y, P.loc[:,m]))
    print('Done!')
    
    acc = pd.Series(data=acc,index=P.columns,name='Accuracy')
    return acc
    
models = base_learners()
P = train_predict(models, Xtrain, ytrain, Xval)
acc = score_models(P, yval) 


# In[ ]:


iplot(ff.create_table(acc.reset_index()))


# Among the baseline models, GradientBoosting achieves the best prediction accuracy.

# ### Cross Validation
# In the above training procedure, models are trained on training set and tested on validation set once. Usually we need to do the training and validation many times to determine the robustness of the classifier. Furthermore, we want to train on different training set and validate on different validation set, so as to improve the generalization of the classifier.   **Cross Validation** is a good way to achieve this objective. The other advantage is that we can fully use the training set when its size is not large enough.
# - The K-Fold Cross Validationfirst divides the dataset into k-subsets.
# - We reserve 1 part for validation and train the algorithm over the other (k-1) parts.
# - The process is run k times by chaning the validation subset. Then the final score is the average accuracy of k runs.

# In[ ]:


from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction

SEED = 0 # for reproducibility
Xtrain = train[train.columns[1:]]
ytrain = train[train.columns[:1]]
Xtest = test
kf = KFold(n_splits=5, random_state=SEED) # k=5, split the data into 5 equal part

def trainCV(models, X, y):
    """Fit models using cross validation"""   
    
    print("Fitting models")
    acc, std = [], []
    acc_all = []
    for i, (name, model) in enumerate(models.items()):
        print("%s..."%name, end=' ', flush=False)
        cv_result = cross_val_score(model,X,y,cv=kf,scoring='accuracy')
        acc.append(cv_result.mean())
        std.append(cv_result.std())
        acc_all.append(cv_result)
        print('done')
    print('Fitting done!') 
    acc=pd.DataFrame({'CV Mean':acc,'Std':std},index=models.keys())
    acc_all= pd.DataFrame(acc_all,index=models.keys()).T
    return acc, acc_all
    
models = base_learners()
acc, acc_all = trainCV(models, Xtrain, ytrain)
iplot(ff.create_table(acc.reset_index()))


# In[ ]:


acc_all.iplot(kind='box')


# ### Hyperparameter Tuning
# We use default values for most hyperparamteres in the above learneres. Hyperparamters are paramters preset by human and not updated during training procedure. We can tune them to improve the learning performance of the learners.
# 
# As **Adaboosting** and **GradientBoostinging** achieve the best accuracis, I show its tuning procedure below.  Different values are set for each parameters and **GridSearchCV** is taken advantage of to do the parameter selection. The more granular values are tested for one parameter, the higher chance we find the best parameters, but the training cost is higher

# **AdaBoosting**
# 
# The most important hyperparameters inAdaBoosting is **n_estimators**.

# In[ ]:


from sklearn.model_selection import GridSearchCV
n_estimators=list(range(100,1100,100))
learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyper={'n_estimators':n_estimators,'learning_rate':learn_rate}
gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)
gd.fit(Xtrain,ytrain)
print(gd.best_score_)
print(gd.best_estimator_)


# The best score of AdaBoosting achieves ** 83.16%** with **n_estimators=200, learning_rate=0.05**.

# **GradientBoosting**
# 
# There are many hyperparameters in GradientBoosting, among which **n_estimators**, **max_depth**, **min_samples_leaf** are the most important.

# In[ ]:


n_estimators=[100,200,300,400,500,600]
max_depth=[2,3,4,5,6]
min_samples_leaf=[1,2,3]
hype_param={'n_estimators':n_estimators,'max_depth':max_depth,'min_samples_leaf':min_samples_leaf}
grid=GridSearchCV(estimator=GradientBoostingClassifier(random_state=SEED),param_grid=hype_param,verbose=True)
grid.fit(Xtrain,ytrain)
print(grid.best_score_)
print(grid.best_estimator_)


# The best score for GradientBoosting is **82.72% with n_estimators=100, max_depth=2, and min_sample_split=2**.

# ## Ensembling
# The above section is the first taste of using machine learning models to do prediction. Now let's dive deep into the ensemble method to build a powerful learner. Ensembling can be done in following ways:
# 1. Voting classifier (Average regressor)
# 2. Bagging
# 3. Boosting
# 4. Stacking

# Let's copy and paste the code of building base_learners below for better coherence.

# In[ ]:


# build a set of base learners
def base_learners():
    """Construct a list of base learners"""
    lr = LogisticRegression(random_state=SEED)
    svc = SVC(kernel='linear', C=0.1, gamma=0.1, probability=True,random_state=SEED)
    knn = KNeighborsClassifier()
    nb = GaussianNB()
    nn = MLPClassifier((80,10),random_state=SEED)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    et = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    ab = AdaBoostClassifier(n_estimators=100, random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    
    models = {
        'Logistic Regression': lr,
        'SVM': svc,
        'KNN': knn,
        'Naive Bayes': nb,
        'Neural Network': nn,
        'Random Forest': rf,
        'Extra Trees': et,
        'AdaBoosting': ab,
        'GradientBoosting': gb
    }
    
    return models


# ### Voting Classifier
# It is the simplest way of ensembling, giving an average prediction result based on the prediction of all the submodels.

# In[ ]:


from sklearn.ensemble import VotingClassifier
models = base_learners()
ensemble_voting=VotingClassifier(estimators=list(zip(models.keys(),models.values())), 
                       voting='soft')
scores=cross_val_score(ensemble_voting,Xtrain,ytrain, cv = 10,scoring = "accuracy")
print('The cross validated score is',scores.mean())


# ### Bagging
# 
# Bagging is a general ensemble method. It works by applying similar classifiers on small partitions of the dataset and then taking the average of all the predictions. Due to the averaging, there is reduction in variance. Unlike Voting Classifier, Bagging makes use of similar classifiers. Actually, random forest is a bagging method of decision tree. **sklearn** provides a **BaggingClassifier** to wrap various base learners.

# #### Bagged KNN
# 
# Bagging works best with models with high variance. We can use KNN with small value of **n_neighbours**, as small value of *n_neighbours*.

# In[ ]:


from sklearn.ensemble import BaggingClassifier
model =BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),random_state=0,n_estimators=700)
scores=cross_val_score(model,Xtrain,ytrain,cv=10,scoring='accuracy')
print('The cross validated score for bagged KNN is:',scores.mean())


# The bagging method greatly improves the leanring power compared with using single KNN.

# ### Boosting
# 
# Boosting is an ensembling technique which uses sequential learning of classifiers. It is a step by step enhancement of a weak model.Boosting works as follows:
# 
# A model is first trained on the complete dataset. Now the model will get some instances right while some wrong. Now in the next iteration, the learner will focus more on the wrongly predicted instances or give more weight to it. Thus it will try to predict the wrong instance correctly. Now this iterative process continous, and new classifers are added to the model until the limit is reached on the accuracy.
# 
# Actually, we have seen two powerful boosting methods, namely **AdaBoost** and **GradientBoost**. They are both boosting methods built over decision trees. 

# #### xgboost
# Now let's have a look at another popular decision tree-based boosting method, XGBoost. It was built to optimize large-scale boosted tree algorithms. 

# In[ ]:


xgboost=xgb.XGBClassifier(n_estimators=900,learning_rate=0.1)
scores=cross_val_score(xgboost,Xtrain,ytrain,cv=10,scoring='accuracy')
print('The cross validated score for XGBoost is:',scores.mean())


# ### Stacking
# Stacking is combine the prediction results of base learners and use as input of secondary learner (meta-learner) which gives out the final predicition results. It's more complicated than the above three emsemble methods, containing the following steps:
# 1. Define the base learners
# 2. Define a meta learner
# 3. Train the base learners
# 4. Generate prediction results of base learners
# 5. Train the meta learner
# 6. Obtain the final prediction results

# #### 1. Define the base learners

# In[ ]:


base_models = base_learners()


# #### 2. Define a meta learner
# There is no general consensus what model to be used as meta learner. It can be a linear model, a kernel model, a tree-based model or even another ensemble method in which case two emsmeble layer are used in the model framework. In this kernel  **xgboost** is used as the meta learner.

# In[ ]:


meta_learner = xgb.XGBClassifier(n_estimators= 2000,
                                 max_depth= 4,
                                 min_child_weight= 2,
                                 gamma=0.9,                        
                                 subsample=0.8,
                                 colsample_bytree=0.8,
                                 objective= 'binary:logistic',
                                 nthread= -1,
                                 scale_pos_weight=1)


# #### 3. Train base learners

# In[ ]:


## Put the training and testing data here for better coherence
Xtrain = train[train.columns[1:]]
ytrain = train[train.columns[:1]]
Xtest = test

def train_base_learners(models, X, y):
    """Fit base models"""

    print("Fitting models")
    for i, (name, model) in enumerate(models.items()):
        print("%s..."%name, end=' ', flush=False)
        model.fit(X,y)
        print('done')
    print('Fitting done!')   


# #### 4. Generate prediction results of base learners
# 
# We do not use the predicted class label of each base learner but the predicted probability of being class 1. It is more reasonable to use probabilities as input of meta classifier.

# In[ ]:


def pred_base_learners(models, X):
    "Generate a prediction matrix"
    P = np.zeros((X.shape[0],len(models)))
    P = pd.DataFrame(P)
    
    print("Generating base learner predictions")
    for i, (name, m) in enumerate(models.items()):
        print("%s..."%name, end=' ', flush=False)
        P.iloc[:,i] = m.predict_proba(X)[:,1]
        print('done')
        
    P.columns = models.keys()
    
    return P


# #### Correlation between base learners
# The emsemble method achieves the greatest power the base learners are diverse. 

# In[ ]:


train_base_learners(base_models, Xtrain,ytrain)
P = pred_base_learners(base_models, Xtest)

sns.heatmap(P.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# In[ ]:


P_label = P.apply(lambda x: 1*(x>=0.5))
sns.heatmap(P_label.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# The corrleation between the predicted probabilities of base learners are pretty high except for AdaBoosting. This greatly limits the advantage of ensembling. However, the correplation between predicted labels of base learners are mostely in the range of 0.5-0.8, we still expect we can achieve a little perfomence improvement using ensembling. This kernel will not attemp to select base learners to guarantee the diversity of base models, but it's indeed a plausible direction to improve classification accuracy.

# #### Feature Importance
# **Sklearn** package is very powerful that it can return the feature importance for some learners, like **Random Forest**, **ExtraTrees**, **AdaBoost**, **GradientBoosting**, etc.

# In[ ]:


feature_importance = pd.DataFrame({'Random Forest': base_models['Random Forest'].feature_importances_,
                                   'Extra Trees': base_models['Extra Trees'].feature_importances_,
                                   'GradientBoosting': base_models['GradientBoosting'].feature_importances_,
                                   'AdaBoosting': base_models['AdaBoosting'].feature_importances_},
                                  index=Xtrain.columns.values)
display(feature_importance)
fig = tools.make_subplots(rows=2, cols=2,subplot_titles=(
                        'AdaBoosting Feature Importance', 
                        'Extra Trees Feature Importance', 
                        'GradientBoosting Feature Importance',
                        'Random Forest Feature Importance'))

bars = feature_importance.iplot(kind='barh',subplots=True,asFigure=True)
fig.append_trace(bars.data[0], 1, 1)
fig.append_trace(bars.data[1], 1, 2)
fig.append_trace(bars.data[2], 2, 1)
fig.append_trace(bars.data[3], 2, 2)
fig.layout.update({'showlegend':False})
iplot(fig)


# It can be seen that features play differnt roles in different learners. For example, **Parch** is the most important feature for **AdaBoosting**, **Sex** for **Extra Trees**, **Initial** for **GradientBoosting**, and **Initial** for **Random Forest**. We can also conlucde that the feature **IsAlone** is not very meaningful cause its importance is very small in all models. In this kernel, no features will be removed at current stage. But it is worth exploring if improving accuracy is the key objective.

# #### 5. Train the meta learner
# 
# Let's come back to building stacking ensemble model. As we use cross validation to train base learners, we must be careful when feeding output of base learners to the meta learner. Always remember that no test data should be seen during training, either in training base models or meta model.

# In[ ]:


from sklearn.base import clone

## Some useful parameters
ntrain = train.shape[0]
ntest  = test.shape[0]
kf = KFold(n_splits=5, random_state=SEED) # k=5, split the data into 10 equal part

def stacking(base_learners, meta_learners, X, y):
    """Training routine for stacking"""
    print('Fitting final base learners...')
    train_base_learners(base_learners, X, y)
    print('done')
    
    ## geneerate input of meta learner
    print('Generate cross-validated predictions...')
    cv_pred, cv_y = [], []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        fold_Xtrain, fold_ytrain = X[train_index,:], y[train_index]
        fold_Xtest,  fold_ytest = X[test_index,:], y[test_index]
        
        # inner loops: step 3 and 4
        fold_base_learners = {name:clone(model) for name, model in base_learners.items()}
        
        train_base_learners(fold_base_learners, fold_Xtrain, fold_ytrain)
        fold_P_base = pred_base_learners(fold_base_learners, fold_Xtest)
        
        cv_pred.append(fold_P_base)
        cv_y.append(fold_ytest)
    print('CV prediction done')
    
    cv_pred = np.vstack(cv_pred)
    cv_y = np.vstack(cv_y)
    
    print('Fitting meta learner...', end='')
    meta_learner.fit(cv_pred, cv_y)
    print('done')
    
    return base_learners, meta_learner


# #### 6. Obtain the final prediction results

# In[ ]:


def ensemble_predict(base_learners,meta_learner,X):
    """Generate prediction from ensemble"""
    
    P_base = pred_base_learners(base_learners, X)
    return P_base.values, meta_learner.predict(P_base.values)


# In[ ]:


cv_base_learners, cv_meta_learner= stacking(base_learners(), meta_learner,Xtrain.values, ytrain.values)
P_base, P_ensemble = ensemble_predict(cv_base_learners, meta_learner, Xtest)


# You must find the stacking ensemble is complicated and not easy to handle. Lucky you! There are many existing ensemble packages help us do the stacking in a easy- peasy manner. Just call the wrapper and add baselearners and meta learners as layers. The idea of tenforflow, hah! Look at the simple code below.

# In[ ]:


p_ens[:10]


# In[ ]:


p_ens1[:10]


# In[ ]:


from mlens.ensemble import SuperLearner

val_train, val_test = train_test_split(train,test_size=0.3,random_state=SEED,stratify=train['Survived'])
val_Xtrain=val_train[val_train.columns[1:]]
val_ytrain=val_train[val_train.columns[:1]]
val_Xtest=val[val_test.columns[1:]]
val_ytest=val[val_test.columns[:1]]
# Instantiate the ensemble with 10 folds
super_learner = SuperLearner(folds=10,random_state=SEED,verbose=2,backend='multiprocessing')
# Add the base learners and the meta learner
super_learner.add(list(base_learners().values()),proba=True)
super_learner.add_meta(LogisticRegression(), proba=True)

# Train the ensemble
super_learner.fit(val_Xtrain,val_ytrain)
# predict the test set
p_ens = super_learner.predict(val_Xtest)[:,1]
p_ens_label = 1*(p_ens>=0.5)
print('The acccuracy of super learner:',metrics.accuracy_score(p_ens_label, val_ytest))


# ### Producing the Submission file
# 
# Finally having trained and fit the base and meta learners, we can now output the predictions into the proper format for submission to the Titanic competition as follows:

# In[ ]:


# Generate Submission File 
Submission = pd.DataFrame({ 'PassengerId': PassengerId_test,
                            'Survived': P_ensemble })
Submission.to_csv("Submission.csv", index=False)


# ## Conclusions
# This script demonsrates some visualization methods to explore features containing in a dataset and discusses how we should engineer and select features. Then, some basic machine learning models and ensemble methods are discussed. Large portion is given to stacking ensemble which generates the final submission results of the Titanic competition.
# There are many directions that the method can be improved to advance the classification accuracy.
# - Design other features manually.
# - Carefully compare the correlation of features and select the most appropriate features.
# - Carefully fine tune the parameters of base learners.
# - Select the appropriate combinations of base learners.
# - Try other meta learner and fine tune the hyperparameters.
# - etc.

# In[ ]:




