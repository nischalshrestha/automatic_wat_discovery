#!/usr/bin/env python
# coding: utf-8

# # Titanic Survivors: What Can We Learn from History?
# 
# 
# This notebook is going to describe data analysis and predictive modeling of famous Titanic survivors dataset from my perspective. The goal here is to be concise and informative. So without any further due, let's start:
#  

# In[ ]:


# Import General Libraries 
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# Loading Data-Set
path1 = '../input/train.csv'
path2 = '../input/test.csv'
path3 = '../input/genderclassmodel.csv'

train = pd.read_csv(path1)
test = pd.read_csv(path2)
test_label = pd.read_csv(path3)


# ## 1. Data Inspection

# In[ ]:


# Data-Set Priliminery Review 

print(train.info())
print('------------------------')
print(train.isnull().sum())
print('------------------------')
print(train.describe())

print('########################')

#print(test.info())
#print('------------------------')
#print(test.isnull().sum())
#print('------------------------')
#print(test.describe())


# It is also shown that 'Cabin', 'Age' and 'Embarked' columns have missing values. More than 75% of Cabin data is missing, which makes it impossible to replace them with statistically meaningful values. Therefore this column will not play any role in this analysis. Missing values in other two columns ('Age' & 'Embarked') will be replaced by appropriate values in next section.
# 
# 
# “describe” method reveals that 'Fare' columns contains zero value (min), which doesn't seem right and needs to be addressed.

# ## 2. Exploratory Data Analysis (EDA)
# 
# ### 2.1 Gender
# 
# Passenger’s gender data is presented in 'Sex' column. Fortunately this column doesn't have any missing value. So there is no need for specific data preparation and all we need is to conduct some visual EDA. 

# In[ ]:


# Gender visual EDA (Figure 1)
get_ipython().magic(u'matplotlib inline')
plt.figure(figsize=(5,5))

men = train['Sex'][train['Sex']=='male'].count()
women = train['Sex'][train['Sex']=='female'].count()

slices = [men,women]; labels = ['Men','Women']; colors = ['b','m']

plt.pie(slices,labels = labels, colors = colors, startangle = 90, autopct = '%1.1f%%',explode = [0.01,0])
plt.title('Fig.1 : Passenger Gender Ratio',fontweight="bold", size=12)
plt.show()


# In[ ]:


# Gender visual EDA (Figure 2)
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
men_survived = train['Sex'][(train['Sex']=='male')&(train['Survived']==1)].count()
men_not_survived = men - men_survived
slices = [men_survived,men_not_survived];labels = ['Survived','Not Survived'];colors = ['b','lightgray']

plt.pie(slices,labels = labels, colors = colors, startangle = 90, autopct = '%1.1f%%',explode = [0.01,0])
plt.title('Men')

plt.subplot(1,2,2)
women_survived = train['Sex'][(train['Sex']=='female')&(train['Survived']==1)].count()
women_not_survived = women - women_survived
slices = [women_survived,women_not_survived];labels = ['Survived','Not Survived'];colors = ['m','lightgray']

plt.pie(slices,labels = labels, colors = colors, startangle = 90, autopct = '%1.1f%%',explode = [0.01,0])
plt.title('Women')
plt.suptitle('Fig.2 : Survaival Rate by Gender',fontweight="bold", size=12)
plt.subplots_adjust(top=0.75)

plt.show()
# Gender visual EDA (Figure 3)
plt.figure(figsize=(5,5))

slices = [men_survived,men_not_survived,women_survived,women_not_survived]
labels = ['Men - Survived','Men - Not Survived','Women - Survived','Women - Not Survived']
colors = ['b','lightgray','m','lightgray']
plt.pie(slices,labels = labels, colors = colors, startangle = 90, autopct = '%1.1f%%',explode = [0.02,0,0.02,0])
plt.title('Fig.3 : Survival Ratio devided by Gender',fontweight="bold", size=12)
plt.show()


# Figure 1 shows men and women share 65% and 35% of ship manifest. Figure 2 illustrates survival rate by gender. It is evident that while 3/4 of all women onboard are survived the odd for their male counterpart was less than 20%. Transposing these probabilities to total number of passenger, leads to data visualization shown in figure 3. Comparing figure 3 to figure 1 gives better understanding of survivors divided by gender. It is also worth mentioning that this analysis is only conducted on train dataset, so actual numbers may vary. As mentioned earlier, 'Sex' column type is categorical. In order to use this type of data in machine learning, it has to be converted to numerical value (next cell).

# In[ ]:


# Sex is another object column in the dataset, which needs to be converted to categorial type

train['Sex']=train['Sex'].astype('category')
train['Sex_cat']=train['Sex'].cat.codes

test['Sex']=test['Sex'].astype('category')
test['Sex_cat']=test['Sex'].cat.codes


# ### 2.2 Age
# 
# Data in 'Age' column suffers from large number of missing values, which requires serious attention in terms of preprocessing and replacement. 
# 
# First step; visual EDA on the available data in train dataset:

# In[ ]:


# Visual EDA on Age column
train['Age'].dropna().hist(bins=8, color='m',alpha=0.5,label='Onboard') # All passengers onboard 
train['Age'][train['Survived']==1].dropna().hist(bins=8, color='b',alpha=0.75,label='Survived') # Survived passengers

plt.xlabel('Age'); plt.ylabel('Number of Passenger')
plt.title('Fig.4 : Passengers Age Distribution',fontweight="bold", size=12)

plt.legend()
plt.tight_layout()


# Figure 4 shows that the largest passenger populations are in their 20's (20-30), followed by 30-40 and 10-20. Same trend could be observed for survived passengers.
# 
# In preliminary data inspection, the large number of missing value is found in the 'Age' column. The common practice for filling out missing values is to replace them with statistical significant parameters like mean, median or mode. However, when sampling data is not normally distributed, simple implementation of these methods might be misleading. For instance in current situation it is quite possible that the age distribution alters between different classes and genders. To examine this speculation we construct “FacetGrid” histogram for different genders and classes as follow:
#    

# In[ ]:


# Age distibution based on class and sex
fig = sns.FacetGrid(train, row='Pclass', col='Sex', size=2.0, aspect=2.0)
fig.map(plt.hist, 'Age', alpha=.5, bins=10, color = 'darkslateblue')
plt.subplots_adjust(top=0.85)

fig.fig.suptitle('Fig.5 : Age Distribution based on Gender & Class',fontweight="bold", size=12)
fig.add_legend()
plt.show()


# Figure 5 clearly depicts different age distribution for different group categorized by class and gender. In the next session we try to extract mean value for each of these groups and replace missing values in the dataset according to their gender and class.

# In[ ]:


# Claculating mean for 'Age' column based on gender and class
# Neat impelementation of this session is inspired by nice work of "Manav Sehgal". 
# You can find his original notebook at following URL:
# https://www.kaggle.com/startupsci/titanic-data-science-solutions

age_train = np.zeros((2,3))
age_test = np.zeros((2,3))

for i in range(0,2):
    for j in range(0,3):
        age_train[i,j] = train['Age'][(train['Sex_cat'] == i) & (train['Pclass'] == j+1)].mean()
        age_test[i,j] = test['Age'][(test['Sex_cat'] == i) & (test['Pclass'] == j+1)].mean()

for i in range(0,2):
    for j in range(0,3):
        train.loc[(train['Age'].isnull())&(train['Sex_cat'] == i)&(train['Pclass'] == j+1),'Age'] = age_train[i,j] 
        test.loc[(test['Age'].isnull())&(test['Sex_cat'] == i)&(test['Pclass'] == j+1),'Age'] = age_test[i,j]   


# Although the above method is a descent approach to this type of problems, but in current situation the average values for different groups are not varied significantly (women: [34,28,21], men: [41,30,26] in the order of “Pclass”). So one should take cost & benefit analysis into account before implementing for-loop construction, which is not the most efficient way to replace the data specially when dealing with big data.
# 
# Below, you can find the alternative method to replace “Age” missing values. Here, estimated values are randomly assigned in a range of sigma around the mean.
# 
# Finally, since the age data varies in a broad range of 0-80, it is highly recommended to scale them into the similar range of other features. (See section 4)

# In[ ]:


# Alternative method (common practice)       

#ave_age_train = train['Age'].mean()
#std_age_train = train['Age'].std()
#ave_age_test = test['Age'].mean()
#std_age_test = test['Age'].std()
        
#random.seed(42)
#train['Age']=train['Age'].fillna(ave_age_train + random.uniform(-1,1) * std_age_train)
#test['Age']=test['Age'].fillna(ave_age_test + random.uniform(-1,1) * std_age_test)


# ### 2.3 Fare
# There is one missing value in 'Fare' column of test dataset. Preliminary review also revealed some rows in both training and test datasets have zero values. All these values are required to be replaced by proper value. But first lets conduct visual EDA to get better insight from the data:

# In[ ]:


# Visualization of survival rate based on fare
bins = np.arange(0, 550, 10)
index = np.arange(54)
train['fare_bin'] = pd.cut(train.Fare,bins,right=False)
total_bin = train.groupby(['fare_bin']).size().values
survived_bin = train[train['Survived']==1].groupby(['fare_bin']).size().values

np.seterr(divide='ignore', invalid='ignore') # ignoring "divide by zero" or "divide by NaN"
survived_fare = survived_bin*100/total_bin
###############
fig, ax = plt.subplots(1,1,figsize=(10,6))
colormap=plt.cm.get_cmap('jet')
ax.scatter(index*10,survived_fare,marker='o', edgecolor='black', c=survived_fare**0.2,cmap=colormap,alpha=0.75,s = 7*total_bin )
plt.xlabel('Fare'); plt.ylabel('Survival Rate (%)')
plt.title('Fig.6 : Survival Rate vs. Fare',fontweight="bold", size=12)
plt.xticks([10,50,100,150,200,250,300,350,400,450,500,550])
plt.xlim([-50,550]);plt.ylim([0,110])  
plt.tight_layout()
plt.show()


# Figure 6, illustrates survival rate vs. fare in training dataset. The marker sizes are proportional to the population of each fare range. It is quite evident that the largest population belongs to cheapest fare range with lowest survival rate. The next three-biggest groups are also among cheapest fare price and relatively minimum survival rate. General trend suggests that the survival rate improves as fare increases, although it is not monotonic.
# 
# ## So As Always: Heads, the rich win, Tails, the poor lose !!!
# 

# In[ ]:


# There is one missing value in Fare column of test dataset, which is required to 
# be replaced by proper value. Preliminary review also revealed some rows in both training
# and test datasets have zero values. Therfore:

train['Fare'].replace('0',None,inplace=True)
test['Fare'].replace('0',None,inplace=True)

train_fare_trans = train['Fare'].groupby(train['Pclass'])
test_fare_trans = test['Fare'].groupby(test['Pclass'])

f = lambda x : x.fillna(x.mean())
train['Fare'] = train_fare_trans.transform(f)
test['Fare'] = test_fare_trans.transform(f)
###############
# similar to age, fare values are also needed to be scaled
###############
train = train.drop(['fare_bin'],axis=1) # 'fare_bin' only generatad for visualization purposes


# ### 2.4 Port of Embarkation
# Missing data regarding port of embarkation is replaced by most frequent value. This is a categorical feature, which has to be converted to numerical value.

# In[ ]:


# Missing data regarding port of embarkation is replaced by most frequent value
mode_emb_train = train['Embarked'].mode()
train['Embarked']=train['Embarked'].fillna("S")
train['Embarked']=train['Embarked'].astype('category')
train['Embarked_cat']=train['Embarked'].cat.codes
###############
mode_emb_test = test['Embarked'].mode()
test['Embarked']=test['Embarked'].fillna("S")
test['Embarked']=test['Embarked'].astype('category')
test['Embarked_cat']=test['Embarked'].cat.codes


# # 3. Feature Engineering 
# 
# Since this dataset has limited number of feature to play with, there is no choice but to use all of them. So it is only need to pick relevant columns and leave the rest out.

# In[ ]:


# Dropping unnecessary columns
X_train = train.drop(['Survived','Sex','Embarked','PassengerId','Name','Ticket','Cabin'],axis=1)
Y_train = train['Survived']
###############
X_test  = test.drop(['Sex','Embarked','PassengerId','Name','Ticket','Cabin'],axis=1)
Y_test = test_label['Survived']


# Before using the data in ML model, it is better to conduct a sanity check by calculating pair correlation coefficient between all designated features. That way, feeding degenerate or dependent data to ML model can be avoided.

# In[ ]:


# Making correlation coefficients pair plot of all feature in order to identify degenrate features
ax = plt.axes()
sns.heatmap(X_train.corr(), vmax=1.0,vmin=-1.0, square=True, annot=True, cmap='Blues',linecolor="white", linewidths=0.01, ax=ax)
ax.set_title('Fig.7 : Correlation Coefficient Pair Plot',fontweight="bold", size=12)
plt.show()


# Fortunately, there is no evidence of strong correlation between any two features so it is safe to move on and proceed to the Machine Learning section.

# # 4. Machine Learning and Model Development
# 
# The ultimate goal of this work is to utilize Machine Learning (ML) techniques in order to construct a reliable predictive model based on the current dataset. Among all available algorithms; Logistic Regression, K Nearest Neighbors (KNN) and Random Forest are chosen to address this classification problem, due to their simplicity and yet their effectiveness. Systematic approach is taken by constructing ML pipeline for model development. Pipeline provides a framework for applying step-by-step transformation to feeding dataset and a final estimator. It also enables cross-validation of effective parameters in each step to identify optimum parameters for final model.
# 
# For instance, one can use pipeline to impute missing data and scale the dataset at once. Since in current case, data preprocessing is already done, only data scaling is passed through the pipeline. The main reason for rescaling the data is that most of the ML algorithms use some form of distance calculation therefore, features on larger scales can influence the model. Thus, it is desired to have features on a similar scale. The normalizing process is usually done by scaling and centering.
# 

# In[ ]:


# Import ML Libraries 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# 
# ### 4.1 Logistic Regression
# 
# 

# In[ ]:


# Logistic Regression
steps = [('scaler', StandardScaler()),('logreg', LogisticRegression())]
pipeline = Pipeline(steps)
pipeline.fit(X_train, Y_train)
Y_pred1 = pipeline.predict(X_test)
print('Logistic Regression')
print('==========================')
print('Test score:',pipeline.score(X_test, Y_test))
print('==========================')
print('Final Report on Prediction Result')
print(classification_report(Y_test, Y_pred1))


# As mentioned earlier, cross validation (GridSearchCV) of all involved parameters enables unique possibility to construct a grid from all combinations of parameters, tries each combination, and then reports back the best combination/model. This can be done for every aspect of each step. Even one can use that to explore the optimum number of features for the model. In two following sessions, GridSearchCV is utilized to identify optimum number of neighbors and estimators for KNN and Random Forest algorithms in order to prevent over-fitting or under-fitting during the model development.

# 
# ### 4.2 K Nearest Neighbors
# 
# 

# In[ ]:


# KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore') # Updated version of GridSearchCV is not available
steps = [('scaler', StandardScaler()),('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
parameters = {'knn__n_neighbors':np.arange(1, 100)}
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, Y_train)
Y_pred2 = cv.predict(X_test)
print('K Neighbors Classifier')
print('==========================')
d = cv.best_params_
print('Optimum Number of Neighbors:',d.get('knn__n_neighbors'))
print('Test score:', cv.score(X_test, Y_test))
print('==========================')
print('Final Report on Prediction Result')
print(classification_report(Y_test, Y_pred2))


# 
# ### 4.3 Random Forest
# 
# 

# In[ ]:


# RandomForestClassifier
steps = [('scaler', StandardScaler()),('randfor', RandomForestClassifier())]
pipeline = Pipeline(steps)

parameters = {'randfor__n_estimators':np.arange(1, 100)}
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, Y_train)
Y_pred3 = cv.predict(X_test)

print('Random Forest Classifier')
print('==========================')
d = cv.best_params_
print('Optimum Number of Estimators:',d.get('randfor__n_estimators'))
print('Test score:', cv.score(X_test, Y_test))
print('==========================')
print('Final Report on Prediction Result')
print(classification_report(Y_test, Y_pred3))


# 
# 
# ## --------------------------------------------------------------------------------------------
# 
# To draw a conclusion about these models performances, all classification metrics are reported from confusion matrix. Evidently, Logistic Regression shows highest scores on all metrics and shall be used for final prediction.
# 
# ## Take Away Massage: Keep It Simple !
# ## --------------------------------------------------------------------------------------------
# 

# # 5. Final Submission

# In[ ]:


# Final submission

submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': Y_pred1 })
submission.to_csv('titanic.csv', index=False)

