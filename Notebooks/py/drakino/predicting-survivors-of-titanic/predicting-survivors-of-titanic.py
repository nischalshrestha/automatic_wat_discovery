#!/usr/bin/env python
# coding: utf-8

# # 1. Collecting Data
# First and arguably the most time consuming step of data science or statistics is collecting the data. The data should be gathered via data wrangling methods such as scraping. Then, the gathered data generally requires a cleaning strategy for removing corrupted or duplicated information. 
# 
# Luckily, this time consuming step was already conducted and the data is available in a csv file. Thank you for dealing with this step and providing the data!
# 
# ## Reading csv File into a DataFrame

# In[59]:


import pandas as pd 

trainFilePath = '../input/train.csv'
testFilePath  = '../input/test.csv'
trainDf = pd.read_csv(trainFilePath)


# # 2. Summarize Data (Exploratory Data Analysis)

# The second step is summarizing the available data by examining the distributions and relationships between features. At this point, we can initially assume that training data (sample) represents fairly enough the Titanic population.

# ## A Quick Look into the Data 

# In[60]:


from IPython.display import display, Markdown # For displaying texts more elegantly
pd.options.display.float_format = '{:,.1f}'.format #For showing a single decimal after point
display(trainDf.head(5))


# ## Types of the Variables (Features)
# As seen in the table above, the variables Survived, Pclass, Sex and Embarked can be considered as categorical variables while Age, SibSp, Parch and Fare can be considered as numerical variables. Also, we can use the PassengerId as the index of the dataframe since it seems to be a unique integer assigned to each row by the data wrangler. 

# In[61]:


def check_duplicates(df, colName):
    temp = df.duplicated(colName).sum()
    if temp == 0:
        display(Markdown("No duplicate values found in the {} column of the dataframe.".format(colName)))
    else:
        display(Markdown("There are {} duplicates in the {} column of the dataframe.".format(temp, colName)))
# Check if there are any duplicates in the PassengerId column of the train DataFrame
check_duplicates(trainDf, 'PassengerId')
trainDf = trainDf.set_index('PassengerId')


# As an initial guess, Name and Ticket variables should be unique to each passenger. Let's check to see if that is indeed the case.

# In[62]:


check_duplicates(trainDf, 'Name')
check_duplicates(trainDf, 'Ticket')


# Since all the values in the Name column are unique and I do not expect any relation between Name variable and any other variable in the trainDf, we do not need to examine the distribution of this variable. However, we should later investigate the Ticket variable and understand why there are duplicates.

# ## Examining Distributions
# We will consider the following:
#     1. Possible values the variable can take
#     2. Frequency of the variable taking these values

# ### Categorical Variables

# First, let's check to see if there are any null value in the categorical variables

# In[63]:


def check_null(df, colName):
    temp = df[colName].isnull().values.sum()
    if temp == 0:
        display(Markdown("No null element found in the {} column of the dataframe.".format(colName)))
    else:
        display(Markdown("There are {} null elements in the {} column of the dataframe.".format(temp, colName)))

check_null(trainDf, 'Survived')
check_null(trainDf, 'Pclass')
check_null(trainDf, 'Sex')
check_null(trainDf, 'Embarked')


# So, 2 elements of the Embarked column are null. Let's investigate these two elements.

# In[64]:


display(trainDf[trainDf['Embarked'].isnull()==True])


# The only missing columns int the PassengerId 62 and 830 rows are of variable Embarked. Let's see if there is any person with the same Cabin number or Ticket number. 

# In[65]:


if trainDf[trainDf['Cabin']=='B28'].index.tolist() == trainDf[trainDf['Embarked'].isnull()==True].index.tolist():
    display(Markdown('No person with cabin number B28 other than PassengerId 62 and 830'))
if trainDf[trainDf['Ticket']=='113572'].index.tolist() == trainDf[trainDf['Embarked'].isnull()==True].index.tolist():
    display(Markdown('No person with ticket number 113572 other than PassengerId 62 and 830'))


# Since these two persons have no siblings or parents, there is no straightforward way to figure out the Embarked columns for these two persons.

# Pie charts are useful tools for examining the distribution of categorical variables and gives insight about the importance of a particular variable wrt to whole range of possible values. 

# In[66]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True, palette='deep')

def subplot_pie(df, colName, colLabel, _title, _ax):
    temp = df.groupby(colName).count()
    temp = temp.rename(columns={'Name': colLabel})
    temp.plot(kind='pie', y=colLabel, ax=_ax, autopct='%1.0f%%', 
              startangle=90, shadow=True, #explode=[0.01, 0.01, 0.01],
             fontsize=13, legend=False, title=_title)

plt.figure(figsize=(10,10))
ax1 = plt.subplot(221, aspect='equal')
subplot_pie(trainDf, 'Pclass', 'Class', 'Class Distributions', ax1)
ax2 = plt.subplot(222, aspect='equal')
subplot_pie(trainDf, 'Survived', 'Survived', 'Survival Status', ax2)
ax3 = plt.subplot(223, aspect='equal')
subplot_pie(trainDf, 'Sex', 'Sex', 'Sex Distributions', ax3)
ax4 = plt.subplot(224, aspect='equal')
subplot_pie(trainDf, 'Embarked', 'Embarked', 'Port of Embarkation distributions', ax4)


# More than half of the passenger were from class 3 and 2/3 of the passengers were male. Also, most of the passengers embarked from Southampton port. Finally, survived passengers were 1/3 of the whole sample.

# Bar charts are also useful for examing the distribution of categorical variables. 

# In[67]:


ax = trainDf.groupby('Cabin').count()["Name"].plot(kind='barh', title ="Cabin distributions", figsize=(15,30),legend=False, fontsize=12)
ax.set_ylabel("cabin name",fontsize=12)
ax.set_xlabel("# of persons",fontsize=12)
display(Markdown('{} people do not have defined Cabin names'.format(trainDf['Cabin'].isnull().values.sum())))


# For the Cabin variable, we can see that most of the passengers do not have a defined Cabin name. The people with definde Cabin names generally stayed in their own. However, even 4 persons can share the same cabin.

# For quantitive variables, histogram and stem plots can be used to visualize their distributions.

# In[68]:


def subplot_hist(df, colName, _title, _ax):
    bins_ = 20
    df[colName].plot(kind="hist", alpha=0.8, bins= bins_, title=_title, ax=_ax)

plt.figure(figsize=(10,10))
ax1 = plt.subplot(221)
subplot_hist(trainDf, 'Age', 'Age Distributions', ax1)
ax2 = plt.subplot(222)
subplot_hist(trainDf, 'Fare', 'Fare Distributions', ax2)
ax3 = plt.subplot(223)
subplot_hist(trainDf, 'SibSp', 'Relatives Distributions', ax3)
ax4 = plt.subplot(224)
subplot_hist(trainDf, 'Parch', 'Close Relatives Distributions', ax4)


# All 4 quantitive variables are almost right skewed and unimodal. However, age distributions also have a peak near age 0 since small children require mothercare.

# In[69]:


display(trainDf[["Age", "SibSp", "Parch", "Fare"]].describe())


# When we investigate the measures of center, we can see that fare is affected by outliers since there is a large difference between the mean and median. Also, we can see that most of the passengers were alone when SibSp and Parch are considered.

# ## Examining Relationships

# The response variable is the Survived variable which is a categorical variable. The explanatory (independent) categorical variables are PClass, Sex, Embarked, Cabin while explanatory quantitative variables are Age, SibSp, Parch and Fare. Therefore, we have the following relationship structures:
# * C->C
# * Q->C

# For the C->C relationships, we can use two-way tables to examine the relationship between explanatory and response variables 

# In[70]:


def create_twowaytable_counts(df, explanatory, response):
    xIndices = trainDf.groupby(explanatory).count().index.values
    yIndices = trainDf.groupby(response).count().index.values
    resultDf = pd.DataFrame(index=xIndices)
    for y in yIndices:
        tempDf = df[df[response]==y].groupby(explanatory).count()
        tempDf = tempDf.rename(columns={"Name": y})
        resultDf = pd.concat([resultDf, tempDf[y]], axis=1)
    resultDf['total'] = resultDf.sum(axis=1)
    resultDf.loc['total'] = resultDf.sum(axis=0)
    return resultDf


# In[71]:


def create_twowaytable_percentages(df, explanatory, response):
    xIndices = trainDf.groupby(explanatory).count().index.values
    yIndices = trainDf.groupby(response).count().index.values
    resultDf = pd.DataFrame(index=xIndices)
    for y in yIndices:
        tempDf = df[df[response]==y].groupby(explanatory).count()
        tempDf = tempDf.rename(columns={"Name": y})
        resultDf = pd.concat([resultDf, tempDf[y]], axis=1)
    resultDf['total'] = resultDf.sum(axis=1)
    resultDf = resultDf.div(resultDf.max(axis=1), axis=0)*100
    return resultDf


# In[72]:


def create_doubleBarChart(df, explanatory, response):
    xIndices = trainDf.groupby(explanatory).count().index.values
    yIndices = trainDf.groupby(response).count().index.values
    resultDf = pd.DataFrame(index=yIndices)
    for x in xIndices:
        tempDf = df[df[explanatory]==x].groupby(response).count()
        tempDf = tempDf.rename(columns={"Name": x})
        resultDf = pd.concat([resultDf, tempDf[x]], axis=1)
    resultDf.loc['total'] = resultDf.sum(axis=0)
    resultDf = resultDf.div(resultDf.max(axis=0), axis=1)*100
    resultDf = resultDf.drop("total")
    resultDf.plot(kind="bar")


# In[73]:


display(create_twowaytable_counts(trainDf, 'Sex', 'Survived'))
display(create_twowaytable_percentages(trainDf, 'Sex', 'Survived'))
create_doubleBarChart(trainDf, 'Sex', 'Survived')


# When the relationship between sex and survived variables are examined, we can see that most of the female passengers survived while less than 20% of the male passengers survived.

# In[74]:


display(create_twowaytable_counts(trainDf, 'Pclass', 'Survived'))
display(create_twowaytable_percentages(trainDf, 'Pclass', 'Survived'))
create_doubleBarChart(trainDf, 'Pclass', 'Survived')


# When the relationship between class and survived variables are examined, we can see that survive ratio increases as the class value decreases from 3 to 1. 

# In[75]:


display(create_twowaytable_counts(trainDf, 'Embarked', 'Survived'))
display(create_twowaytable_percentages(trainDf, 'Embarked', 'Survived'))
display(embarkedSurvivedDf)


# When the relationship between embarkation and survived variables are examined, we can see that S and Q have similar percentages while C has the highest percentage. 

# In[95]:


import numpy as np
from sklearn import linear_model

tempDf = trainDf[['Age', 'Survived']].dropna(axis=0, how='any')
clf = linear_model.LogisticRegression()
clf.fit(tempDf['Age'].values.reshape(-1, 1), tempDf['Survived'])
x = np.linspace(tempDf['Age'].min(), tempDf['Age'].max(), 100)
y = 1 / (1 + np.exp(-(x*clf.coef_[0][0] + clf.intercept_[0])))
plt.scatter(tempDf['Age'], tempDf['Survived'])
plt.plot(x, y, color="red")

