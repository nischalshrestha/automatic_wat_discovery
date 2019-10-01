#!/usr/bin/env python
# coding: utf-8

# # Titanic Dataset Project
# 
# #### This Jupyter notebook is initially being created as primarily a self-learning tool. The broader goal is to provide other aspiring data scientists with a cleanly coded view of data analysis. I plan to explain topics so that people can understand my thought process and the general flow that I use when analying new data. I will provide links to source material whenever possible.

# ## 1. Module Imports
# Pandas is used extensively in this notebook for data munging. Seaborn is used for visualization where applicable for a more modern look. Seaborn is built on top of matplotlib functionality.
# 
# My standard imports are listed below:

# In[165]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# The following section imports individual models from sklearn. This generally follows the format I've seen from other kernels but in the future may move to a more generic *import sklearn as skl*. I'll test all of the models I've imported using cross-validation and provide a chart to display relative performance.

# In[166]:


from sklearn import model_selection
from sklearn import ensemble
from sklearn import linear_model


# ***Side note: I have found that when beginning a Kaggle project I am overly conerned with obtaining the best score. This is a failure on my part. I should be more focused on the following:***   
# *1. Understanding data science techniques.*  
# *2. Creating clear and concise documenation supporting my choices.*  
# *3. Developing interpretable results.*  

# ## 2. Load Test / Train Data
# 
# Load test data and train data into separate dataframes then join the two into a single dataframe. This will allow for the following:
# 1. Viewing the distribution of features using all observations.
# 2. Allow data to be imputed using the full distribution.
# 
# To recover the test data from the full dataframe I will filter on NaN in the response column.
# 
# ***Side note: I will use the following terms interchangeably:***  
# *1. Dependent variable, predicted variable, response, label*  
# *2. Independent variable, input variable, predictor, feature*  

# In[167]:


test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
full = train.append(test, ignore_index = True)

print('Dataset \'full\' shape:', full.shape)


# ## 3. Perform Initial Data Exploration
# 
# I will apply a few boilerplate data analysis functions to understand the basics of our data. This will start to build my intuition for the data and help with later data munging steps.
# 
# 1. Run 'head()' to have an idea of the data looks like. 
# 2. Determine which features are missing data. 
# 3. Run 'describe' to provide an initial quantiative view into the data.
#     1. This will help me find outliers and understand the distribution of the features.

# In[168]:


full.head(n=10)


# From the initial *head* method I have made the following observations:
# 1. Cabin has many NaN values so it may end up having to be dropped.
# 2. Name seems to include titles which may need to be extracted to provide useful information. 
#     1. It may be possible to build familial relationships with the names but with such a small dataset that may not be accurate.
# 3. PassengerId seems to be an autoincremented field which provides no predictive power.
# 4. Sex will need to be mapped to a binary variable for parametric modeling.
# 5. Ticket appears relatively arbitrary so it may need to be dropped.

# In[169]:


names = full.columns
null_count = full.isnull().sum()
null_pct = full.isnull().mean()
null_count.name = 'Null Count'
null_pct.name = 'Null Percent'
nulls = pd.concat([null_count, null_pct], axis=1)
nulls


# From the *isnull* method I see the following:
# 1. Age missing data will need to be imputed if it is going to be used as a regressor.
# 2. Cabin is overwhelmingly null and may not be useful in the prediction model.
# 3. Embarked has only two missing values and Fare only one so both will be imputed manually.  
# 
# *** Side note: If anyone has a cleaner way to create the count/percent data let me know. Perhaps grouping can be used. ***

# In[170]:


full.describe(include = 'all')


# From describe I can determine the following:
# 1. Age has a mean greater than the median (50% value) so that data is right-skewed. This makes sense given the relatively low median age and the likely number of older passengers.
# 2. Embarked has only three values so I'll investigate the correlation of this variable to the survival rate.
# 3. Fare has a max value significantly greater than the median which may mean there is invalid data.
# 4. Parch and SibSp have large max values so large family may be a useful indicator.
# 
# I'll look at the three NaN rows that will later be manually imputed. This will help determine what correlations and plots I'll create.

# In[171]:


full[full['Embarked'].isnull() | full['Fare'].isnull()]


# The passenger without a fare has 0 for Parch and 0 for SibSp so they are likely alone meaning I won't a chance to pull fare data from a fellow passenger. Later I will impute fare data by grouping on applicable columns and using the calculated median.

# In[172]:


full['Fare'] = full.groupby(['Pclass'])['Fare'].transform(lambda x: x.fillna(x.median()))


# I'll search for additional passengers with the same last names as the two passengers with nulls for embarked. I'll also look for additional records with the same ticket number as the two embarked nulls.

# In[173]:


full[full['Name'].str.contains('Stone') | full['Name'].str.contains('Icard') | full['Ticket'].str.contains('113572')]


# There doesn't appear to be additional data to help determine the embarked code for the two nulls. I will impute this data with the mode from the same Pclass.

# In[174]:


full['Embarked'] = full['Embarked'].fillna(full['Embarked'].mode()[0])
full['Embarked'].value_counts()


# Finally I'll confirm that I imputed the missing data correctly.

# In[175]:


names = full.columns
null_count = full.isnull().sum()
null_pct = full.isnull().mean()
null_count.name = 'Null Count'
null_pct.name = 'Null Percent'
nulls = pd.concat([null_count, null_pct], axis=1)
nulls


# ## 4. Perform Initial Data Munging and Feature Engineering
# 
# This section will perform more advanced data exploration using visualizations, aggregation, and statistical methods. First though I'll perform the following:
# 1. Map male/female to 0/1 so that the correlation matrix provides more useful data.
# 2. Engineer a new feature named family size that includes the passenger + Parch + SibSp. 

# In[176]:


full['Sex'] = full['Sex'].map( {'male':0, 'female': 1} ).astype(int)

full['FamilySize'] = full['Parch'] + full['SibSp'] + 1
full.drop(['Parch', 'SibSp'], axis=1, inplace=True)

full.corr()


# The correlation matrix provides only basic information. A pair of predictors or a predictor and the response variable could have non-linear relationships that aren't captured by the standard correlation matrix. I noticed the following which may be useful for prediction or imputing data:
# 1. Sex and Survived has a strong positive correlation meaning that females were more likely to survive (females encoded as 1 vs males encoded as 0 so the more positive number correlates to a greater probability of survival).
# 2. Pclass has a strong negative correlation with Fare.

# In[177]:


full['Fare'].hist(bins=50, grid=False);


# The fare histogram shows the bulk of ticket prices less than 100 dollars. Perhaps there is a data issue with tickets greater than $100. I'll look at rows of data corresponding to higher priced to tickets to see if I can find a trend.

# In[178]:


full[full['Fare'] >= 250].sort_values('Fare')


# My initial thought was that the fare for some passengers was a multiple of the single ticket price. This thought was based on, for example, three cabins associated with Thomas Cardeza. However if I look at Gustave Lesurer the price is identical Thomas Cardeza's but with only a single cabin. From this brief analysis I am going to conclude that the fare column is accurate.
# 
# Since the number of cabins won't be a differentiator I'll modify the cabin column to extract the first letter. I'll fill nulls with 'U' and then check the correlation between cabin letter and survival rate.

# In[179]:


full['CabinCode'] = full[full['Cabin'].notnull()].Cabin.astype(str).str[0]
full['CabinCode'].replace(np.NaN, 'U', inplace=True)


# I'm going to start working with the Name column. I'll do the following:
# 1. Extract title into a new column.
# 2. Extract last name into a new column.
# 3. Create a new column that indicates the aggregate survival rate for a family. Nulls will be filled with a toin-coss value of .5. The idea is that a passenger who is a member of a family with a high survival rate will be likely to themself survive.

# In[180]:


full['Title'] = full.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
full['FamilyName'] = full.Name.str.extract('([A-Za-z]+),', expand=False)
full['Title'].value_counts()


# After extracting titles from names it appears that there are many unique entries. A quick Google search revealed that Mlle is French for Miss and Mme is French for Mrs. I'll map the more unique names to standard replacements. The title field will be used later to impute ages but will probably not be specifically used in the training data.

# In[181]:


replacements = [
                [['Mr'],   ['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Major', 'Sir']],
                [['Miss'], ['Dona', 'Lady', 'Mlle', 'Ms']],
                [['Mrs'],  ['Countess', 'Mme']]
                ]

for title, replacement in replacements:
    full['Title'] = full['Title'].replace(replacement, ''.join(title))
    
full['Title'].value_counts()


# Rev likely means Reverend and was purposefully left out of the title mapping. I'll look at these rows to determine if they're only in the training data set and to see if a trend emerges.

# In[182]:


full[full['Title'].isin(['Rev'])]


# There are six Rev entries in the training set and all died. There are also two in the test data set. We can either force these to death in the final prediction or engineer a feature that strongly correlates Rev with death. For now I'll do that later.

# In[183]:


full['FamilySurvivalRate'] = full.groupby(['FamilyName'])['Survived'].transform(lambda x: x.mean())
full['FamilySurvivalRate'].fillna(value=.5, inplace=True)
full.loc[full['Title'] == 'Rev', 'FamilySurvivalRate'] = 0 # This is how I'll strongly correlate "Rev" with death.
full[full['Title'] == 'Rev']


# In[184]:


full['Age'] = full.groupby(['Pclass', 'Sex', 'Title'])['Age'].transform(lambda x: x.fillna(x.mean()))
full.drop(['Cabin', 'FamilyName', 'Name', 'Ticket', 'Title'], axis=1, inplace=True)
full.corr()


# To do:
# 1. Compare proportions for Pclass and Embarked.
# 2. Create scatter plot and outlier analysis for Fare.
# 3. Create a new family size variable and compare to survival rate.
# 4. Create a new Title variable and compare to survival rate.
# 5. Extract data from Cabin and compare to survival rate.

# In[185]:


def plot_distribution(df, var, target, **kwargs ):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue=target, aspect=4, row = row, col = col)
    facet.map(sns.kdeplot, var, shade= True)
    facet.set(xlim=( 0, df[ var ].max() ))
    facet.add_legend()

plot_distribution(full[full['Survived'].notnull()], var='Age', target='Survived', row='Sex')


# In[186]:


plot_distribution(full[full['Survived'].notnull()], var='Fare', target='Survived')


# In[187]:


def plot_bars(df, var, target, **kwargs):
    col = kwargs.get('col', None)
    sns.set()
    sns.set_style('white')
    sns.set_context('notebook')
    colors = ['sky blue', 'light pink']
    sns.barplot(data=df, x=var, y=target, hue=col, ci=None, palette=sns.xkcd_palette(colors))
    sns.despine()

plot_bars(full[full['Survived'].notnull()], 'Embarked', 'Survived', col='Sex')


# In[188]:


plot_bars(full[full['Survived'].notnull()], 'Pclass', 'Survived', col='Sex')


# ## 4. Begin data munging for the full dataset.
# 
# ### I. Transform four columns and remove unnecessary columns.
# 
# 1. Use dummy variables for passenger class and drop the original Pclass column.
# 2. Map gender to binary data: male = 0, female = 1.
# 3. Create a large family size column that indicates when a passenger was in a family of 8 or more. Train data indicates that large families (8 or ll people) had 100% mortality rates.
# 4. Extract passenger titles from names.

# In[189]:


full = pd.concat((full, pd.get_dummies(data=full['Pclass'], prefix='Pclass')), axis=1)
full.drop('Pclass', axis=1, inplace=True)
full.head()


# ## 5. Perform cross-validation and model prediction.

# In[191]:


X_train = full[full['Survived'].notnull()].copy()
X_train.drop(['Embarked', 'PassengerId', 'CabinCode', 'Survived'], axis=1, inplace=True)

Y_train = full[full['Survived'].notnull()].Survived.copy()

X_test = full[full['Survived'].isnull()].copy()
X_test.drop(['Embarked', 'PassengerId', 'CabinCode', 'Survived'], axis=1, inplace=True)


# In[192]:


models = [ensemble.RandomForestClassifier(n_estimators=100),
          ensemble.GradientBoostingClassifier(),
          linear_model.LogisticRegression()]
          
for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, Y_train)
    model_score = model.score(X_train, Y_train)
    accuracy = model_selection.cross_val_score(model, X_train, Y_train, scoring='accuracy', cv=10).mean() * 100
    Y_pred = model.predict(X_test).astype(int)
    submission = pd.DataFrame( {'PassengerId': test['PassengerId'], 'Survived': Y_pred} )
    submission.to_csv('./' + model_name + '_submission.csv', index=False)
    print('*' * 10, model_name, '*' * 10)
    print('Model score is:', model_score)
    print('Cross validation accuracy is:', accuracy)

