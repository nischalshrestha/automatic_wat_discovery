#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
sns.set_style("whitegrid")
get_ipython().magic(u'matplotlib inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# **Preambule**
# 
# This notebook builds upon previous EDA analysis done by other kernels. Hence, I would like to focus on several variables that have largest impact: Sex, Age, Family Members, and Social Class. This note also attempts to look deeper on several variables that seemed closely correlated, so that we can merge those features.
# 
# This note will be organized in three sections:
# 
# 1. **Understanding  passengers' class and ticket fare.** Higher class passengers are usually  prioritized. But how does it relate with ticket fare? Can we simplify the variables to make a more concise prediction?
# 
# 2. ** Understanding gender x age x family status.** Children and women are usually prioritized. But is it always the case? Would certain type of men are more likely to survive? How are the dynamics with families?
# 
# 3. **Decision Tree modeling**. Finally we will create decision tree model with two purpose: (1) Describing the socio-economic dynamics based on decision tree modeling. While not the strongest prediction model, Decision Tree can give a strong explanatory power and can give insight of what is likely happening. (2) Making the prediction that will be submitted via Kaggle. I am curious how a simple model like this would fare in the prediction.
# 

# **1. Understanding  passengers' class and ticket fare.**
# 
# Quick list of steps that I will do here:
# * Set up the dataframe
# * Show Violin + Swarm Plots on pclass and fare to understand relationship between the two
# * Show KDE distribution for more visualization lens
# * Create new segment var to combine pclass and fare
# * - Here I also tested 2 options to choose from
# * - Create bins of fare for easier EDA

# In[ ]:


# Setting up the dataframe
train = pd.read_csv('../input/train.csv')
train.info()
train.head()


# In[ ]:


# Creaet violin and swarm plots
sns.violinplot(x='Pclass',y='Fare',data=train,inner=None)
sns.swarmplot(x='Pclass',y='Fare',data=train,color='w',alpha=0.5)
plt.title("Violin and Swarm Plot to compare fare distribution among Pclass groups")
# Note: This takes a long time to run


# Key observations from violin and swarm plots:
# * Pclass=1 fare distribution is very interesting. A significant number of passenger with "premium" tickets. But still a good number of passengers with same fare as pclass
# * Pclass=2 and Pclass=3 have a relatively similar distribution. But it might need a zoom
# * It seems that the premium fare passengers should be carved out as separate segment. But should we pick from pclass=1 only? 
# 
# Some more visualizations below to better understand the distribution of pclass vs. fares

# In[ ]:


# Create kde plot
sns.kdeplot(train['Fare'][train['Pclass']==1],shade=True)
sns.kdeplot(train['Fare'][train['Pclass']==2],shade=True)
sns.kdeplot(train['Fare'][train['Pclass']==3],shade=True)
plt.title("Distribution of fares across Pclass groups")


# In[ ]:


# Create distribution plot again for pclass=1, because the scale is hard to read in previous one
fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(12,4))
fig.suptitle("PDF and CDF of fares among Pclass=1")
sns.distplot(train['Fare'][train['Pclass']==1],rug=True,ax=axis1)
axis1.set_title("PDF")
sns.kdeplot(train['Fare'][train['Pclass']==1],shade=True,cumulative=True,ax=axis2)
axis2.set_title("CDF")
x_forplot = train['Fare'][train['Pclass']==1]
axis2 = plt.xticks(np.arange(min(x_forplot),max(x_forplot)+1,40.0))


# In[ ]:


# Create distribution plot again for pclass=2 and 3, because the scale is hard to read in previous one
sns.kdeplot(train['Fare'][train['Pclass']==2],shade=True)
sns.kdeplot(train['Fare'][train['Pclass']==3],shade=True)
plt.title("PDF of fares among Pclass= 2 and 3")

# Note: You need to add this line of code for the code above to work - %matplotlib inline


# For feature engineering, I would combine pclass and fare using the following options:
# * PaxclassA: Keep pclass 2 and 3 segment as-is. Split pclass 1 into premium and normal segment. Fare > 100 as threshold, based on 20-80 distribution.
# * PaxclassB: Carve out premium fares as one segment. Then keep existing pclass segment. Fare > 60 set as premium
# 
# 

# In[ ]:


# Create the 2 new segmentation columns
train['PaxclassA'] = train['Pclass']
train.loc[(train['PaxclassA'] == 1) & (train['Fare']>100),'PaxclassA'] = 0
train['PaxclassB'] = train['Pclass']
train.loc[(train['Fare']>60),'PaxclassB'] = 0


# In[ ]:


# Group fare into bins to analyze survival rate across brackets. The brackets are informed by the dist plot above
bins = [0,20,40,60,80,100,200,400,800]
train['Fare_Groups'] = pd.cut(train['Fare'],bins)


# In[ ]:


# Create plots to compare survival between the 2 new segmentation columns. We also show similar plot based on original Pclass and Fare buckets
fig, ((axis1,axis2),(axis3,axis4)) = plt.subplots(2,2,sharey=True,figsize=(12,4))
sns.factorplot("PaxclassA","Survived",data=train,ax=axis1)
sns.factorplot("PaxclassB","Survived",data=train,ax=axis2)
sns.factorplot("Pclass","Survived",data=train,ax=axis3)
sns.factorplot("Fare_Groups","Survived",data=train,ax=axis4)
fig.suptitle("Survival Rate across Segments, based on 2 new segments and 2 original vars")

# Note: I still don't know why these line of codes produce the blank charts on the bottom....


# So how do the two new segments compare?
# 1. The first model takes premium only from pclass-1 with Fare >100. Here we could see the 4 groups with large gaps between each other. So this seems to work better
# 2. The second model takes premium from any class that pays Fare >60. Here, it looks like the pclass=1 and pclass=2 seemed similar in survivability. 
# 
# Before concluding to choose first segment model, let's do a heatmap in next analysis to better interpret  the result from 2nd mode. In the heatmap below, we could see that within pclass=1, we could see drop in survivability at fares less than 80

# In[ ]:


train['Fare_Groups2'] = train['Fare_Groups'].astype("object") # Need this conversion for heatmap to work
sns.heatmap(pd.crosstab(train['Pclass'],train['Fare_Groups2'],values=train['Survived'],aggfunc=np.mean).T,annot=True,cmap="Blues")
plt.title("Crosstab Heatmap of Pclass x Fares")

# Ideally, I should add 1 more heatmap to show the count. But let me put it in backburner, as the count is quite large for the regular pclass=1


# **2.  Understanding gender x age x family status.**

# Quick list of steps that I will do here:
# * Show cross-tab heatmap to see interaction of Sex and Age Groups
# * - Children < 12 are special! But no other interaction
# * - Then we integrate sex and age variables
# * Show cross-tab on impact of having Siblings or Spouse
# * - This is useful to indicate single man / woman, which turned out to have higher survival rate
# * Show cross-tab on impact of having Children / Parents
# * - Large number of childrens make it less likely to survive

# In[ ]:


# Analyzing cross-tab of age and sex on survival
bins = [0,12,18,35,50,70,100]  # General age group breakdown
train['Age_Groups'] = pd.cut(train['Age'],bins)
sns.heatmap(pd.crosstab(train['Sex'],train['Age_Groups'],values=train['Survived'],aggfunc=np.mean).T,annot=True,cmap="Blues")
plt.title("Crosstab Heatmap of Sex x Age: Children (<12yo) seems prioritized, but elderly were not")


# In[ ]:


# Create combined variable and show the survival rate
train['SexAge'] = train['Sex']
train.loc[(train['Age']<=12),'SexAge'] = 'children'
sns.factorplot("SexAge","Survived",data=train)


# In[ ]:


# Crosstab and heatmap on the impact of having parents / children
print(pd.crosstab(train['SexAge'],train['Parch']))
crosstab1 = pd.crosstab(train['SexAge'],train['Parch'],values=train['Survived'],aggfunc=np.mean)
sns.heatmap(crosstab1.T,annot=True,cmap="Blues")
plt.title("Crosstab Heatmap of SexAge x Parch")


# Key observations:
# * Age only matters if you are children, defined as < 12 yo
# * If you are a male, bringing a children won't help your survival
# * If you are a female, bringing a children don't impact your survival. Except if you have too many children (4+ kids)
# * For kids, seems that they are more likely to survive if they come with one parent rather than both parents. Some hypothesis of why this is likely:
# * - Perhaps if you had both parents, you are more likely to stick together (thus less likely to survive)
# * - If the child came with only mom, s/he is more likely to survive
# * - If the child cdme with only dad, s/he is less likely to survive
# * - Unfortunately I am not sure if we can identify if a child goes with mom or dad

# In[ ]:


print(pd.crosstab(train['SexAge'],train['SibSp']))
crosstab1 = pd.crosstab(train['SexAge'],train['SibSp'],values=train['Survived'],aggfunc=np.mean)
sns.heatmap(crosstab1.T,annot=True,cmap="Blues")
plt.title("Crosstab Heatmap of SexAge x SibSp")


# Key observations:
# * Similar to above, children with many siblings are unlikely to survive
# * For females, it doesn't really matter whether you are married or single :)
# * For adults (incl. teenagers), having many siblings make you less likely to survive. But the amount of these instances are few
# * For males, if you are married, you're a bit more likely to survive :). Nonetheless, I am still curious what is the profile of the men who survived.

# **3. Decision Tree Modeling**

# Quick list of steps that I will do here:
# * Convert categorical into binary variables
# * Run the training model and then export into graphviz
# * - The graphviz output needs to be visualized using external website. The result is posted here.
# * Run the prediction on test dataset
# * - Do all the pre-processing on test data
# * - Run the prediction and output into csv for submission

# In[ ]:


# We need to convert categorical variables into binary variable
train['Female'] = 0
train.loc[(train['SexAge']=="female"),'Female'] = 1
train['Children'] = 0
train.loc[(train['SexAge']=="children"),'Children'] = 1
train['Class1_Premium'] = 0
train.loc[(train['PaxclassA']==0),'Class1_Premium'] = 1
train['Class1'] = 0
train.loc[(train['PaxclassA']==1),'Class1'] = 1
train['Class2'] = 0
train.loc[(train['PaxclassA']==2),'Class2'] = 1


# In[ ]:


# Define the variables for training
from sklearn import tree
Xtrain = train[['Female','Children','Parch','SibSp','Class1_Premium','Class1','Class2']]
Ytrain = train['Survived']


# In[ ]:


# Set up and fit the decision tree model. Then export as graphviz
Tree1 = tree.DecisionTreeClassifier(max_depth=4,min_samples_split=50,random_state=1)
Tree1.fit(Xtrain,Ytrain)
Tree1_dot = tree.export_graphviz(Tree1,out_file=None,feature_names=Xtrain.columns,class_names=['Not Survived','Survived'],proportion=True,filled=True)
print(Tree1_dot)


# Some key comments on decision I made on the Decision Tree parameters:
# * Overall, I want a parsimonious model that tries to explain a lot with few variables
# * Thus, I set a minimum sample of 50, and stop the branching right there
# * I set filled=True to show color gradient on the decision tree
# 
# To makes sense the decision tree model, you need to paste the output above into this website: http://www.webgraphviz.com/
# 
# Below is the decision tree output. I added my own annotation to make it easier to read and interpret
# ![](https://i.imgur.com/Igoy3DG.jpg)
# 
# Some most interesting observations:
# * The top differentiating variable is whether you are male, female, or children
# * If you are male, your best bet is to be a single man in 1st class. Survival rate is about 50-50. That's good considering average survival for male is 16%
# * If you are children with < 3 siblings, your chances are also strong at 86%. If you have 3+, survival rate is at 8%
# * If you are an adult woman, you are generally very fine. Unless if you are on 3rd class (pclass=3). But it's still 50-50, roughly the same as single man in 1st class
# 
# Do note that some of the splitting is not really necessary, because the survival rate is already at ~90% and no significant changes in next branching. At some point I might be interested to add a 2nd decision tree model that sets a stopping point based on improvement of the gini ratio. It is set by adding min_impurity_decrease parameter.

# In[ ]:


# Check the score of prediction accuracy
Tree1.score(Xtrain,Ytrain)


# Lastly... we will now do all the same steps for the test data, and finally produce the submission file!

# In[ ]:


test = pd.read_csv('../input/test.csv')

# Create new combined variables
test['SexAge'] = test['Sex']
test.loc[(test['Age']<=12),'SexAge'] = 'children'
test['PaxclassA'] = test['Pclass']
test.loc[(test['PaxclassA'] == 1) & (test['Fare']>100),'PaxclassA'] = 0

# Create binary variables out of categorical variables
test['Female'] = 0
test.loc[(test['SexAge']=="female"),'Female'] = 1
test['Children'] = 0
test.loc[(test['SexAge']=="children"),'Children'] = 1
test['Class1_Premium'] = 0
test.loc[(test['PaxclassA']==0),'Class1_Premium'] = 1
test['Class1'] = 0
test.loc[(test['PaxclassA']==1),'Class1'] = 1
test['Class2'] = 0
test.loc[(test['PaxclassA']==2),'Class2'] = 1

# Create the prediction
Xtest = test[['Female','Children','Parch','SibSp','Class1_Premium','Class1','Class2']]
Ytest_pred = Tree1.predict(Xtest)


# In[ ]:


submission = pd.DataFrame({
    "PassengerId":test['PassengerId'],
    "Survived":Ytest_pred
})
submission.to_csv('titanic.csv',index=False)


# Public Score: 0.77990
# Rank: 4335

# **Appendix 1: Alternative Decision Tree**

# In[ ]:


# Set up and fit the decision tree model. Then export as graphviz
Tree2 = tree.DecisionTreeClassifier(max_depth=6,min_samples_split=50,random_state=1,min_impurity_decrease=0.0003)
Tree2.fit(Xtrain,Ytrain)
Tree2_dot = tree.export_graphviz(Tree1,out_file=None,feature_names=Xtrain.columns,class_names=['Not Survived','Survived'],proportion=True,filled=True)
print(Tree2_dot)


# In[ ]:


# Create the prediction
Xtest = test[['Female','Children','Parch','SibSp','Class1_Premium','Class1','Class2']]
Ytest_pred = Tree2.predict(Xtest)

submission = pd.DataFrame({
    "PassengerId":test['PassengerId'],
    "Survived":Ytest_pred
})
submission.to_csv('titanic2.csv',index=False)


# New Public Score: 0.77033
# Not much of an improvement. Clearly need to do 2 things: (1) Consider adding variables, (2) Implement random forest or 
