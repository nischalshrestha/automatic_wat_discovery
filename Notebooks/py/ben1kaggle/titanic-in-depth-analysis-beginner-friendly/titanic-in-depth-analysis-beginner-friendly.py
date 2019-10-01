#!/usr/bin/env python
# coding: utf-8

# # Titanic survival prediction
# 
# ## Import general packages

# In[ ]:


# Importing general packages and set basic plot styling
get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from IPython.display import YouTubeVideo
import warnings
warnings.filterwarnings('ignore')

sns.set_style("white")
sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
plt.rcParams['axes.color_cycle'] = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', 
                                    u'#9467bd', u'#8c564b', u'#e377c2']


# ## Introduction
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships. One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew.   
# <br>
# In this notebook, we will first clean data and fill in missing values with linear regression. Then some feature engineering after which we will explore the data thoroughly. Finally, we will use random forest machine learning to predict the survival of Titanic passengers. 

# In[ ]:


YouTubeVideo('NdZ6TY1pxL8')
# https://www.youtube.com/watch?v=NdZ6TY1pxL8 


# ## Load data

# In[ ]:


# Load and merge the train and test data
df1 = pd.read_csv('../input/train.csv')
df2 = pd.read_csv('../input/test.csv')
df1['Set'] = 'train'
df2['Set'] = 'test'
df=df1.append(df2)
df=df.reset_index()
df.info()
df.head()


# ## Missing values (part 1)
# There are 4 variables that contain missing values which we will have to deal with first.
# * A fair amount of missing values in Age, we can fill those in reasonably with a regression model. 
# * By far the most values for Cabin are missing, so we will not try to fill these in and simply mark them as unknown.
# * Only 2 values are missing for Embarked.
# * Just one 1 value is missing in Fare.    
# <br>
# 
# #### Embarked
# Two ladies on the same 1st class ticket have unknown embarked values. A quick google on the ticket number leads to encyclopedia-titanica.org which says they embarked in Southampton. 

# In[ ]:


# Missing values Embarked 
display(df[df['Embarked'].isnull()])

df['Embarked'] = df['Embarked'].fillna('S') 


# #### Fare
# The Fare for Mr. Storey is missing. On encyclopedia-titanica.org his name does not lead to a known Fare. His ticket number is actually 370160 instead of 3701 which has 5 other people on it. Too bad, Ticket 370160 does not occur in our dataset. It looks like Mr. Storey is part of the staff on the Titanic so he probably did not pay a Fare at all. We will ignore that and simply fill in his Fare with the median for 3rd class passengers embarked in Southampton. 

# In[ ]:


# Missing value Fare
display(df[df['Fare'].isnull()])

a = df['Fare'].loc[(df['Pclass']==3) & (df['Embarked']=='S')]

plt.figure(figsize=[7,3])
sns.distplot(a.dropna(), color='C0')
plt.plot([a.median(), a.median()], [0, 0.16], '--', color='C1')

df['Fare'] = df['Fare'].fillna(a.median())

sns.despine(bottom=0, left=0)
plt.title('Fare for 3rd class embarked in S')
plt.xlabel('Fare')
plt.legend(['median'])
plt.show()


# #### Age
# There are many Age values missing, 263 to be precise. We will perform some linear regression to reasonably predict the ages of the passengers, so that the fake ages blend in nicely with the known Titanic population. 
# 
# Because we would like to incorporate as much information as possible in the regression model, we will first move on to the feature engineering and cleaning.
# 

# ## Feature engineering
# In this section we first clean up some features. Survived and Sex are simply changed to string values, to make the plotting a bit nicer later on. Fare is in british pounds from 1912, because of inflation that is wildly incomparable to today's value of the GBP. We correct for that and then convert to USD, just for fun.  
# 
# * The Name column contains information about the social status of the person because it includes the person's title. We extract it and merge some rare titles into more common categories.  
# 
# 
# * Another interesting feature we can make from the existing data is family size. Did larger families perhaps get priority to get on the lifeboats? Or would they be lost while looking for eachother in the chaos? We sum the Parch and SibSp together and add 1 for the individual itself into FamSize. FamSize2 is categorized as single, small and large families.  
# 
# 
# * Similarly to FamSize, we make a group size variable. This is based on number of people sharing the same Ticket. So this includes families (and potential nannies for example), but also groups of friends or colleagues. GrpSize is the actual number of people in the group and GrpSize2 is categorized as FamSize2.  
# 
# 
# * The Cabin column is pretty useless on it's own. However, we can easily extract information about on which deck the passenger was staying. Perhaps certain decks are easier to escape from. We take the first letter from the Cabin number and assign it to the Deck column. Most Cabin values are NaN, which we assign as 'X' for unknown. 
# 
# 
# * Cabin contains more information. The cabin number after the Deck letter tells us something about the location from front to back. The deckplans are available at encyclopedia-titanica.org/titanic-deckplans/ . Based on that, we make a rough distinction between front, mid and back of the ship. 
# 

# In[ ]:


# Clean up and feature engineering

# Label Survived for plot
df['Survived'] = df['Survived'].replace([0, 1], ['no', 'yes']) 

# Label Sex for plot
df['Sex'] = df['Sex'].replace([0, 1], ['male', 'female']) 

# Transform Fare to today's US dollar, for fun
df['Fare'] = df['Fare']*108*1.3 #historic gbp to current gbp to current usd

# Get personal title from Name, merge rare titles
df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split(' ')[1])
toreplace = ['Jonkheer.', 'Ms.', 'Mlle.', 'Mme.', 'Capt.', 'Don.', 'Major.', 
             'Col.', 'Sir.', 'Dona.', 'Lady.', 'the']
replacewith = ['Master.', 'Miss.', 'Miss.', 'Mrs.', 'Sir.', 'Sir.', 'Sir.',
              'Sir.', 'Sir.', 'Lady.', 'Lady.', 'Lady.']
df['Title'] = df['Title'].replace(toreplace, replacewith)

# Get family names
df['FamName'] = df['Name'].apply(lambda x: x.split(',')[0])

# Get family sizes based on Parch and SibSp, classify as single/small/large
df['FamSize'] = df['Parch'] + df['SibSp'] + 1
df['FamSize2'] = pd.cut(df['FamSize'], [0, 1, 4, 11], labels=['single', 'small', 'large'])

# Get group sizes based on Ticket, classify as single/small/large
df['GrpSize'] = df['Ticket'].replace(df['Ticket'].value_counts())
df['GrpSize2'] = pd.cut(df['GrpSize'], [0, 1, 4, 11], labels=['single', 'small', 'large'])

# Get Deck from Cabin letter
def getdeck(cabin):
    if not pd.isnull(cabin) and cabin[0] in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        return cabin[0]
    else:
        return 'X'    
    
df['Deck'] = df['Cabin'].apply(getdeck)

# Get a rough front/mid/back location on the ship based on Cabin number
'''
A front
B until B49 is front, rest mid
C until C46 is front, rest mid
D until D50 is front, rest back
E until E27 is front, until E76 mid, rest back
F back
G back
Source: encyclopedia-titanica.org/titanic-deckplans/
'''
def getfmb(cabin):
    
    if not pd.isnull(cabin) and len(cabin)>1:
        if (cabin[0]=='A'
            or cabin[0]=='B' and int(cabin[1:4])<=49
            or cabin[0]=='C' and int(cabin[1:4])<=46
            or cabin[0]=='D' and int(cabin[1:4])<=50
            or cabin[0]=='E' and int(cabin[1:4])<=27):
            return 'front'
        
        elif (cabin[0]=='B' and int(cabin[1:4])>49
            or cabin[0]=='C' and int(cabin[1:4])>46
            or cabin[0]=='E' and int(cabin[1:4])>27 and int(cabin[1:4])<=76):
            return 'mid'

        elif (cabin[0]=='F'
           or cabin[0]=='G'
           or cabin[0]=='D' and int(cabin[1:4])>50):
            return 'back'
        
        else:
            return 'unknown'
    else:
        return 'unknown'        
    
df['CabinLoc'] = df['Cabin'].apply(getfmb)

dfstrings = df.copy() # save df containing string features to use for plotting later


# ## Missing values (part 2)
# Now that we have all our features, we can use them to create a regression model for filling in the missing values in Age. First, we have to factorize the string based features.

# In[ ]:


# Factorize the string features

df['CabinLoc'] = df['CabinLoc'].replace(['unknown', 'front', 'mid', 'back'], range(4))

df['Deck'] = df['Deck'].replace(['X', 'A', 'B', 'C', 'D', 'E', 'F', 'G'], range(8))

df['GrpSize2'] = df['GrpSize2'].astype(str) #convert from category dtype
df['GrpSize2'] = df['GrpSize2'].replace(['single', 'small', 'large'], range(3))

df['FamSize2'] = df['FamSize2'].astype(str) #convert from category dtype
df['FamSize2'] = df['FamSize2'].replace(['single', 'small', 'large'], range(3))

df['Title'] = df['Title'].replace(df['Title'].unique(), range(8))

df['Embarked'] = df['Embarked'].replace(['S', 'C', 'Q'], range(3))

df['Sex'] = df['Sex'].replace(['male', 'female'], range(2)) 

df['Survived'] = df['Survived'].replace(['no', 'yes'], range(2)) 

dfnum = df.copy() # save df containing factorized features to use for subsequent analysis


# #### Age
# The first thing to do is split the data for which we have the known Age into a train and test set. We use 3/4 of the 1046 values as the training set. This leaves 262 as test set, coinciding nicely with our unknown data which is n=263. 
# 
# After fitting a model to the train set, we look at the root mean squared error (rmse). The rmse is 12.2 years, certainly not perfect. For cross validation (top figure), we use the model for predicting the test set. The rmse in this case is 12.5. Thus, the rmse of train and test are the same, which means the model generalizes well to out of sample data (no over or underfitting). 
# 
# Next, we use the model to predict the unknown ages. The resulting ages range from 3.8 to 44.1. Compared to applying the model to the known dataset, it does not look so great (middle figure). The model predicts the values quite narrowly around the mean for the unknown data. We'll take it anyway, because better than throwing all the unknown rows out or simply replacing the NaNs with mean/median.
# 
# Lastly, we merge the predicted missing values with the known data. Then, we compare our original age distribution with the distribution of the final result (bottom figure). The density around the mean is increased somewhat compared to the original distribution. 
# 
# 

# In[ ]:


# Multiple linear regression modeling of Age
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

feats = ['Sex', 'Embarked', 'Pclass', 'Fare', 'Title', 'Parch', 'SibSp', 'FamSize', 'FamSize2', 
         'GrpSize', 'GrpSize2', 'Deck', 'CabinLoc' ]

dffeats = df[feats][df['Age'].notnull()]
dfresp = df['Age'][df['Age'].notnull()]
dfmiss = df[feats][df['Age'].isnull()]

X_train, X_test, y_train, y_test = train_test_split(dffeats, dfresp, test_size=0.25, random_state=100)

lm = LinearRegression()
lm.fit(X_train, y_train )

y_predtrain = lm.predict(X_train)
print('Mean train: ' + str(np.mean(y_predtrain)))
print('Std predtrain: ' + str(np.std(y_predtrain)))
print('RMSE predtrain: ' + str(np.sqrt(metrics.mean_squared_error(y_train, y_predtrain))))

y_predtest = lm.predict(X_test)
print('Mean test: ' + str(np.mean(y_predtest)))
print('Std predtest: ' + str(np.std(y_predtest)))
print('RMSE predtest: ' + str(np.sqrt(metrics.mean_squared_error(y_test, y_predtest))))

pred1 = lm.predict(dffeats)
pred2 = lm.predict(dfmiss)
print(pred2.min(), pred2.max())


# In[ ]:


# Plots for Age regression
fig, [ax1, ax2, ax3] = plt.subplots(3,1, figsize=[7,6])

sns.distplot(y_predtrain, hist=False, label='prediction of train set (n=784)', ax=ax1)
sns.distplot(y_predtest, hist=False, label='prediction of test set (n=262)', ax=ax1)
ax1.set_xlim([0, 80])

sns.distplot(pred1, hist=False, label='prediction of known (n=1046)', ax=ax2)
sns.distplot(pred2, hist=False, label='prediction of missing (n=263)', ax=ax2)
ax2.set_xlim([0, 80])
ax2.set_ylim([0, 0.15])

sns.distplot(dfresp, hist=False, label='known (n=1046)')
sns.distplot(dfresp.values.tolist() + pred2.tolist(), hist=False, label='known + predicted missing (n=1309)')
ax3.set_xlim([0, 80])
ax3.set_ylim([0, 0.05])

fig.tight_layout()
sns.despine(bottom=0, left=0)


# #### Update dataframe with Age
# Below we add the the newly predicted Age values to the dataframe. Additionally, AgeGrp, AgeDec, and Persontype are created with the Age variable. AgeGrp groups ages into child/teen/adult. AgeDec simplifies Age by categorizing people's age into decades. PersonType is a combination of Sex and Age to differentiate between male kids and male adults, because of the 'women and children first' rule. 

# In[ ]:


# Updating the dataframe (strings version) with Age related features
df = dfstrings
df['Age'].loc[df['Age'].isnull()] = pred2

# Classify age groups
df['AgeGrp'] = pd.cut(df['Age'], [0, 12, 20, 200], labels = ['child', 'teen', 'adult'])

# Classify age decade
df['AgeDec'] = pd.cut(df['Age'], range(0,90,10), labels=range(8))
df['AgeDec'] = df['AgeDec'].astype(int)

# Classify male/female/child
df['PersonType'] = df['Sex']
df.loc[df['Age']<12,'PersonType'] = 'child'


# In[ ]:


# Updating the dataframe (numbers version) with Age related features
dfnum['Age'] = df['Age']

dfnum['AgeGrp'] = df['AgeGrp'].astype(str) #convert from category dtype
dfnum['AgeGrp'] = dfnum['AgeGrp'].replace(['child', 'teen', 'adult'], range(3))

dfnum['AgeDec'] = df['AgeDec']

dfnum['PersonType'] = df['PersonType'].replace(['male', 'female', 'child'], range(3)) 


# ## Data exploration
# #### Histograms
# Now, let's have a look at all features in the dataframe that we can use for predicting survival. Making use of the handy 'hue' parameter in the seaborn countplot function we can easily plot histograms split on survival to get a sense of the proportions. 
# <br>
# 1. First, we simply plot how many people **Survived** and how many did not. Most people died, so if we were to simply predict that everybody dies we would be correct approximately 62% (342/549=0.62). Hopefully we can do better than that. 
# 
# 2. **Sex** has a clear effect. By far most men died whereas most women survived. 
# 
# 3. **Age** also shows promise. Of the survivors, more people are young and less are in the 20-30 range relative to the victims. Strangely, there is a bit of a dip in the number of people aboard that are about 10-15. Perhaps people with school age kids tend not to go on long trips in April?
# 
# 4. The conclusions from Age are also reflected in **AgeGrp** and **AgeDec**. Children died about 50-50, which are good odds compared to teens and certainly adults. People below 10 are slightly more likely to survive, but it is already reversed for people in 10-20, with 20-30 being the worst odds to survive. **PersonType** merely combines Sex and AgeGrp.
# 
# 5. Place of **Embarkment** strangely has an effect, since people from Cherbourg were more likely to survive whereas Southamptom and Queenstown were more likely to die. Perhaps this is due to differences in class of the people embarking in each location?
# 
# 6. **Pclass** is obviously an important feature. First class passenger clearly got priority boarding the lifeboats as most of them survived. Equal odds for 2nd class passengers. By far most people on titanic were 3rd class and by far most of them did not survive.
# 
# 7. The Pclass effect is also reflected in the **Fare**. Most people paying less than about 2500 today's USD worth did not survive. Above that amount of money, you were slightly more likely to survive. The most expensive ticket was worth almost 72000 bucks. Pricey. 
# 
# 8. **Title** seems to reflect mostly the male/female difference, the higher titles do have increased survivability but there are so few of them it might not be of much relevance. 
# 
# 9. From **Parch, SibSp, FamSize1 **and** FamSize2**, we can conclude that being part of a small family was a good benefit. However, being a single traveller or part of a large family was detrimental to survival odds. Perhaps small families did indeed get priority for lifeboats, with larger families more likely to have to search for eachother?
# 
# 10. The same conclusions can be drawn for **GrpSize** and **GrpSize2**. Most groups are families and the non-family groups might not be too plentiful to affect the distribution much. 
# 
# 11. Lastly, **Deck** and **CabinLoc**. It seems most people survive overall? These distributions are based on only 295 values of known Cabin number. Probably, if your cabin number was known you were 1st or 2nd class and therefore more likely to survive. 
# 
# 

# In[ ]:


# Plot histograms of all the features split on 'Survived'

fig, axes = plt.subplots(5,4, figsize=[12,13])
axes = axes.ravel()
axnr = 0
for i in ['Survived', 'Sex', 'Age', 'Age', 'AgeGrp', 'AgeDec', 'PersonType', 'Embarked', 'Pclass', 
          'Fare', 'Fare', 'Title', 'Parch', 'SibSp', 'FamSize', 'FamSize2', 'GrpSize', 'GrpSize2', 
          'Deck', 'CabinLoc' ]: 
    sns.countplot(i, hue='Survived', data=df, ax=axes[axnr])
    axes[axnr].set(xlabel=i, ylabel="")
    axes[axnr].legend().set_visible(False)
    axnr += 1

axes[0].cla() #clear and replace plot
sns.countplot('Survived', data=df, ax=axes[0])
axes[0].set(xlabel='Survived', ylabel="")
    
axes[2].cla()
sns.distplot(df['Age'][df['Survived']=='no'], ax=axes[2], bins=range(0, 100, 2))
sns.distplot(df['Age'][df['Survived']=='yes'], ax=axes[2], bins=range(0, 100, 2))

axes[3].cla()
sns.distplot(df['Age'][df['Survived']=='no'], ax=axes[3], bins=range(0, 100, 2))
sns.distplot(df['Age'][df['Survived']=='yes'], ax=axes[3], bins=range(0, 100, 2))
axes[3].set(xlim=[0,40], ylim=[0,0.05])

axes[9].cla()
sns.distplot(df['Fare'][df['Survived']=='no'], ax=axes[9], bins=range(0, 72000, 300))
sns.distplot(df['Fare'][df['Survived']=='yes'], ax=axes[9], bins=range(0, 72000, 300))
axes[9].set(ylim=[0,0.0005])

axes[10].cla()
sns.distplot(df['Fare'][df['Survived']=='no'], ax=axes[10], bins=range(0, 72000, 300))
sns.distplot(df['Fare'][df['Survived']=='yes'], ax=axes[10], bins=range(0, 72000, 300))
axes[10].set(xlim=[0,10000], ylim=[0,0.0005])

axes[11].set_xticklabels(axes[11].get_xticklabels(), rotation = 45, size='x-small', ha="center")

axes[18].cla()
sns.countplot(df['Deck'][df['Deck'] != 'X'], hue='Survived', data=df, 
              ax=axes[18], order=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
axes[18].set(xlabel='Deck', ylabel="")
axes[18].legend().set_visible(False)

axes[19].cla()
sns.countplot(df['CabinLoc'][df['CabinLoc'] != 'unknown'], hue='Survived', data=df, 
              ax=axes[19], order=['front', 'mid', 'back'])
axes[19].set(xlabel='CabinLoc', ylabel="")
axes[19].legend().set_visible(False)    
    
plt.title('Histograms')    
fig.tight_layout()
sns.despine(bottom=1, left=1)


# #### Correlation heatmap
# After we got a basic understanding of the features in our dataset, let's plot a correlation heatmap too see if we can detect some interesting patterns.   
# <br>
# As expected, a strong correlation exists between Survival and Sex. However, the purely Age related features do not show a strong correlation to Survival, potentially due to a U shape relation?  
# PersonType is basically the Sex predictor for Survival, but a bit weaker because of the Age related component. Pclass and Fare are naturally correlated to eachother, and both are strongly correlated to Survival. Pclass and Age are also obviously related because older people tend to be richer. Deck and CabinLoc are decently correlated with Survival, but as we hypothesized before, these are indeed strongly correlated with Pclass. Title's correlation with Survival is mainly influenced by Sex and not so much Pclass.  
# FamSize2 and GrpSize2 are interesting for Survival prediction. The family/group size variables are logically correlated with age variables since children tend to be part of families and many adults travel single.  
# Embarked has a minor correlation to Survival as well, but not to Pclass as we hypothesized before.   
# 

# In[ ]:


df = dfnum
fig = plt.subplots(figsize=[15, 15])

sns.heatmap(df[['Survived', 'Sex', 'Age', 'AgeGrp', 'AgeDec', 'PersonType', 'Embarked', 'Pclass', 
                'Fare', 'Title', 'Parch', 'SibSp', 'FamSize', 'FamSize2', 'GrpSize', 'GrpSize2', 
                'Deck', 'CabinLoc']].corr(), 
            annot=True, fmt=".2f", square=1, cmap="RdBu_r", vmin=-1, vmax=1)
plt.title('Correlation heatmap')
plt.show()


# Let's also quickly check Age's relation to Survival and investigate our Embarked/Pclass theory. In the plots below, the size of the blue markers indicates the number of people in that group. Red dots are means of the groups on Y, per group on X. Green lines are the regression lines of the data.    
# <br>
# As we could already see in the histograms; if you were under 10 years old, you'd have a slightly higher chance on survival (above 50%). It was a bit harder to see what happens at the 60+ year level because there are so few. Turns out, seniors were the most likely to perish. All other decades are around 40% survival, and this includes by far the most people so it is also the overall survival rate.   
# <br>
# Regarding Embarked/Pclass, we can confirm our hypothesis that people embarking in Cherbourg were primarily 1st class. In Southampton and Queenstown the clear majority was 3rd class. Since 1st class passengers are much more likely to survive, this explains why people from Cherbourg were more likely to survive. The order of the factors causes a U shape relation which leads to the flat regression line. We could switch around the order here to form a better correlation, but it is not important. 
# 

# In[ ]:


fig, [ax1, ax2] = plt.subplots(1,2, figsize=[10,4])
x='AgeDec'
y='Survived'
df['Grpcount'] = df.groupby([x, y]).transform('count')['index'].values
df['Grpmean'] = df.groupby([x]).transform('mean')[y].values
sns.regplot(x, y, data=df, scatter_kws={'s': df['Grpcount']*7}, line_kws={'color':'C2'}, ax=ax1)
ax1.plot(df[x], df['Grpmean'], 'o', color='C3')
ax1.set_ylim([-0.15, 1.1])

x='Embarked'
y='Pclass'
df['Grpcount'] = df.groupby([x, y]).transform('count')['index'].values
df['Grpmean'] = df.groupby([x]).transform('mean')[y].values
sns.regplot(x, y, data=df, scatter_kws={'s': df['Grpcount']*7}, line_kws={'color':'C2'}, ax=ax2, color='C0')
ax2.plot(df[x], df['Grpmean'], 'o', color='C3')
ax2.set_ylim([0.7, 3.4])
ax2.set_xlim([-0.4, 2.3])
ax2.set_xticks(range(3))

fig.tight_layout()
sns.despine(bottom=0, left=0)
plt.show()


# ## Data analysis
# #### Machine learning: random forest
# We have cleaned the data, created some new features and filled in missing values. After that, we explored the variables to get a good sense of what is going on. Now we are ready to predict the missing survival values of 418 Titanic passengers.  
# <br>
# To classify survival, we will use the machine learning technique called Random Forest. The idea of a random forest is quite simple. It is a collection of many single decision trees. For every data sample, each tree predicts an outcome. The votes of all the trees are tallied up and the final prediction is whichever outcome was voted for the most.   
# <br>
# First, we split our dataframe back into the original train and test set that we downloaded. Then we fit the random forest to our train data.   
# <br>
# Just out of interest, let's also visualize one of the trees. Each tree is different from another because they are based on random subsets of the train data. How is a tree built? At each node, the goal is to minimize impurity/uncertainty. So, the data is split on some value of a feature and the resulting subsets should contain as much as possible of Survival=yes class and as little of Survival=no (the reverse for the other subset). To determine where to split the data, the Gini index is used. The Gini index essentially gives the probability of guessing the wrong class within the subsets created by splitting the data on that particular value. If the subsets are completely pure, you have Gini of 0. If they are both 50/50 Gini is 1.    
# <br>
# Open the image in a new tab to view it in readable resolution.
# 

# In[ ]:


# Split into train/test
feats = ['Sex', 'Age', 'AgeGrp', 'AgeDec', 'PersonType', 'Embarked', 'Pclass', 'Fare', 'Title',
         'Parch', 'SibSp', 'FamSize', 'FamSize2', 'GrpSize', 'GrpSize2', 'Deck', 'CabinLoc'] 

dff = df[['PassengerId', 'Set', 'Survived'] + feats]

train = dff[dff['Set'] == 'train'].drop('Set', 1)
test = dff[dff['Set'] == 'test'].drop('Set', 1)

trainx = train[feats].values
trainy = train['Survived'].values.astype(int)

testx = test[feats].values


# In[ ]:


# Fit random forest to the data
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=500, verbose=0, random_state=1)
clf.fit(trainx, trainy)


# In[ ]:


# Visualize a tree
import graphviz 
from sklearn import tree

dotdata = tree.export_graphviz(clf.estimators_[0], out_file=None, feature_names=feats, filled=True, 
                               rounded=True, class_names=True, special_characters=False, 
                               leaves_parallel=False)  
# graphviz.Source(dotdata)
# I uploaded a .png image instead, for viewing convenience


# 
#  ![tree]](http://i.imgur.com/LINsqUm.png)

# #### Confusion Matrix
# The first thing to look at for describing the performance of our model would be the confusion matrix. The primary measure of interest is the accuracy of the model. This tells us how often our model predicts the correct class. Quite good with 82%.  
# Other measures that are generally of interest are sensitivity and specificity. Sensitivity tells us how often the model is correct when the actual value is yes (is it "sensitive" enough to detect the disease, for example). Likewise, specificity tells us how often the model is correct when the actual value is no (a test can be perfectly "sensitive" by always predicting that a disease occurs, but that is not "specific" enough). 

# In[ ]:


# Plot confusion matrix
from sklearn.metrics import confusion_matrix

xtrain, xtest, ytrain, ytest = train_test_split(trainx, trainy, test_size=.2, random_state=0)
clf.fit(xtrain, ytrain)

confm = confusion_matrix(ytest, clf.predict(xtest))
confm = confm.astype(float)
#confm = confm/len(ytest)
sns.heatmap(confm, annot=True, fmt=".2f", square=1, cmap='Blues', vmin=0, vmax=100)

plt.title('Confusion matrix')
plt.xlabel('predicted survival')
plt.ylabel('actual survival')
plt.xticks([0.5,1.5],['no','yes'])
plt.yticks([0.5,1.5],['no','yes'])
plt.show()

print('Accuracy: ' + str((confm[0,0] + confm[1,1]) / np.sum(confm))) # (true neg + true pos) / total
print('Sensitivity: ' + str(confm[1,1] / np.sum(confm, 1)[1])) # true pos / actual pos
print('Specificity: ' + str(confm[0,0] / np.sum(confm, 1)[0])) # true neg / actual neg


# #### ROC Curve and AUC
# More info can be extracted from the confusion matrix and this is best plotted as the ROC curve. Namely, the false positive rate (fpr) and the true positive rate (tpr). The tpr is the same as sensitivity (if actually yes, does it predict yes?), and the fpr is simply 1-specificity (if actually no, does it predict yes?).   
# 
# Imagine two, somewhat overlapping, normal distributions, one for actually positive and negative samples. You have to set a classifying threshold between the distributions: all samples above the threshold are classified as positive by your model, below the threshold are negative. Most naturally you would make the distinction where the distributions meet. Because they overlap, some samples end up at the wrong side of the threshold thus getting classified wrongly. This way, the fpr and tpr are balanced. However, perhaps it is very important to e.g. minimize false positives. You could then shift the threshold beyond the distribution with negative samples. No false positives, at the cost of less true positives. The ROC curve shows the tpr/fpr balance of many thresholds.    
# 
# The diagonal line is where the tpr and fpr are equal, meaning it's just 50/50 guessing. The further towards the top left corner the ROC curve goes, the better the classifier is. This can be summarized in one number; the area under the curve (AUC). AUC of 1 is perfect, and 0.5 means pure guessing.   
# 
# In our case: the ROC looks good and the AUC is 0.87. This means our model is actually pretty good at discriminating between survivors and victims. While AUC and accuracy are numbers with similar meaning, AUC is generally better because accuracy can be misleading in the case of very imbalanced true positives and true negatives). 
# 

# In[ ]:


# Plot ROC Curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

xtrain, xtest, ytrain, ytest = train_test_split(trainx, trainy, test_size=.2, random_state=0)
clf.fit(xtrain, ytrain)

fpr, tpr, th = roc_curve(ytest, clf.predict_proba(xtest)[:,1])

plt.figure(figsize=[4,4])
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')

plt.title('ROC curve')
plt.xlabel('false pos rate')
plt.ylabel('true pos rate')
sns.despine(left=False, bottom=False)
plt.show()

print('AUC: ' + str(roc_auc_score(ytest, clf.predict_proba(xtest)[:,1])))


# #### Learning Curves
# In order to see how the number of samples affect the model, we can create learning curves. Ideally, the curve of the train set starts with a fair amount of error, which decreases as the number of training samples increases until the error plateaus. To test the generalization of the model, it is cross validated with a test set. With low number of samples you would want a high error, which subsequently converges towards the training curve. 
# 
# Our train set has a near perfect score already with a small training set. The test set does not have a great score and is not really improving with increasing samples. This is a clear-cut case of overfitting: the training set gets a very high score, but it does not generalize well to the test set.

# In[ ]:


# Get learning curve data (can take minutes)
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    clf, trainx, trainy, train_sizes = np.linspace(0.1, 1.0, 10), cv=5, verbose=0) #10 steps, cv=5


# In[ ]:


# Plot the learning curves
plt.figure(figsize=[4,3])
sns.tsplot(test_scores.transpose(), time=train_sizes, ci=95, color='C1', marker='o')
sns.tsplot(train_scores.transpose(), time=train_sizes, ci=95, color='C0', marker='o')
plt.ylim((0.6, 1.01))
plt.gca().invert_yaxis()
plt.legend(['test','train'])
leg = plt.gca().get_legend()
leg.legendHandles[0].set_alpha(1)
leg.legendHandles[1].set_alpha(1)
plt.xlabel('N of samples')
plt.ylabel('Score')
plt.title('Learning curves')
sns.despine(left=False, bottom=False)
plt.show()


# #### Feature importance
# Feature importances are a very interesting aspect of random forests. We can check which features contribute the most to our model. The decrease in impurity after splitting on each feature is averaged over all trees in the forest. We then plot them, preferably sorted, to easily compare the features.   
# In our model, Age and Fare are clearly very important features. Title, PersonType and Sex are also important, but they have huge standard deviations. This is because all three of them essentially split on male/female. This leads to one of them being the root node in the tree, leaving the other 2 with very low importance. The remaining features are not all that useful anymore. Perhaps GrpSize is meaningful since it is not explained by Age/Fare/Sex. 

# In[ ]:


# Plot feature importances
featstats = pd.DataFrame(feats, columns=['featnames'])
featstats['featimp'] = clf.feature_importances_
featstats['featstd'] = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
featstats = featstats.sort_values('featimp', ascending=True)

plt.figure(figsize=[8,6])
xerr=featstats['featstd']
plt.barh(range(len(featstats)), featstats['featimp'], color=sns.color_palette()[0], xerr=xerr)
plt.yticks(range(len(featstats)), featstats['featnames'])

plt.xlabel('Decrease in impurity')
plt.title('Feature importances')
sns.despine(left=False, bottom=False)
plt.show()


# ## Exporting survival predictions
# All there is left to do now, is exporting our predictions for the unknown Survival data. A dataframe is properly formatted according to Kaggle's standards and then saved as a .csv file. 

# In[ ]:


# Set proper format and export for kaggle submission
result = pd.DataFrame(index=test['PassengerId'])
result['Survived'] = clf.predict(testx)

#result.to_csv('prediction.csv')


# ## Results
# After submitting to Kaggle, our model predicted 76.5% of the unknown set correct. Not bad! Pure guessing would be 50%. Simply setting all to 'not survived' would have lead to about 62% correct because 62% in the known set died. Our 76.5% is significantly better than that, but also quite a bit less than expected from the AUC (87%). Since so many of the features were related to each other, a simpler model could do just as good, if not better. I quickly ran the analysis with only the features Title, Fare, Age and GrpSize. The AUC was similar (85%) but the Kaggle score was a tiny bit improved to 77%. Feature importances of Title, Fare and Age (in that order) were similar, with GrpSize a bit behind. 
# 
# 

# 
