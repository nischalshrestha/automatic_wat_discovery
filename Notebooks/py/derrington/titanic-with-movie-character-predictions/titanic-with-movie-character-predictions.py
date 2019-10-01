#!/usr/bin/env python
# coding: utf-8

# # Titanic Survivor Prediction (with additional predictions on movie characters)
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# <br>
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# ![](https://media1.popsugar-assets.com/files/thumbor/EPf-QLPGGdbYGlKExZ3gIKLc1IU/fit-in/1024x1024/filters:format_auto-!!-:strip_icc-!!-/2014/10/03/806/n/1922283/5d8c153f0f488cb0_anigif_enhanced-buzz-18158-1381217382-36/i/When-He-Tries-Keep-Things-Light-While-Freezing-Death.gif)
# <br>
# In this challenge, we are asked to complete the analysis of what sorts of people were likely to survive. In particular, we are asked to apply machine learning to predict which passengers survived the tragedy.

# # Part 1: Load packages and first glimpse of dataset

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


# Load the dataset

train_data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")
test_data_PassengerId = pd.read_csv("../input/titanic/test.csv")
movie_data = pd.read_csv("../input/movie-test-with-dummies/movie_test.csv")


# ### First inspection of training set
# Looking at the first and last 10 rows below gives us a flavour of the dataset.
# We will try to predict the categorical variable in the "Survived" column using the data from other columns.

# In[3]:


# Look at the first few rows to get an idea of what data we are working with
train_data.head(10)


# In[4]:


train_data.tail(10)


# ### Basic Stats
# Get some basic statistical info on those features that are numerical. The categorical features will be dealt with later.

# In[5]:


# Let's get some basic statistical properties for the training set (for numerical columns)
train_data.describe()


# Some observations:
# * There is data on 891 passengers to train on. About 40% of the total passenger list (2,224 people)
# * Of the 891 passengers to train on around 38% survived. A little higher than overall survival rate (32%)
# * The mean age was 29 years. The upper quartile was 38 and the maximum was 80 meaning very few passengers were very old.
# * SibSp gives a score of 1 for every sibling or spouse present on board
# * Parch gives a score of 1 for every parent or child on board (note some children travelled with nannies and thus score 0)
# * The mean fare was \$32. The upper quartile was \$31 and the max was \$512 meaning very few passengers paid these very high fares.

# # Part 2: Investigate effect of individual features on survival rates

# ## Feature 1: Age Analysis
# There are 177 passengers for which the age is unkown. This appears as "Nan" in the original dataset. Since dropping all of these ($\sim 20\%$ of entire training set) is unacceptable we will need to come up with a clever means of filling them:
# *  We replacr these unkown ages by randomly generated ages in the range $[\mu_{\text{age}} - 0.5\sigma_{\text{age}}, \mu_{\text{age}} + 0.5\sigma_{\text{age}}]$. Note that half a standard deviation either side of the mean provided an improvement in accuracy over a full standard deviation. 
# *  We show a histogram of passenger numbers by age before and after the journey (for both original and preprocessed data)
# <br>
# <br>
# Then for preprocessed data only:
# * We show a distribution of survival rates by age
# * We show a second histogram of survival numbers by age but now with a heat map

# In[6]:


print("Number of passengers with unknown age: {}".format(train_data["Age"].isnull().sum()))


# In[7]:


# Deal with age first. Idea is to find mean and std and fill Nan values with random numbers in range [mean-std, mean+std]

# Get mean and std of age as well as number of Nan values for train set
train_age_mean = train_data["Age"].mean()
train_age_std = train_data["Age"].std()
train_age_nan = train_data["Age"].isnull().sum()

# Get mean and std of age as well as number of Nan values for train set
test_age_mean = test_data["Age"].mean()
test_age_std = test_data["Age"].std()
test_age_nan = test_data["Age"].isnull().sum()

# Generate enough random numbers in range [mean-0.5*std, mean+0.5*std] for train set
train_age_rand = np.random.randint(train_age_mean - 0.5*train_age_std, train_age_mean + 0.5*train_age_std, size = train_age_nan)

# Generate enough random numbers in range [mean-0.5*std, mean+0.5*std] for test set
test_age_rand = np.random.randint(test_age_mean - 0.5*test_age_std, test_age_mean + 0.5*test_age_std, size = test_age_nan)

# Create a figure for plotting original and preprocessed age data


# Original age data (simply don't plot Nan values)
plt.figure(figsize=(15,7))
#train_data['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
plt.subplot(1,2,1)
plt.style.use('bmh')
plt.xlabel('Age (original data)')
plt.ylabel('Survived')
plt.title('Original Age vs Survival')
plt.hist(train_data.Age[(np.isnan(train_data.Age) == False)], bins= 15, alpha = 0.4, color = 'r', label = 'Before')
plt.hist(train_data.Age[(np.isnan(train_data.Age) == False) & (train_data.Survived == 1)], bins= 15, alpha = 0.4, color = 'b', label = 'After')
#plt.hist(data.Age[data.Age != np.NaN])
plt.legend(loc = 'upper right')


# Preprocessed age data (first fill the Nan values with the random numbers and then plot them)
train_data.loc[train_data.Age.isnull(), 'Age'] = train_age_rand
test_data.loc[test_data.Age.isnull(), 'Age'] = test_age_rand


#train_data["Age"].hist(bins=70, ax = axis2)
plt.subplot(1,2,2)
plt.style.use('bmh')
plt.xlabel('Age (preprocessed data)')
plt.ylabel('Survived')
plt.title('Preprocessed Age vs Survival')
plt.hist(train_data.Age, bins= 15, alpha = 0.4, color = 'r', label = 'Before')
plt.hist(train_data.Age[(train_data.Survived == 1)], bins= 15, alpha = 0.4, color = 'b', label = 'After')
plt.legend(loc = 'upper right')
plt.tight_layout()
plt.show()



# In[8]:


# Using only the preprocessed data, we can further investigate the relationship betweena ge and survival
# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()

# number of survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(40,4))
survival_number_by_age = train_data[["Age", "Survived"]].groupby(['Age'],as_index=False).sum()
sns.barplot(x='Age', y='Survived', data=survival_number_by_age)


# It looks like survival rates were highest for those in their twenties and thirties. There is another, smaller, peak for the very young but sever drop off for the old.
# <br>
# One thing to be aware of is the technique used to fill in the missing data which may be responsible for the large peaks on both survival and death rates around the mean age.

# ## Dealing with remaining missing data 
# We have already dealt with missing ages which affected a significant amount of our dataset (this has been done for both the training and the test set). 
# <br>
# We will next drop various bits of data from both the training and test sets if they are not going to be useful:
# <br>
# ### Missing Embarked data
# There are 2 missing entries in Embarked column. To avoid these entire rows being dropped we'll fill them with S (Southampton) as this is most common entry.
# <br>
# ### The PassengerId and Ticket Number columns can be dropped
# This does not offer any information that will be useful for making our prediction
# <br>
# ### The Cabin column can be dropped
# Actually this column does offer useful information i.e. which deck of the ship the passenger was on. However, there are 687 passengers for which we don't konw their Cabin location. Therefore I have decided to drop this as well since I couldn't think of a good way to fill missing values.
# 
# ### Drop remaining Nan values
# We can also drop any remaining data entries (rows) that have missing values for any of the features

# In[9]:


# Fill missing Embarked data in training set to avoid entire rows being dropped
train_data.loc[train_data.Embarked.isnull(), 'Embarked'] = 'S'

# Drop PassengerId, Name and Ticket Number from training set 
train_data = train_data.drop(["PassengerId", "Cabin", "Ticket"], axis = 1)

# Drop remaining Nan values
train_data = train_data.dropna() 


# In[10]:


# There is one passenger in test set with missing Fare data. Below, we identify him.
test_data.loc[test_data.Fare.isnull()]


# In[11]:


# As this Passenger is travelling in 3rd class, it will be sensible to fill his Fare data with th emean fare paid by other third class passengers. 
test_3class = test_data.loc[test_data['Pclass'] == 1]
mean_fare = test_3class[["Fare"]].mean()
test_data.ix[152, 'Fare'] = mean_fare[0]
test_data.ix[152]

# Drop PassengerId, Name and Ticket Number from test set 
test_data = test_data.drop(["PassengerId", "Cabin", "Ticket"], axis = 1)


# To summarise, we have done the following to our training set:
# * Replaced missing ages with randomly generated ages (taken from a suitable range) 
# * Removed useless features
# * Removed remaining missing entries
# <br>
# <br>
# And we have done the following to our test set:
# * Replaced missing fare with mean of all fares paid by other passengers travelling in the same class
# * Removed useless features
# <br>
# <br>
# Now let's examine the nice clean dataset:

# In[12]:


train_data.head(20)


# ## Feature 2: Passenger Class Analysis
# We have Passenger Class (PClass) data for all passengers. Therefore we can go straight to plotting the data.
# * We show the distribution of passenger classes by age
# * We show the survival numbers by class
# * We show the survival rates by class (along with a confidence interval)

# In[13]:


# First compare class to age
fig = sns.FacetGrid(train_data,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade='True')
oldest = train_data['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

# Then a simple bar graph of survived passengers by class
fig, axis1 = plt.subplots(1,1,figsize=(3,4))
survival_number_by_class = train_data[["Pclass", "Survived"]].groupby(['Pclass'],as_index=False).sum()
sns.barplot(x='Pclass', y='Survived', data=survival_number_by_class)

sns.factorplot('Pclass','Survived',data=train_data)


# The first graph shows that those in 1st class were generally a little older than those in second and third.
# <br>
# We see in the second graph that the number of survivors was highest for 1st class, followed by 3rd class and finally 2nd class.
# <br>
# However, the most interesting graph is the third one, this shows the percentage of passengers that survived from each of the three classes. It is much, much higher for 1st class than 2nd class, which is itself much, much higher than 3rd class. In particular, the survival rate for third class passengers is staggeringly low at $\sim 25\%$.

# ## Feature 3: Gender Analysis
# We have Gender (Sex) data for all passengers. Therefore we can go straight to plotting the data.
# * We show the distribution of genders by age
# * We show the survival numbers by gender
# * We show the survival numers and survival rate by gender

# In[14]:


# First compare gender to age
fig = sns.FacetGrid(train_data,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade='True')
oldest = train_data['Age'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()

# Then a simple bar graph of survived passengers by class
fig, axis1 = plt.subplots(1,1,figsize=(3,4))
survival_number_by_class = train_data[["Sex", "Survived"]].groupby(['Sex'],as_index=False).sum()
sns.barplot(x='Sex', y='Survived', data=survival_number_by_class)

fig = plt.figure(figsize=(30,4))

#create a plot of two subsets, male and female, of the survived variable.
#After we do that we call value_counts() so it can be easily plotted as a bar graph. 
#'barh' is just a horizontal bar graph
df_male = train_data.Survived[train_data.Sex == 'male'].value_counts().sort_index()
df_female = train_data.Survived[train_data.Sex == 'female'].value_counts().sort_index()


ax1 = fig.add_subplot(141)
df_male.plot(kind='barh',label='Male', color = 'blue', alpha=0.9)
plt.title("Male Survival (raw) "); plt.legend(loc='best')
 

#adjust graph to display the proportions of survival by gender
ax2 = fig.add_subplot(142)
(df_male/float(df_male.sum())).plot(kind='barh',label='Male', color = 'blue', alpha=0.9)  
plt.title("Male survival (proportional)"); plt.legend(loc='best')

ax3 = fig.add_subplot(143)
df_female.plot(kind='barh', color='#FA2379',label='Female', alpha=0.9)
plt.title("Female surivival (raw)"); plt.legend(loc='best')

ax4 = fig.add_subplot(144)
(df_female/float(df_female.sum())).plot(kind='barh',color='#FA2379',label='Female', alpha=0.9)
plt.title("Female survival (proportional)"); plt.legend(loc='best')

plt.tight_layout()


# Female survival exceeds male survival both in terms of raw numbers and by the survival percentage. Indeed, $>70\%$ of women survived compared to $<20\%$ of men. The "Women and children first policy" for the lifeboats presumably explains this.
# 

# ## Feature 4: Family Members Analysis
# We now examine how the presence of family members affected survivavl rates. To do this we will use a total family score of FamScore $=$ SibSp $+$ Parch, i.e. number of siblings, spouses, parents and children accompanying them. Note that our predictive models will continue to treat SibSp and Parch separately - combining them is just to help gauge both features at once.

# In[15]:


family_data = train_data[['SibSp', 'Parch', 'Survived']].copy()
family_data['FamScore'] = family_data['SibSp'] + family_data['Parch']
family_data.drop(['SibSp','Parch'], axis=1)
columnsTitles=["FamScore","Survived"]
family_data=family_data.reindex(columns=columnsTitles)

# Get distribution of family sizes
family_data.hist('FamScore')

#Transform a nonzero FamScore to "With Family" and a zero FamScore to "Alone"
family_data['FamScore'].loc[family_data['FamScore']>0] = "With Family"
family_data['FamScore'].loc[family_data['FamScore']==0] = "Alone"
sns.factorplot('FamScore',data=family_data,kind='count',palette='Blues')

fig = plt.figure(figsize=(30,4))
#create a plot of two subsets, with family and alone, of the survived variable.
#After we do that we call value_counts() so it can be easily plotted as a bar graph. 
#'barh' is just a horizontal bar graph
df_fam = family_data.Survived[family_data.FamScore == 'With Family'].value_counts().sort_index()
df_alone = family_data.Survived[family_data.FamScore == 'Alone'].value_counts().sort_index()


ax1 = fig.add_subplot(141)
df_fam.plot(kind='barh',label='With Family', color = 'orange', alpha=0.5)
plt.title("With Family Survival (raw) "); plt.legend(loc='best')
 

#adjust graph to display the proportions of survival by gender
ax2 = fig.add_subplot(142)
(df_fam/float(df_fam.sum())).plot(kind='barh',label='With Family', color = 'orange', alpha=0.5)  
plt.title("With Family survival (proportional)"); plt.legend(loc='best')

ax3 = fig.add_subplot(143)
df_alone.plot(kind='barh', color='green',label='Alone', alpha=0.5)
plt.title("Alone surivival (raw)"); plt.legend(loc='best')

ax4 = fig.add_subplot(144)
(df_alone/float(df_alone.sum())).plot(kind='barh',color='green',label='Alone', alpha=0.5)
plt.title("Alone survival (proportional)"); plt.legend(loc='best')

plt.tight_layout()


# The above shows that there were $\sim350$ passengers travelling with family. These passengers had a survival rate of above $50\%$.
# <br>
# Meanwhile, there were in excess of $500$ passengers travelling without family members for whom, rather alarmingly, the survival rate was just $\sim 30\%$. 
# <br>
# <br>
# We can now understand Jack's reaction to Rose reboarding from the lifeboat without first tossing her mother back on board: ![](http://www.blog.urbanoutfitters.com/files/tumblr_mgw445XA221r4gtljo1_500.gif)

# ## Feature 5: Fare Analysis
# We should next examine the impact ticket price had on survival rate

# In[16]:


fig = sns.FacetGrid(train_data,hue='Survived',aspect=4,size=5)
fig.map(sns.kdeplot,'Fare',shade='True')
oldest = train_data['Fare'].max()
fig.set(xlim=(0,oldest))
fig.add_legend()


# The above survival distribution shows that those with the lowest price tickets were substantially more likely to die in the disaster.

# ## Feature 6: Embarkation Point
# It will be interesting to see whether the embarkation point (C = Cherbourg, Q = Queenstown, S = Southampton) has any impact on survival rate. We plot:
# * the number of passengers boarding in each location
# * the survival rates for each boarding location

# In[17]:


train_data.Embarked.value_counts().plot(kind='bar', figsize=(5,5))
# specifies the parameters of our graphs
plt.title("Passengers per boarding location")

fig = plt.figure(figsize=(30,4))
#create a plot of two subsets, with family and alone, of the survived variable.
#After we do that we call value_counts() so it can be easily plotted as a bar graph. 
#'barh' is just a horizontal bar graph
df_C = train_data.Survived[train_data.Embarked == 'C'].value_counts().sort_index()
df_Q = train_data.Survived[train_data.Embarked == 'Q'].value_counts().sort_index()
df_S = train_data.Survived[train_data.Embarked == 'S'].value_counts().sort_index()

ax1 = fig.add_subplot(161)
df_C.plot(kind='barh',label='Cherbourg', color = '#377eb8', alpha=0.6)
plt.title("Cherbourg (raw) "); plt.legend(loc='best')
 

#adjust graph to display the proportions of survival by gender
ax2 = fig.add_subplot(162)
(df_C/float(df_C.sum())).plot(kind='barh',label='Cherbourg', color = '#377eb8', alpha=0.6)  
plt.title("Cherbourg (proportional)"); plt.legend(loc='best')

ax3 = fig.add_subplot(163)
df_Q.plot(kind='barh', color='#4daf4a',label='Queenstown', alpha=0.6)
plt.title("Queenstown (raw)"); plt.legend(loc='best')

ax4 = fig.add_subplot(164)
(df_Q/float(df_Q.sum())).plot(kind='barh',color='#4daf4a',label='Queenstown', alpha=0.6)
plt.title("Queenstown (proportional)"); plt.legend(loc='best')

ax5 = fig.add_subplot(165)
df_S.plot(kind='barh', color='#e41a1c',label='Southampton', alpha=0.6)
plt.title("Southampton (raw)"); plt.legend(loc='best')

ax6 = fig.add_subplot(166)
(df_S/float(df_S.sum())).plot(kind='barh',color='#e41a1c',label='Southampton', alpha=0.6)
plt.title("Southampton (proportional)"); plt.legend(loc='best')

plt.tight_layout()


# Above we can see that the vast majority of passengers boarded in Southampton. The survival rates were:
# * $\sim 30\%$ for Cherbourg
# * $\sim 38\%$ for Queenstown
# * $\sim 34\%$ for Southampton
# 
# As expected, boarding location doesn't appear to play a huge role in determining survival rates and we would expect our classification model to "damp" out its effect.

# # Part 3: Classification Models
# Having cleaned our training and test datasets and played around with the training data to get an idea of the importance of each of our features, we will now build a predictive model.
# <br> 
# This model will train on the feature set $(\text{Pclass}, \text{Name}, \text{Sex}, \text{Age}, \text{SibSp}, \text{Parch}, \text{Fare}, \text{Embarked})$ to find a model that accuractely predicts survival.
# <br>
# We will test the accuracy of a variety of models using our test set (recall it has also been preprocessed)
# <br>
# Ultimately we would like to be able to feed it novel data and get a prediction.

# ### Prepare dataset for models
# First split into independent and dependent variables, then one hot encode the categorical features and perform feature scaling on all features. Lastly we split the training set to allow hold-out validation and cross-validation to be applied. 
# <br>
# We will also need to one hot encode and feature scale the test data.
# <br>
# For reference:
# * Categorical data must be converted to numerical data in order to run modelling. One hot encoding first assigns a number to each different category. However, this leads to a problem e.g. consider our different Embarkation points $\{C,Q,S\}$. If we assign $C=0, Q=1, S=2$, then the model will believe that $S>Q>C$ which there is no evidence to support. To avoid this, we must introduce "dummy variables". Essentially we replace the "Embarkation" column by three new columns "Embarkation_C", "Embarkation_Q" and "Embarkation_S". We can then assign a binary response to each column and the model will proceed without misinterpreting any categorical variable as being more valuable than the others.
# * After dropping the Name variable, we now have entirely numerical data. However, since most machine learning equations are based on Euclidean distances between points $\mathbb{R}^n$ space, we will need to perform feature scaling to avoid certain features e.g. Fare dominating the others since they are on a completely different scale. This would effectively lead to the regression being preformed only with respect to the Fare data which is highly undesirable as we want to learn from **all** the data we have. Feature scaling can be done by standardising the variables i.e. $\frac{x-\mu_X}{\sigma_X}$ or by the normalisation procedure i.e. $\frac{x-\text{min}(X)}{\text{max}(X)-\text{min}(X)}$. We will standardise it below to bring all variables into the range $[-1,+1]$.

# In[18]:


# First split data into independent and dependent variables
y = train_data['Survived'].copy()
X = train_data.drop(["Survived"], axis = 1)

# Deal with categorical features
# Note the dependent variable (Survived) is already binary and therefore the only things that need to be encoded are Sex and Embarked.
X_hot = pd.get_dummies(X, prefix=['Sex', 'Embarked'], columns=['Sex', 'Embarked'])
X_hot = X_hot.drop('Name',axis=1)
X_hot[:10]

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_hot = sc_X.fit_transform(X_hot)

# Split training data to allow us to perform hold-out validation and cross-validation
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_hot, y, test_size = 0.1, random_state = 0)

# Do the same for test data as we'll need it for making final predictions (we don't need to split the test data though)
test_hot = pd.get_dummies(test_data, prefix=['Sex', 'Embarked'], columns=['Sex', 'Embarked'])
test_hot = test_hot.drop('Name',axis=1)
test_hot = sc_X.fit_transform(test_hot)


# ### Prepare a function to allow quick testing of models

# In[19]:


# Generic code to run on all of our different models
from sklearn.model_selection import GridSearchCV, cross_val_score
def train_test_model(model, hyperparameters, X_train, X_test, y_train, y_test, folds = 5):
    """
    Given a [model] and a set of possible [hyperparameters], an exhaustive search is performed across all possible hyperparameter values. The optimum model is returned.
    We then print out some useful info.
    """
    optimized_model = GridSearchCV(model, hyperparameters, cv = folds, n_jobs = -1)
    optimized_model.fit(X_train, y_train)
    y_pred = optimized_model.predict(X_valid)
    print('Optimized parameters: {}'.format(optimized_model.best_params_))
    print('Model accuracy (hold-out validation): {:.2f}%'.format(optimized_model.score(X_test, y_test)*100))
    # Take our best model and run it on different train/valid splits and take the mean accuracy score. n_jobs=-1 allows all CPU corees to be used.
    kfold_score = np.mean(cross_val_score(
            optimized_model.best_estimator_, np.append(X_train, X_test, axis = 0), 
            np.append(y_train, y_test), cv = folds, n_jobs = -1))
    print('Model accuracy ({}-fold cross validation): {:.2f}%'.format(folds, kfold_score*100))
    return optimized_model


# ### Quick Check for Multicollinearity
# We can investigate multicollinearity using a heat map:

# In[20]:


fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(X.corr(),cmap="YlGnBu")


# Logistic Regression is a natural starting point given that our dependent variable is categorical. We need to be wary of collinearity for logistic regression. This is because multicollinearity poses problems in getting precise estimates of the coefficients corresponding to particular variables. With a set of collinear variables all related to survival status, it's hard to know exactly how much credit each of them should get individually. Other classifiers like decision stumps don't have this problem. 
# <br>
# **Caveat:** If you don't care about how much credit to give to each feature, it is possible for them to work very well together for prediction. 
# <br>
# The heat map indicates the only potentially dangerous correlation is between SibSp (Siblings & Spouses) and Parch (Parents & Children). As we aren't concerned what the actual regression coefficients are, we are probably safe to proceed without removing either of these features (generally speaking accuracy will decrease if you remove predictor variables). The above collinearity was only mentioned for completeness.
# <br>
# As an aside, at first glance it was surprising to me that Fare and Pclass have a low negative correlation. Re-examining thetraining set, there does appear to be a wide range of prices paid for tickets of different classes and so it appears to make sense.

# ## Model 1: Logistic Regression

# In[21]:


get_ipython().run_cell_magic(u'time', u'', u"from sklearn import linear_model\nlr_model = train_test_model(linear_model.LogisticRegression(random_state = 0), {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'class_weight': [None, 'balanced']}, X_train, X_valid, y_train, y_valid)")


# ## Model 2: Decision Tree 

# In[22]:


get_ipython().run_cell_magic(u'time', u'', u"from sklearn.tree import DecisionTreeClassifier\n# Use our function train_test_model with a variety of values for the hyperparameter C (inverse regularization strength) and class_weight (different weights can be given to different features).\ndt_model = train_test_model(DecisionTreeClassifier(random_state = 0), {'min_samples_split': [2, 4, 8, 16], 'min_samples_leaf': [1, 3, 5, 10], 'max_depth': [2,3,4,5, None], 'class_weight': [None, 'balanced']}, X_train, X_valid, y_train, y_valid)")


# ## Model 3: Random Forest

# In[23]:


get_ipython().run_cell_magic(u'time', u'', u"from sklearn.ensemble import RandomForestClassifier\n# Use our function train_test_model with a variety of values for the hyperparameter C (inverse regularization strength) and class_weight (different weights can be given to different features).\nrf_model = train_test_model(RandomForestClassifier(random_state = 0), {'min_samples_split': [2, 4, 8, 16], 'min_samples_leaf': [1, 3, 5, 10], 'max_depth': [3, None], 'class_weight': [None, 'balanced']}, X_train, X_valid, y_train, y_valid)")


# ## Model 4: Support Vector Machine (SVM)

# In[24]:


get_ipython().run_cell_magic(u'time', u'', u"from sklearn.svm import SVC\n# Use our function train_test_model with a variety of values for the hyperparameter C (inverse regularization strength) and class_weight (different weights can be given to different features).\nsvc_model = train_test_model(SVC(random_state = 0), {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'gamma': np.logspace(-9, 3, 13), 'kernel': ['rbf','linear']}, X_train, X_valid, y_train, y_valid)")


# ## Model 5: K-Nearest Neighbours (KNN)

# In[25]:


get_ipython().run_cell_magic(u'time', u'', u"from sklearn.neighbors import KNeighborsClassifier\n# Use our function train_test_model with a variety of values for the hyperparameter C (inverse regularization strength) and class_weight (different weights can be given to different features).\nknn_model = train_test_model(KNeighborsClassifier(), {'n_neighbors': [1,3,5,7,9,11,13,15,17,19,21,23,25]}, X_train, X_valid, y_train, y_valid)")


# ## Model 6: Neural Network (Keras)

# In[26]:


get_ipython().run_cell_magic(u'time', u'', u'import keras\n# This is required to initialise our ANN\nfrom keras.models import Sequential \n# This is required to build the layers of our ANN\nfrom keras.layers import Dense \n# We initialise the ANN by building an object of the sequential class and then add layes below.\nclassifier = Sequential() \n# Add the hidden layers\nclassifier.add(Dense(input_dim = 10, activation = \'relu\', units = 6, kernel_initializer = \'uniform\')) \nclassifier.add(Dense(activation = \'tanh\', units = 6, kernel_initializer = \'uniform\'))\nclassifier.add(Dense(activation = \'relu\', units = 6, kernel_initializer = \'uniform\'))\nclassifier.add(Dense(activation = \'tanh\', units = 6, kernel_initializer = \'uniform\'))\nclassifier.add(Dense(activation = \'sigmoid\', units = 6, kernel_initializer = \'uniform\'))\n# Add the output layer\nclassifier.add(Dense(activation = \'sigmoid\', units = 1, kernel_initializer = \'uniform\'))\n# Tell the ANN which loss function to use when applying stochastic gradient descent to find optimal weights\nclassifier.compile(optimizer = \'adam\', loss = \'binary_crossentropy\', metrics = [\'accuracy\'])\n# Fit classifier to training data\nclassifier.fit(X_train, y_train, batch_size = 10, epochs = 100)\n# Make prediction on validation set\ny_pred = classifier.predict(X_valid)\ny_pred = (y_pred > 0.5) \n\n# Making the Confusion Matrix\nfrom sklearn.metrics import confusion_matrix\ncm = confusion_matrix(y_valid, y_pred)\ncorrect = cm[0][0]+cm[1][1]\nwrong = cm[1][0]+cm[0][1]\naccuracy=(correct/(correct+wrong))*100\nprint("Model accuracy: {:.2f}".format(accuracy))')


# **Important:** The above neural network has somewhat confusing output. Throughout the majority of the training epochs, it claims to have an accuracy between $%83\%$ and $85\%$ which would put it in contention for our best model. Yet, when we calculate the Model accuracy directly at the end we get a dissapointing result of $\sim 78\$.
# This is because the score method used in keras does not calculate accuracy like the sklearn's accuracy_score method. The actual classification accuracy we are interested in will be the Model accuracy coming from the confusion matrix calculation.
# <br>
# This is probably to be expected since our data is linearly separable and so strictly speaking we shouldn't need any any hidden layers at all. Indeed, the evidence above seems to support the fact that for this dataset we don't need an neural entwork to resolve our data (although it will still do the job). 

# # Part 4: Submission
# The Decision Tree Classifier and K-Nearest Neighbour Classifier both achieved$84.44\%$ accuracy in the confusion matrix. However, the Decision Tree Classifier was higher across $5$-fold cross-validation than the K-Nearest Neighbour Classifier ($82.50\%$ vs. $81.03\%$) and so we will use the Decision Tree Classifier for our submission.

# In[27]:


test_hot.shape


# In[67]:


# Call dt_model which loads the Decision Tree Classifier (with optimal hyperparameters) and use it to make predictions on the test set.
best_model = classifier
test_pred = best_model.predict(test_hot)
#test_pred = (test_pred > 0.5)
for i in range(len(test_pred)):
    if test_pred[i] < 0.5:
        test_pred[i] = int(0)
    else:
        test_pred[i] = int(1)
test_pred = test_pred.astype(np.int64)
test_pred = test_pred.T
test_pred = test_pred.reshape(1,418)
test_pred = test_pred[0]
print(test_pred)


# In[68]:


submission = pd.DataFrame({
        "PassengerId": test_data_PassengerId["PassengerId"],
        "Survived": test_pred
    })
submission.to_csv('titanic.csv', index=False)


# In[30]:


submission.head()


# # Part 4: Predicting on Movie Characters
# I created my own dataset with five characters. These include:
# * Jack (age sourced from actor's age in 1997 and fare = $\$0$ since he won the ticket)
# * Rose (age sourced from Wikipedie article on Titanic movie and fare calculate as being 2 standard deviations above the mean of all other 1st class passengers i.e. $94.280297 + 2 \times 84.435858 \simeq \$263.15$)
# * Calvin (age sourced from actor's age in 1997 and fare calculated in same way as for Rose)
# * Because all three of the above characters boarded in Southampton, my one hot encoding only needed 8 rows to create all necessary dummy variables. This meant it wasn't the right size to be fed into the classification models trained above. Consequently, I have added two additional characters, "Jimbo" and "Jimbo Two", with random values for all categories except embarked where I made sure to give one of them a "Q" and one of them a "C" so that the one hot encoded movie data would take up 10 columns and be compatible with our classifiers above. We do not care about the fate of Jimbo or Jimbo Two in our summary below....

# In[31]:


# Explore the movie dataset
movie_data


# In[32]:


movie_data = movie_data.drop(["PassengerId", "Cabin", "Ticket"], axis = 1)
movie_hot = pd.get_dummies(movie_data, prefix=['Sex', 'Embarked'], columns=['Sex', 'Embarked'])
movie_hot = movie_hot.drop('Name',axis=1)
movie_hot = sc_X.fit_transform(movie_hot)
movie_hot


# In[33]:


movie_hot.shape


# In[34]:


best_model = dt_model
movie_pred = best_model.predict(movie_hot)
Names = ['Jack','Rose','Calvin']
Dead = ['die', 'survive to tell the tale to Paramount Pictures']
for i in range(3):
    print("{} will {}".format(Names[i], Dead[movie_pred[i]]))


# # Part 5: Conclusion
# Thanks for reading. I'm new to Kaggle so welcome any feedback.

# In[ ]:




