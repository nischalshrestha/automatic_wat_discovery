#!/usr/bin/env python
# coding: utf-8

# **The titanic dataset is primarily targetted at curious minds and aspiring data scientists.** Unsurprisingly, this was also how I started, and this kernel was designed to support a datascience introductory workshop.
# 
# In this kernel I am not trying to reach the best absolute prediction score, by magically chosing the right parameters.
# Instead, I am trying to take a simple yet rational approach, in order to illustrate what data science might look like, and potentially teach transferable skills. This is my first kernel though, so feedback welcomed!
# 
# If you are new to python, don't worry if you don't understand all the syntax - you will understand most of it if you follow most 10-20h courses. What I believe you should seek to understand and develop is critical thinking, and the questions and steps to try to answer them.

# # **Structure: a thinking process**
# In my worklife I usually build structured presentations, as this often is the most efficient way to associate complex pieces of information to influence decision making. However, this rarely reflects how insights were actually extracted. In this kernel I've wanted to explore a different structure, more similar to a thought process.
# 
# --------------------
# 1. quick and dirty model to understand how hard the problem is.
# 2. have a closer look at the data and create new features that have plausible causality.
# 3. throw lots of features to a few models and see how they perform. Look at the learning rate to diagnose bias vs variance.
# 3. select a few features that both perform and intuitively make sense. Tune model's hyperparameters. Ensemble averaging a few not-too-correlated models. Reflect on what to improve.
# 4. visualize and understand decision-making process by plotting a decision tree (suboptimal performance for good to develop intuition)
# --------------------

# **Import libraries and get an idea of the data and the problem**

# In[129]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os ; os.environ['OMP_NUM_THREADS'] = '4'


# In[130]:


titanic = pd.read_csv('../input/train.csv') # labelled survival 
titanic.head()


# The dataset is a list of passengers (rows), some of which survived and some didn't, as indicated by the 'Survived' column. Passagers are described by a few extra features, described on this page: https://www.kaggle.com/c/titanic/data. You should note that some features are numerical and other categorical, as machine learning algorithms behave differently with each. 
# 
# The aim of this competition is to learn enough from those features to predict who died in the the test data set. Perhaps cynically, this kind of model could be used to predict insurance premium for passengers, if an identical second titanic journey was to happen.

# Typically at this stage I would encourage you to run titanic.describe() and titanic.info() to have a first (limited) impression of the numerical data. I usually go through (semi) random plots to get an intuition about the data, without trying to make firm conclusions.

# In[131]:


titanic.describe()


# This is a first sanity check as to whether the data makes sense.
# * We learn here that only 38% of passengers survived (aouch!).
# * More than 50% of people were in 3rd class. Most people are between 20 and 40. They had generally neither siblings nor spouse (sibsp). Neither did they generally have parents or children. This all seems reasonable at this stage, given that the titanic was populated with young and poor migrant moving to the US for work.
# * Fare is highly variable:  there seems to be some pretty big social differences on this boat.  In a real data setting, extremes should be checked: did people actually get onto the Titanic for free (Fare == 0)? And who paid $512?
# * (PassengerId probably isn't that informative. Unless the order in which tickets were sold are associated with some sort of priviledge which could be predictive of survival... We can't rule it out, but I certainly wouldn't start with this.)
# * There are some missing values, e.g. in Age

# **Data completeness**

# In[132]:


# we join the train and test datasets to full diversity of features. No predictions are made at this stage.
test = pd.read_csv('../input/test.csv')
titanic_comb = pd.concat([titanic.drop('Survived',axis=1, inplace=False), test])


# In[133]:


sns.heatmap(titanic_comb.isnull(),yticklabels=False,cbar=False,cmap='plasma')


# The plot above shows that most columns have values on all rows, apart from 'Age' that has a few missing values and 'Cabin' that has lots of missing values.
# 
# Typically, if only a negligible amount of lines are missing, you can get rid of those with negligible impact on your learning algorithm. The opposite extreme is a case in which most values are missing - in this case it may be worth getting rid of this feature completely.
# 
# Here Cabin has a lot of missing values. This may be because most people on the Titanic didn't actually have cabins. If this is true, it may be worth capturing who had cabins and who didn't (although this may be redundant with Pclass and Fare). There may be some extra information hidden in those people that have cabins - e.g. which side of the boat they were and how far from the exit they were. This is probably quite a lot of work that could be undertaken, but won't be done here.
# 
# Age seems like a relevant variable, and has most values, but quite a few are missing. How much would we lose getting rid of those?

# In[134]:


print('fraction of values missing')
print(titanic_comb['Age'].isnull().sum() / titanic_comb['PassengerId'].count())
print()
print('how many lines?')
print(titanic_comb.shape)


# We could drop those lines, but this dataset isn't that large (891 rows for training set, 1309 for training + test set), and 20% starts to be quite a bit. However, we should only fill missing values with information  that makes sense.
# What I suggest we do is to quickly assess the exercise difficulty by doing a very quick-and-dirty model, do understand how much work we need to do to get satisfactory results.

# -------------------------------------------------------------------------

#  # **(1) How hard is the problem? First quick-and-(very)-dirty model**

# The easiest way to get a machine learning algorithm to run is to drop the columns or lines that aren't complete. There is also a temptation to focus on numerical variables. Actually a very quick way to get an idea of the importance of numerical variables is to look at correlations.

# In[135]:


sns.heatmap(titanic.corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


#  Here 'Survived' seems quite well linked linearly to the class and the fare (which are probably quite redundant, to be checked). 
#  
#  Linear associations with Age, number of siblings or spouse (SibSp), parents or children (Parch) don't seem obvious. However, this doesn't mean that these parameters aren't strong predictors of survival
# * There may be several ambiguities or confonding factors at play: 
#  * Maybe old people were all rich and rich people were preferentially saved. At the same time, if females were generally younger than males and favored, this could create a high likelihood for both young and old people to be saved. Without balancing the data out, it is quite hard to take meaningful conclusions from single variables.
#  * Parch has the same value for parents and children, and children might have been saved preferentially. However, children and adults can be differentiated from age.
# * It may also mean that predictive relationships are non-linear:
#  * For example, it isn't uncommon to protect preferentially young and old people, which are at opposite end of the age spectrum (non-linear).
# 
#  Numerical variables do not look that strong by themselves, so we'll do the minimal effort to transform categorical variables into numerical ones

# In[136]:


#---------- build a quick and dirty dataframe ---------- 
titanic_dirty = titanic.copy()

titanic_dirty.drop(['PassengerId', 'Cabin', 'Name', 'Ticket'],axis=1,inplace=True) # most likely to create spurious correlations than causal ones

#categorical variables transformed into simple several binary columns e.g. male = 1 female = 0 type of thing
sex = pd.get_dummies(titanic_dirty['Sex'],drop_first=True) # probably very relevant, not that hard
embark = pd.get_dummies(titanic_dirty['Embarked'],drop_first=True)

titanic_dirty.drop(['Sex', 'Embarked'],axis=1,inplace=True) # drop categorical columns now that we've converted them
titanic_dirty = pd.concat([titanic_dirty,sex,embark],axis=1)

#deal with na
titanic_dirty.dropna(inplace=True) # we drop values for lines in which Age is missing for this first model (quicker)
titanic_dirty.head()


# The aim of this section was to quickly assess the difficulty of the problem. Let's accelerate our quick and dirty approach and use an off-the-shelf classic classifier. If have picked Random Forrest Classifiers - many more would have done the job.
# 
# (For the curious:
# * in a nutshell, Random Forest Classifiers are improved versions of decision trees, which take decisions such as "Age < 20 and Sex = Female".
# * understanding the machine learning algorithms and studying a dataset are both useful, but I'd argue they are almost orthogonal when one begins. That being said, here is a simple explanation http://dataaspirant.com/2017/05/22/random-forest-algorithm-machine-learing/
# )

# In[13]:


# one fundamental step in supervised learning is be able to assess the efficacy of our models on "unseen data"
# here we'll do this by randomly spliting our dataset into two parts: 70% for training set and 30% for test set
from sklearn.model_selection import train_test_split

# features are stored in X, and survival in y. Train corresponds to training.
X_train, X_test, y_train, y_test = train_test_split(titanic_dirty.drop('Survived',axis=1), 
                                                    titanic_dirty['Survived'], test_size=0.30, random_state = 2)


# In[14]:


# here we train our random forest classifier on the training set, and predict survival on the test set
from sklearn.ensemble import RandomForestClassifier
random_state = 2 # handy to have repeatable results (and I change it when I want)
RFC = RandomForestClassifier(random_state = 2) 
RFC.fit(X_train, y_train) # this single line trains our machine learning model! (sklearn offers a pretty amazing interface)
y_test_pred = RFC.predict(X_test)


# In[15]:


# here we assess the prediction's accuracy
# Kaggle chose to use accuracy as a metric, so this is what we will focus on
print('Accuracy: ',(y_test_pred == y_test).mean())

# However, I believe it is a good habit to look beyond accuracy, as precision, recall or F1-score are sometimes closer to business objectives, and this can flag odd behaviors
# this can be done easily with the  following functions: from sklearn.metrics import confusion_matrix, classification_report


# So... our quick and dirty algorithm predicts 80% of cases. Is it good or bad?

# In[16]:


print(1 - y_train.mean())
print(1 - y_test.mean())


# By looking only at the survival rate and randomly guessing, our predictions would have been 60% accurate. We've done better, but it's always good to put things in perspective.
# 
# If we hadn't used categorical variables, the prediction accuracy drops from 80% to 67% - much closer to the naive 60% survival rate.
# 
# 

# In[17]:


# let's try to see what numerical variables only yield
RFC_num = RandomForestClassifier(random_state = 2) 
RFC_num.fit(X_train.drop(['male', 'Q', 'S'],axis=1), y_train)
y_test_num_pred = RFC_num.predict(X_test.drop(['male', 'Q', 'S'],axis=1))

print('Test Accuracy (num variables only): ',(y_test_num_pred == y_test).mean())


# In a real world setting, whether we stop at the first model or not depends on the business context - value of marginal gains vs cost of improvements.
# 
# Before we go and work on adding more features, let's see how the classifier did on the train set.

# In[18]:


y_train_pred = RFC.predict(X_train)
print('Train Accuracy: ',(y_train_pred == y_train).mean())
print('Test Accuracy: ',(y_test_pred == y_test).mean())


# There is quite a large gap between the training accuracy and the test accuracy. This may be a sign of an over-fitting problem. We cannot get more data in this context (apart from the test set, which wouldn't change the orders of magnitude). This means we should be careful not to feed the algorithm too many parameters that we think may be spuriously related to the datapoints.

# -------------------------------------------------------------------------

# # ** (2) Second iteration: better processing and additional features extraction **

# To get an intuition for the data I'm plotting a big overview figure. We still haven't improved our data processing though, so insights will be limited.

# In[19]:


# data overview
g = sns.PairGrid(titanic.drop('Pclass',axis=1).dropna(),hue='Sex') # we drop NA and Pclass to quickly get data overview
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot).add_legend()


# The first thing I look at in this figure are the 'Survived' column and row.
# * There seems to be big differences in survival rates between male and female
# * Children seem to have been largely saved. Elderlies seem to have largely died - I wouldn't be surprised to find revenue differences here.
# * Having siblings / spouse / parents / children seems to help marginally, when taken alone
# 
# There are other interesting trends on this plot, but it's hard to know what to do with them.  More data visualization should be done on your own console to support your 

# In[20]:


titanic.head(1)


# **Brainstorming**
# 
# Feature engineering
# * deal with simple categorical variables: Sex, Cabin yes/no, Embarked, and Pclass
# * try to extract useful information from Name: e.g. is name length an indicator? does Mr / Miss etc add anything to the sex and class?
# * fill Age missing values: age seems to impact survival probability. What's the best way to guess the age?
# * is there any relevant information on the ticket or cabin number? E.g. anything related to the distance to the nearest unsubmersed exit? People travelling together?
# * are Fare and Pclass showing complementary information, e.g. for people travelling on the same ticket, or does Fare just give more information than Pclass?
# * any value in making clear categories where we already think there is a strong association? E.g. we "women or children first": shall we have a variable flagging child as a categorical variable, instead of an age? And can we group Parch and SibSp in a way that would be more clearly associated with survival, e.g. familly size, differentiate between having children or parents, etc?
# 
# Processing methods
# * feature scaling: some algorithms work better with normalized data
# * Aavoiding overfitting: if the causality between variable and survival doesn't seem to make sense, consider removing it, e.g. get rid of PassengerId
# * cross-validation and test set, fold cross validation
# * hyperparameters tuning
# * plot learning  curves and check for overfitting / underfitting
# * feature importance
# 
# Usually I do my own brainstorm, look up some ideas and write down what I find interesting, and do a selection afterwards.
# On titanic there are some pretty complete analysis that have been made (e.g. [this one ](https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling/notebook) or [this one](https://www.kaggle.com/thilakshasilva/predicting-titanic-survival-using-five-algorithms?scriptVersionId=1940189), but on other data science projects I'd still look for inspiration at some point in the middle of my analysis.

# **Non-obvious features - title and age missing values**

# In[21]:


# ------- title -------
# Title could be a predictor of status, age and sex. Not sure at this point whether it adds any information.

# notice how the title is in between a comma "," and a dot "." We use split() to isolate that
# there may also be some gap - strip() can deal with this 
titanic_comb['title'] = titanic_comb['Name'].apply(lambda x: x.split(",")[1].split(".")[0].strip())

#plot
plt.figure(figsize=(16,3))
sns.countplot(titanic_comb['title'])
plt.tight_layout()
plt.show()


# In[22]:


titanic['title'] = titanic['Name'].apply(lambda x: x.split(",")[1].split(".")[0].strip())
plt.figure(figsize=(16,3))
sns.barplot(data=titanic, x='title', y='Survived')
plt.tight_layout()
plt.show()


# Generally a good idea to avoid categories with few instances, signal to noise ratio is poor. * Don means Mr, Mme and Dona means Mrs, and Mlle means Miss
#  * however, it's not clear whether this is considered a honorific title or just an indication of gender + mariage status
# * similarly, Master apparently used to designate a young boy
# 
# To avoid redundancy, if we intend to keep age and gender as a variable in our model, the title grouping should focus on new information, e.g. status (title_status). Title could also be pretty precise to define missing values of age also, so we'll do another title grouping for this purpose (title_age). We won't use title_age in our machine learning algorithms.
# 
# I have computed the survival ratesfor the rare titles:
# * low survivors: Capt, Don, Jonkheer, Rev have a probability of 0
#     * this isn't so different from Mr though with a probability of 0.16. This fits the gender and probably isn't too far off on the age. Captains and Reverants are exceptions out of principle.
#     * So we are going to group those titles with Mr.
# * high survivors: we could create a special class "priviledged" for them, but instead we'll affect them to Mrs, which has similarly high probabilities of survival in 1st and 2nd class. 
#  * around 0.5 : Col, Dr, Major
#    * this is about 3x the survival rate of males. This is probably taken into account to some extent by their class, but not fully.
#  * around 1: Lady, Mlle, Mme, Ms, Sir, the Countess
#     * for females this is about 40% higher chance of survival (class at play already). For the Sir, this is of the order of 6x that of Mr.
# 
# This allows us to drop gender, and we'll capture Pclass and/or Fare separately. This also allows to capture the difference between male children and male adults.
# Actually, a deeper look at the difference between Mrs and Miss seems to be due to class, and once Pclass is captured, the difference between Mrs and Miss doesn't seem significant. I'll merge them.
# 

# In[23]:


g = sns.barplot(data=titanic[(titanic['title']=='Miss') | (titanic['title']=='Mrs')], x='title', y='Survived', hue='Pclass')


# In[24]:


titanic_comb.groupby('title')['Age'].median().plot.bar()


# In[25]:


titanic_comb[titanic_comb['Age'].isnull()]['title'].value_counts()


# In[26]:


titanic_comb[titanic_comb['Age'].isnull()]['Pclass'].value_counts()


# In[27]:


def title_preprocessing(df):
    df['title'] = df['Name'].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    
    # grouping for age prediction
    df['title_age'] = df['title'].replace(['Ms'], 'Mrs')
    df['title_age'] = df['title_age'].replace(['Don', 'Rev', 'Mme', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'the Countess', 'Jonkheer', 'Dona'], 'irrelevant_here')
    
    # grouping for status prediction
    df['title_status'] = df['title'].replace(['Capt', 'Don', 'Jonkheer', 'Rev'], 'Mr')
    df['title_status'] = df['title_status'].replace(['Dr', 'Mme', 'Ms', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'the Countess', 'Dona'], 'Mrs')
    df['title_status'] = df['title_status'].replace(['Miss'], 'Mrs')
    
    return df

def age_replacement_table(titanic_comb):
    replacement_table = pd.pivot_table(titanic_comb, values='Age', index='Pclass', columns='title_age',aggfunc=np.median)
    replacement_table['Master'][2] = titanic_comb[titanic_comb['title_age']=='Master']['Age'].median() # we replace this value that looks spurious
    return replacement_table

replacement_table = age_replacement_table(title_preprocessing(titanic_comb))
replacement_table


# In[28]:


#the values above look reasonable, apart from Master's age perhaps.
# so at the risk of using values from too small categories, we'll use this table for the replacement

def input_age(cols):
    '''takes Age, title_age and Pclass as input
    returns 'Age' with null values replaced by smart ones'''
    # grab inputs
    Age = cols[0]
    title_age = cols[1]
    Pclass = cols[2]
    
    # correct outputs
    if pd.isnull(Age):
        return replacement_table[title_age][Pclass]
    else:
        return Age


# In[29]:


input_age([None,'Mr',1])


# In[30]:


def title_and_age_processing(df):
    df = title_preprocessing(df)
    df['Age'] = df[['Age','title_age','Pclass']].apply(input_age,axis=1)
    return df

# process for age and title combined, train and test datasets
titanic_comb = title_and_age_processing(titanic_comb)
titanic = title_and_age_processing(titanic)
test = title_and_age_processing(test)


# In[31]:


g = sns.barplot(data=titanic,x='title_status', y='Survived', hue='Pclass')


# ** Non obvious features - Familly size**
# * Already tried to book a table for 10 vs a table for 2: it's much harder. Same on titanic?
# * You could also imagine that the familly doesn't want to leave the titanic before they are complete: a large familly is more likely to have lost a child in the crowd
# * however, we saw children tended to be saved first.

# In[32]:


# ------- familly size -------
titanic['familly_size'] = titanic['SibSp'] + titanic['Parch'] + 1
test['familly_size'] = test['SibSp'] + test['Parch'] + 1
titanic_comb['familly_size'] = titanic_comb['SibSp'] + titanic_comb['Parch'] + 1

graph = sns.factorplot(data=titanic, x='familly_size',y='Survived')
graph.set_ylabels('Survival probability')


# In[33]:


graph = sns.factorplot(data=titanic, x='familly_size',y='Survived', col='Pclass')
graph.set_ylabels('Survival probability')


# In[34]:


graph = sns.factorplot(data=titanic, x='familly_size',y='Survived', col='title_status')
graph.set_ylabels('Survival probability')


# If familly size is a real feature, it appears to be pretty complicated. And one can hypothesize why:
# * kids from priviledged families (Pclass 1 and 2) may not have been running around freely, and would all be saved
# * there may be a threshold effect on size: small families may have an advantage over single individuals (psychological, adults embarking with children, cohesion?), but above a certain size it may be too unpractical (getting lost in the crowd, too hard to find free spots in a boat)
# 
# Hard to know the truth...

# In[286]:


titanic.tail(3)


# ** Non-obvious features - ticket and cabin **
# Why some people have a cabin number and not others isn't clear - is it due to random information losses or does this mean something? 
# 
# For people having a cabin, the cabin location could be meaningful:
# * the collision was shortly before midnight - most passengers were in bed
# * "In the event, Titanic's heading changed just in time to avoid a head-on collision, but the change in direction caused the ship to strike the iceberg with a glancing blow. An underwater spur of ice scraped along the starboard side of the ship for about seven seconds; chunks of ice dislodged from upper parts of the berg fell onto her forward decks."
# * ship sunk in about 2h, but water started coming in almost immediately

# 

# ![Titanic cabin placement. First letter on the ticket may correspond to the floor](http://upload.wikimedia.org/wikipedia/commons/thumb/8/84/Titanic_cutaway_diagram.png/687px-Titanic_cutaway_diagram.png)

# While I could convince myself that lower decks were flooded very quickly and upper deck was damaged by falling ice, [the little cabin data we have seems pretty unreliable](https://www.encyclopedia-titanica.org/cabins.html). 
# 
# In contrast, ticket information seems reliable. However, it's impact on survival feels pretty cryptic to me. It may be just a matter of working hard, but time constraints mean I won't explore these routes for now.

# ** Making variables easier to digest - Fare **

# The fare has many extreme values and is really skewed. Many algorithms don't behave well with this. Applying a log could make it behave better. I also wonder if there would be value in breaking down the Fare in bins, to almost turn it into a categorical variable.

# In[35]:


# first, test has one missing  value for Fare
# this is a passenger in third class, travelling alone. Getting the median fare price should get him covered.
test = test.fillna(titanic_comb['Fare'].median())


# In[36]:


sns.distplot(titanic['Fare'])


# In[37]:


titanic['log_fare'] = np.log(titanic['Fare']+1)
test['log_fare'] = np.log(test['Fare']+1)
titanic_comb['log_fare'] = np.log(titanic_comb['Fare']+1)

sns.distplot(titanic['log_fare'])


# **Embarkment port - real data or spurious correlation?**

# Let's have a look at the embarkment port. I can't get my head around why that would matter. I bet it is due to differences in the population that embarked in different places, but in a way that's already captured by Title, Pclass and Fare. Keeping it may not hurt, but the risk is (i) spurious correlations and (ii) redundancy, which destabilizes some algorithms. Let's see if the data support this.

# In[38]:


g = sns.factorplot(data=titanic, x='title_status', y='Survived', hue='Pclass', col='Embarked',kind='bar')


# The relative survival rate is mostly unaffected by the embarquement port, apart from (Mrs, 1st, Q) and (Master, 3rd, S). The error bars are massive though - actually very few samples are impacted. 

# In[39]:


print(((titanic['title_status'] == 'Mrs') & (titanic['Pclass']==1) & (titanic['Embarked']=='Q')).sum())
print(((titanic['title_status'] == 'Master') & (titanic['Pclass']==3) & (titanic['Embarked']=='S')).sum())


# **Transform categorical variables into dummies**

# In[40]:


titanic_2 = titanic.copy() #for convenience, copy the dataset before dropping variables, so I can keep playing with the old dataframe
titanic_2 = pd.get_dummies(titanic_2, columns=['Sex'], drop_first=True)
titanic_2 = pd.get_dummies(titanic_2, columns=['Embarked'], prefix='Em',drop_first=True)
titanic_2 = pd.get_dummies(titanic_2, columns=['title_status'],drop_first=True)

test_2 = test.copy()
test_2 = pd.get_dummies(test_2, columns=['Sex'], drop_first=True)
test_2 = pd.get_dummies(test_2, columns=['Embarked'], prefix='Em', drop_first=True)
test_2 = pd.get_dummies(test_2, columns=['title_status'], drop_first=True)


# In[300]:





# In[78]:


# ----- to try later?
# create categorical variable from numerical ones?
# Pclass
# Fare
# Familly size


# In[41]:


# get rid of variables we won't use
titanic_2.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'title', 'title_age'], axis=1, inplace=True)
test_2.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'title', 'title_age'], axis=1, inplace=True)


# In[302]:


titanic_2.head()


# **Separate train dataset and test dataset**

# In[42]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(titanic_2.drop('Survived',axis=1), 
                                                    titanic_2['Survived'], test_size=0.30, random_state = 2)


# # *** Modeling attemp (2): let's try to feed all those variables to some ML algorithms and see what happens ***

# In[43]:


# Let's import a range of classifiers that have different strengths and weaknesses and seee how they do
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[44]:


random_state = 2

models = [
    RandomForestClassifier(random_state = random_state),
    AdaBoostClassifier(random_state = random_state),
    DecisionTreeClassifier(random_state = random_state),
    LogisticRegression(random_state = random_state),
    SVC(random_state = random_state, gamma='auto'),
    KNeighborsClassifier(),
    MLPClassifier(random_state = random_state),
    LinearDiscriminantAnalysis()
]

model_names = [
    "RandomForestClassifier",
    "AdaBoostClassifier",
    "DecisionTreeClassifier",
    "LogisticRegression",
    "SVC",
    "KNeighborsClassifier",
    "MLPClassifier",
    "LinearDiscriminantAnalysis"
]
i=0
for model in models:
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    print(model_names[i])
    print((y_test == y_test_pred).mean())
    print('')
    i=i+1


# Let's see what result we get when we try Adaboost on the test set and submit it to Kaggle.

# In[336]:


ada = AdaBoostClassifier(random_state = random_state)
ada.fit(X_train, y_train)
y_pred_Adaboost = ada.predict(test_2)
data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':y_pred_Adaboost})
data_to_submit.to_csv('ada_to_submit.csv', index = False)

RF = RandomForestClassifier(random_state = random_state)
RF.fit(X_train, y_train)
y_pred_RF = RF.predict(test_2)
data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':y_pred_RF})
data_to_submit.to_csv('RF_to_submit.csv', index = False)


# Kaggle submission result: 
# * Adaboost = 76%
# * RandomForrest = 76%
# 
# This is lower than our own test data - bad luck?

# After all this work, we don't seem to be doing much better, do we?!
# We still need to diagnose why. However it feels like (i) quick and dirty models can be very cost efficient, (ii) really lots of features unselectively doesn't necessarily help much, (iii) we may need to tune the models. 

# Let's understand better score uncertainty. If you change how the training and test datasets are split or the random initialization value of the model, the score varies. One popular method is the kfold cross-validation procedure.

# In[46]:


from sklearn.model_selection import cross_val_score, StratifiedKFold
kfold = StratifiedKFold(n_splits=10)


# In[109]:


# compute test score several times on different folds for each model
cv_results = []
for model in models:
    cv_results.append(cross_val_score(model, X_train, y=y_train, scoring="accuracy", cv=kfold, n_jobs=4))
    
# from these different scores, assess mean and standard deviation
cv_means = []
cv_std = []
for cv_results in cv_results:
    cv_means.append(cv_results.mean())
    cv_std.append(cv_results.std())                 


# In[417]:


# put results together and plot
cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":model_names})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="coolwarm",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")     


# From here I feel we have a few options:
# * tune hyperparameters of a few models (using gridsearch and kfold for cross-validation). Then potentially ensemble average. How do I chose those models though?
# * reduce number of features, as more features need more data to be learned
# * add features to get more information
# * modify features so they are easier to learn, e.g. categorical data for Familly Size, Fare, Pclass, Fare, or using polynomial features
# * regularization
# 
# How to chose?
# At this stage, my inclination is to try to understand whether the models suffer from a bias problem or a variance problem, i.e. are they underfitting or overfitting the training dataset?

# In[47]:


from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    # I came across this very nice representation of learning curves on a kernel by Yassine Ghouzam
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
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
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(RandomForestClassifier(random_state = random_state),"RandomForestClassifier learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(DecisionTreeClassifier(random_state = random_state),"DecisionTreeClassifier learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(AdaBoostClassifier(random_state = random_state),"AdaBoostClassifier learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(LogisticRegression(random_state = random_state),"LogisticRegression learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(SVC(random_state = random_state, gamma='auto'),"SVC learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(MLPClassifier(random_state = random_state, gamma='auto'),"MLPClassifier learning curves",X_train,y_train,cv=kfold)


# All models seem to have a variance / overfitting problems
# * large variance
#  * Random Forest
#  * Decision Tree
#  * SVC
# * small-ish variance
#  * Adaboost
#  * Logistic Regression

# **feature importance**

# In[ ]:


forest = AdaBoostClassifier()
forest.fit(X_train, y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking and contribution
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d %s (%f)" % (f + 1, indices[f],test_2.columns.tolist()[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# In[335]:


forest = RandomForestClassifier()
forest.fit(X_train, y_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking and contribution
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d %s (%f)" % (f + 1, indices[f],test_2.columns.tolist()[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# # (4) Selecting a small number of meaningful parameters, and optimizing the models

# Different classifiers have selected different primary features. This can be due to chance, different algorithms, or redundancy. To select a smaller number of features:
# * Embark isn't very significant and I bet it is spurious. 
# * Fare and logfare are redundant
# * sex is captured by class_status
# * I'd like to get rid of SibSp and Parch - they seem to contribute, but they could also be part of the overfitting problem

# In[48]:


titanic_3 = titanic_2.copy()
titanic_3.drop(['Sex_male', 'SibSp', 'Parch', 'Em_Q', 'Em_S', 'Fare'], axis=1, inplace=True)

test_3 = test_2.copy()
test_3.drop(['Sex_male', 'SibSp', 'Parch', 'Em_Q', 'Em_S', 'Fare'], axis=1, inplace=True)
titanic_3.head(1)


# In[49]:


X_train, X_test, y_train, y_test = train_test_split(titanic_3.drop('Survived',axis=1), 
                                                    titanic_3['Survived'], test_size=0.20, random_state = 2)

from sklearn.model_selection import GridSearchCV


# **Tuning Meta-Parameters**
# Each model have parameters that make it behave differently. This impacts underfitting / overfitting e.g. through regularization or the model complexity, this impact convergence e.g. learning rate, etc.
# 
# One way to tune those models is to try a range of parameters. SkitLearn has a tool called GridSearchCV that does that for us.
# 
# Usually what I do is to ensure parameters go over large orders of magnitude, and return a first set of optimum parameters. I then refine the grid around these values and run gridsearch again.

# In[397]:


# Adaboost
DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2,5,10],
              "learning_rate":  [0.0001, 0.01, 0.1,0.15, 0.2,0.25, 0.3, 2]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsadaDTC.fit(X_train,y_train)

ada_best = gsadaDTC.best_estimator_

# let's check the cross_validation score and test score
print('Cross-validation accuracy: ', gsadaDTC.best_score_)
print('Test accuracy: ', (ada_best.predict(X_test) == y_test).mean())
print('')
print("Using the following parameters:")
print(gsadaDTC.best_params_)


# In[395]:


# RFC Parameters tunning 
RFC = RandomForestClassifier()

## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 5, 6],
              "min_samples_split": [2, 3, 6],
              "min_samples_leaf": [1, 3, 6],
              "bootstrap": [False],
              "n_estimators" :[100,300,500],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsRFC.fit(X_train,y_train)

RFC_best = gsRFC.best_estimator_

# let's check the cross_validation score and test score
print('Cross-validation accuracy: ', gsRFC.best_score_)
print('Test accuracy: ', (RFC_best.predict(X_test) == y_test).mean())
print('')
print("Using the following parameters:")
print(gsRFC.best_params_)


# In[378]:


### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1, 3],
                  'C': [0.01, 1, 10, 50, 100, 200, 300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(X_train,y_train)

SVMC_best = gsSVMC.best_estimator_

# let's check the cross_validation score and test score
print('Cross-validation accuracy: ', gsSVMC.best_score_)
print('Test accuracy: ', (SVMC_best.predict(X_test) == y_test).mean())
print('')
print("Using the following parameters:")
print(gsSVMC.best_params_)


# In[ ]:





# In[393]:


### MLPClassifier
neural = MLPClassifier()
neural_param_grid = {'solver': ['lbfgs'], 'max_iter': [500, 1000, 1500], 
                     'alpha': [0.0001, 0.0005, 0.001, 0.003, 0.01], 
                     'hidden_layer_sizes':[6,7,8], 
                     'random_state':[0,2,8]
    
}

gsneural = GridSearchCV(neural,param_grid = neural_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsneural.fit(X_train,y_train)

neural_best = gsneural.best_estimator_

# let's check the cross_validation score and test score
print('Cross-validation accuracy: ', gsneural.best_score_)
print('Test accuracy: ', (neural_best.predict(X_test) == y_test).mean())
print('')
print("Using the following parameters:")
print(gsneural.best_params_)


# In[402]:


### LogisticRegression
logistic = LogisticRegression()
logistic_param_grid = {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 10, 30, 100, 300, 1000]
    
}

gslogistic = GridSearchCV(logistic,param_grid = logistic_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gslogistic.fit(X_train,y_train)

logistic_best = gslogistic.best_estimator_

# let's check the cross_validation score and test score
print('Cross-validation accuracy: ', gslogistic.best_score_)
print('Test accuracy: ', (logistic_best.predict(X_test) == y_test).mean())
print('')
print("Using the following parameters:")
print(gslogistic.best_params_)


# Let us submit results to kaggle from a couple of those optimized estimators

# In[404]:


y_pred_Adaboost = ada_best.predict(test_3)
data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':y_pred_Adaboost})
data_to_submit.to_csv('ada_opt_to_submit.csv', index = False)


y_pred_RF = RFC_best.predict(test_3)
data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':y_pred_RF})
data_to_submit.to_csv('RF_opt_to_submit.csv', index = False)


# Kaggle results:
# * Adaboost: 70% (down from 75%)
# * Random Forrest: 76.6% (up from 76%)

# Out of curiosity, let's explore the learning rates again

# In[406]:


g = plot_learning_curve(ada_best,"AdaBoostClassifier learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(RFC_best,"RandomForestClassifier learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(logistic_best,"LogisticRegression learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(SVMC_best,"SVC learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(neural_best,"MLPClassifier learning curves",X_train,y_train,cv=kfold)


# **Ensemble averaging **
# For voting strategies, the models should roughly perform as well and not be too correlated.

# In[409]:


y_test_ada = pd.Series(ada_best.predict(X_test), name='Ada')
y_test_RFC = pd.Series(RFC_best.predict(X_test), name='RF')
y_test_SVMC = pd.Series(SVMC_best.predict(X_test), name='SVM')
y_test_log = pd.Series(logistic_best.predict(X_test), name='logistic')
y_test_neur = pd.Series(neural_best.predict(X_test), name='neural')

ensemble_test_predictions = pd.concat([y_test_ada,y_test_RFC,y_test_SVMC,y_test_log, y_test_neur],axis=1)

g = sns.heatmap(ensemble_test_predictions.corr(),annot=True, cmap="coolwarm")


# Classifiers have similitudes, but they are different. At this point we could test more models and see which models are uncorrelated with each other.
# 

# In[411]:


from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators=[('rfc', RFC_best), ('logistic', logistic_best),
('svc', SVMC_best), ('ada',ada_best),('neural',neural_best)], voting='soft', n_jobs=4)

voting = voting.fit(X_train, y_train)
y_test_voting = voting.predict(X_test)
print('Test accuracy: ', (y_test_voting == y_test).mean())


# Doesn't sound better than RandomForrest. Let's see what the Kaggle submission predicts.

# In[412]:


y_pred_vote = voting.predict(test_3)
data_to_submit = pd.DataFrame({
    'PassengerId':test['PassengerId'],
    'Survived':y_pred_vote})
data_to_submit.to_csv('ensemble_to_submit.csv', index = False)


# Kaggle score: 77% - best score in this notebook so far.
# 
# From there, if I wanted to compete, I would consider
# * using new features (including some I have discarded so far)
# * simplifying existing features: pre-run
# * introducing classifiers that aren't correlated with the other ones
# * potentially normalizing data, looking for outliers
# * I'm surprised how different the results of the test are from the data we got from train_test_split.
#     * Maybe the test dataset has different distributions of parameters. We could try to balance that?
# * Strategies to reduce overfitting should also be deployed.
# 
# As I said at the beginning, this isn't a kernel about how to get the best score, but about a thought process.

# # (5) understand the decision making process by plotting a decision tree (suboptimal predictions)

# Machine learning is great at many things, but not necessarily at being understood by humans. Some models such as some linear regressions and decision trees are easier to interpret, although they are not necessarily the best performers. Here we'll use decision trees and graph viz to get an idea of what those models are doing

# In[54]:


# DTC Parameters tunning 
DTC = DecisionTreeClassifier()

## Search grid for optimal parameters
DTC_param_grid = {"max_depth": [1,2,3,4,5,6,7,8,9,10,20,30,50],
              "max_features": [1, 3, 5, 6],
              "min_samples_split": [2, 3, 6],
              "min_samples_leaf": [1, 3, 6],
              "criterion": ["gini", "entropy"],
                "random_state": [0, 2, 4]}

gsDTC = GridSearchCV(DTC,param_grid = DTC_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsDTC.fit(X_train,y_train)

DTC_best = gsDTC.best_estimator_

# let's check the cross_validation score and test score
print('Cross-validation accuracy: ', gsDTC.best_score_)
print('Test accuracy: ', (DTC_best.predict(X_test) == y_test).mean())
print('')
print("Using the following parameters:")
print(gsDTC.best_params_)


# In[57]:


import graphviz
from sklearn import tree


# In[102]:


# in line function for small trees
dot_data = tree.export_graphviz(DTC_best, out_file=None, 
                         feature_names=test_3.columns.tolist(),
                                class_names=["Died","Survived"],
                         filled=True, rounded=False, precision=2, label='root', impurity=False,
                         special_characters=True, max_depth=2) 
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("titanic_tree")


# In[100]:


graph


# In[98]:


# in line function for small trees
dot_data = tree.export_graphviz(DTC_best, out_file=None, 
                         feature_names=test_3.columns.tolist(),
                                class_names=["Died","Survived"],
                         filled=True, rounded=False, precision=2, label='root', impurity=False,
                         special_characters=True, max_depth=6) 
graph = graphviz.Source(dot_data)
graph


# In[ ]:




