#!/usr/bin/env python
# coding: utf-8

# Listen, there are so many good kernels out there which are well-written, easy to understand and even getting higher scores for this competition much better than what I have done here. So, don't consider this kernel as a comprehensive informative kernel on titanic dataset.
# But, what I'm trying to demonstrate here is only a playground of my understanding of data analysis techniques and Machine Learning algorithms and trying to explain and share it with others. So, if what I mentioned doesn't bother you, keep reading my kernel and make me much happier by giving me back some feedback or comments.
# 
# What we're going to do is simply, loading the dataset, clean it, preprocess it, visualise it, making some hypothesis and then constructing our models.

# In[2]:


# data analysis
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
# Configuring plotting visual and sizes
sns.set_style('whitegrid')
sns.set_context('talk')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

plt.rcParams.update(params)

# tools libraries
import random
import math

# Scientific packages
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV


# We'll use pandas to load CSV file as a convenient way for importing and converting data into data frame.

# In[3]:


# Load the datasets
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# Check datasets dimensions
print(train_df.shape)
print(test_df.shape)


# In[4]:


# Take a look to the first few rows of training dataset
train_df.head()


# For the sake of simplicity in making modification on datasets, we are going to merge train and test datasets into full dataset and when we made all the data wrangling changes and right before making models, we again divide it into train and test datasets.

# In[5]:


# Filling Survived column with 0s as a placeholder in test dataset
test_df['Survived'] = 0

# merge two dataset in order to construct full dataset
full = pd.concat([train_df, test_df], 
                 axis=0, # merge on rows
                 ignore_index=True, 
                 sort=False)


# In[6]:


# Check dataset again
full.head()


# That's just me or some column names are not very readable, let's fix them first:

# In[7]:


# Renaming dataset columns to increase readbility
full.rename(columns={'Pclass':'TicketClass',
                     'SibSp':'Sibling_Spouse',
                     'Parch':'Parent_Children',
                     'Fare':'TicketFare'}, 
            inplace=True)  # Apply changes on the dataset


# In order to understand dataset better and having a view of how numerical columns are spread or how categorical columns are used we can use following snippets:

# In[8]:


# Checking all columns for missed values and types
full.info()


# In[9]:


# Checking numerical columns
full.describe()


# * ** 75%** of the passengers having age less than 39 and 50% of them having age between **21** to **39**.
# * Average ticket fare was around **33** despite having some passenger paid as high as **512**.
# * More than **75%** of passengers travelling without their parents or children.

# In[10]:


# Checking categorical columns
full.describe(include=['O'])


# * We have two groups consists of two persons each with exact same name **Kelly, Mr. James** and **Connolly, Miss. Kate** but different age and embarked in different locations. 

# In[11]:


full[full.Name.isin(['Connolly, Miss. Kate','Kelly, Mr. James'])].sort_values(by='Name')


# Here, we're going to depict frequency of survivals based on some selected columns:

# In[12]:


# Make a plot instance 
fig, axes = plt.subplots(2,2,figsize=(15,15))

# Drawing plots
sns.countplot(data=train_df, x='Survived', ax=axes[0][0])
sns.countplot(data=full, x='TicketClass', ax=axes[0][1])
sns.countplot(data=full, x='Sex', ax=axes[1][0])
sns.countplot(data=full, x='Embarked', ax=axes[1][1])

# showing frequency percentage on top of each column
for ax in axes.flatten():
    total = len(full) if ax.get_xlabel() != 'Survived' else len(train_df)
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('Freq: {:.1f}%'.format(100.*y/total), (x.mean(), y), 
                ha='center', va='bottom') # set the alignment of the text


# * Only around **39%** survived :( 
# * Not surprisingly, more than **54%** of tickets are 3rd class.
# * Passengers are mostley gentlemen by having around **64%** of all passengers.
# * The thing which we work on soon is, around **70%** of passengers embarked at Southampton. We'll check socio-economic status of Southampton passengers and trying to find some relation between survivors and their port of embarkation. 
# 
# 

# Making some simple analysis for survivals:

# In[13]:


full.loc[:891,['TicketClass', 'Survived']].groupby(['TicketClass'], as_index=False).mean().sort_values(by='Survived', 
                    ascending=False)


# * Around **63%** of survivals have 1st class ticket.

# In[14]:


full.loc[:891,['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', 
             ascending=False)


# * Around **74%** of all survivals are female and only **19%** are male.

# In[15]:


full.loc[:891,['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', 
             ascending=False)


# * Only a bit more than half of Southampton port (the most crowded port between all three ports) survived.

# Maybe it would be a better idea if we visualise them in graphs:
# 
# For example, can we say passengers who were richer than others mostly survived? 

# In[16]:


grid = sns.FacetGrid(full.loc[:891], 
                     col='Survived', 
                     row='TicketClass',
                     aspect=2)
grid.map(plt.hist, 'TicketFare', alpha=0.8)


# Yes, as we can clearly see in the right columns, passengers which paid more than regular fare had more chance to survive, MONEY!! Last row doesn't indicate any difference since there are a few third class passenger how paid more than regular price.

# Let's see how is percentage of survivals in various ages and genders:

# In[17]:


grid = sns.FacetGrid(full.loc[:891], 
                     col='Survived', 
                     row='Sex',
                     aspect=2)

grid.map(plt.hist, 'Age', alpha=.8)


# In total, we can say mid age men had more fatalities compare to others. 

# Ok, let's move on and try a bit to clean up datasets and doing some **Feature Engineering**:
# Between the columns of dataset it seems ``cabin``, ``ticket`` and ``PassengerId`` columns are kinda useless, so we can start by dropping them:

# In[18]:


# Getting rid of useless columns
full.drop(['Cabin','Ticket','PassengerId'], 
          axis=1, # Drop column
          inplace=True)


# Working on title of the passengers and its relation with being survived.

# In[19]:


# Extracting titles out of names
full['Title'] = full['Name'].str.extract('([A-Za-z]+)\.', 
                                         expand=False)
    
# Making a cross-table of Titles and their gender
title_cross = pd.crosstab(full['Title'], 
                          full['Sex'])

title_cross


# There are so many rare titles, we can replace them with most common ones:

# In[20]:


# Transforming less regular titles to more regulars.
full.Title.replace(['Col','Rev','Sir'],
                   'Mr',
                   inplace=True)

full.Title.replace(['Ms','Mlle'],
                   'Miss',
                   inplace=True)

full.Title.replace('Mme',
                   'Mrs',
                   inplace=True)


# Convert remaining titles as Rare
full.Title.replace([x for x in list(np.unique(full.Title))                     if x not in ['Mr','Miss','Mrs','Master']] ,
                   'Rare',
                   inplace=True)
    
title_report = full.loc[:891,['Title','Survived']].groupby('Title',
         as_index=False).mean()

# Visualization
plt.bar(x=title_report['Title'], 
        height=title_report['Survived'],
        alpha=0.8,color='grkmb')

plt.title('Survival rate based on Titles',
          fontsize=20)
plt.ylabel('Survived Percentage %',
           fontsize=15)

# showing frequency percentage on top of each column
ax = plt.gca()
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.2f}%'.format(y), (x.mean(), y), 
            ha='center', va='bottom') # set the alignment of the text


# Alright, let's see what we learned so far:
# * **57%** of passengers with `Master` title survived, in contrast, only **16%** of `Mr.` could help themselves. MONEY!!
# * There is a slight difference in the percentage of survivals in the female group, **70%** for `Miss` and **79%** for `Mrs.` Thank you gentlemen!

# We are all done with ``Name`` column and we can drop it too: 

# In[21]:


# Getting rid of useless columns
full.drop(['Name'], axis=1, inplace=True)


# Maybe, here would be a right place to checking for columns with **missing values** before moving any further. 

# In[22]:


# Checking for columns with NAs
full.isnull().sum()


# Let's work on `Embarked` and `TicketFare` first, we can easily fill them with more frequent values for categorical values or mean of column for numerical one.

# In[23]:


# Filling NAs in Embarked with the most frequent value.
full.loc[full['Embarked'].isnull(),'Embarked'] = full['Embarked'].dropna().mode()[0]


# In[24]:


# Filling missed value for ticket fare by mean of the column
full.loc[full['TicketFare'].isnull(),'TicketFare'] = full['TicketFare'].dropna().mean()


# The next thing we can work on would be ``Age`` column. First, we need to work on missed values and then using techniques like continuous values binning in order to prepare them for modeling:

# In[25]:


# First see how many missed value we have in Age column
full.Age.isnull().sum()


# **263** looks like a high volume for missing value for a particular column in this dataset. In order to fill NAs we need a more clever approach to have some guess reasonably close to the actual age of passenger by considering other columns in the dataset. One way would be checking the correlation between Age and other columns:

# In[26]:


# Making correlation matrix of all columns except Survived
corrMatt = full[full.columns.difference(['Survived'])].corr()
mask = np.array(corrMatt)

# Turning the lower-triangle of the array to false
mask[np.tril_indices_from(mask)] = False

# Making the heatmap of correlations
fig,ax = plt.subplots()
sns.heatmap(corrMatt, 
            mask=mask,
            vmax=.8, 
            square=True,
            annot=True,
            ax=ax)


# Based on the heatmap, we can see there are some notable correlation between `Age` and `TicketClass` and followed by `Sibling_Spouse`. We'll use all of those columns for making guess for missing ages: 

# In[27]:


# Finding mean and standard deviation for age of passengers 
age_estimator = full[['Age','TicketClass','Sibling_Spouse']].groupby(['TicketClass','Sibling_Spouse']).agg(['mean','std'])

# Filling the NAs by making random numbers around their group mean
age_nulls = full.loc[full.Age.isnull(),:]

for idx,rec in age_nulls.iterrows():
    # For each null age calculating a random age based on correlated Ticketclass and Sibling_Spouse columns
    mean = age_estimator.loc[(rec['TicketClass'],rec['Sibling_Spouse']),('Age','mean')]
    std = age_estimator.loc[(rec['TicketClass'],rec['Sibling_Spouse']),('Age','std')]
    gen_age = random.uniform(mean-std, mean+std)
    
    # Convert negative ages to 1
    full.loc[idx,'Age'] = gen_age if gen_age >= 1 else 1

# Transform ages value to upper integer value
full['Age'] = full['Age'].apply(math.ceil)


# Ok, what we have done above is, first we look for all the rows in the dataframe which has NA as value for age column. Then we calculate mean and std for groups like them by having same `TicketClass` and `Sibling_Spouse` and based on that we produce a random number for age and fill the NAs with it. 

# In[28]:


# Just to make sure there is no more NAs in Age column
full.Age.isnull().sum()


# Often when working with numeric data, we might come across features or attributes which depict raw measures such as values or frequencies. In many cases, often the distributions of these attributes are skewed in the sense that some sets of values will occur a lot and some will be very rare. Besides that, there is also the added problem of varying range of these values. Consider `Age` and `TicketFare` in our dataset.
# 
# In some cases, the view counts will be abnormally large and in some cases very small. Directly using these features in modeling might cause issues. Metrics like similarity measures, cluster distances, regression coefficients and more might get adversely affected if we use raw numeric features having values which range across multiple orders of magnitude. There are various ways to engineer features from these raw values so we can these issues. One method would be **binning**.
# 
#  The operation of binning is used for transforming continuous numeric values into discrete ones. These discrete numbers can be thought of as bins into which the raw values or numbers are binned or grouped into. Each bin represents a specific degree of intensity and has a specific range of values which must fall into that bin. 
#  
#  Before applying that technique to our dataset, we are going to make a visualization of distribution of these two columns.

# In[29]:


full[['Age', 'TicketFare']].describe()


# In[30]:


# Making subplots axes
fig, axes = plt.subplots(2,2,figsize=(12,7))

sns.boxplot(data=full, 
            x='Age', 
            ax=axes[0][0])

sns.countplot(data=full, 
              x='Age', 
              ax=axes[0][1])

sns.boxplot(data=full, 
            x='TicketFare', 
            ax=axes[1][0])

sns.countplot(data=full[['TicketFare']].astype(int), # Only for having less bars
              x='TicketFare', 
              ax=axes[1][1])

# Adjusting xlabelticks to make them more readble
for ax in [axes[0][1],axes[1][1]]:
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontdict={'fontsize':10})


# As we can clearly see, both columns values are skewed and in the case of `TicketFare` there are varied values. So, binning could be a right choice to apply. We start with age and try to transform it into discrete meaningful values and then we apply appropriate ranges for ticket fares.

# In[31]:


# Defining age ranges
age_bin = [0,2,15,40,55,80]

# Set label for age ranges
age_bin_labels = ['infant','kid','young','mid-age','old']

# Overide numeriuous age value with discrete bins
full['Age'] = pd.cut(np.array(full['Age']), 
                     bins=age_bin, 
                     labels=age_bin_labels)

# Checking value counts in each new generated age range
full.Age.value_counts().sort_values()


# But why we did such a thing! what was wrong with the continuous value of age and why we convert them to the age bin. Listen, what will be our ultimate goal! predicting the passengers who survived, right! and to reach our goal we are going to construct models which are using all the independent values including Age to predict the outcome. If we think about situations like titanic tragedy, probably we can make some guesses like how cruise crews decided who should get into rescue boats first, most probably infants, right. By this point of view, it couldn't be a significant difference for people in the same range ages. Maybe young people could help themselves but would be a little chance for elders.

# We can use the same technique and apply it on `TicketFare` column. It shouldn't be a large difference for a passenger which paid 35 for ticket with one which paid 36 or 37, right?. 
# The slight difference in the case of `TicketFare` is, we saw how skewed data distribution is and if we want to bin it as what we've done for age (even size bins), we will end up with bins which might be densely populated (e.g. around bin which contains values around 35), and some bins might be sparsely populated (e.g. bins of values bigger than 300).
# 
# **Adaptive binning** is a safer and better approach where we use the data distribution itself to decide what should be the appropriate bins.
# 
# **Quantile** based binning is a good strategy to use for adaptive binning. Quantiles are specific values or cut-points which help in partitioning the continuous values distribution of a specific numeric field into discrete contiguous bins or intervals.
# 
# Ticket fare varies from 0! to 512, let's take a 3-Quantile quartile based adaptive binning scheme:

# In[32]:


# we need to have 3 quantiles
quantile_list = np.linspace(0,1,4,endpoint=True)

# Finding quiatiles in data
fare_quantiles = full['TicketFare'].quantile(quantile_list)
print(fare_quantiles)

# Visualise the binning
fig, ax = plt.subplots(figsize=(15,10))

full[['TicketFare']].astype(int).hist(bins=50,
                                      color='b',
                                      alpha=.5,
                                      ax=ax)

# Drawing quantile lines in red over the histogram
for q in fare_quantiles:
    qvl = plt.axvline(q,color='r')
    
ax.legend([qvl],['Quantiles'],fontsize=18,loc='upper center')
ax.set_title('Ticket Fare Histogram with Quntiles', fontsize=25)
ax.set_xlabel('Ticket Fare', fontsize=18)
ax.set_ylabel('Frequency', fontsize=18)


# In[33]:


# Using qunatile binning to bin each each of passenger ticket fare
quantile_label = ['Cheap','Regular','Premium']
full['TicketFare'] = pd.qcut(full['TicketFare'],
                             q=quantile_list,
                             labels=quantile_label)

full.TicketFare.value_counts().sort_values()


# Let's take a look again to the dataset after making recent changes:

# In[34]:


full.head()


# The other feature engineering which we can make would be identifying solo-travellers and Family-travelers as well as family size in the latter case. Based on same reason as what we have done we age column, we are going to use the same approach for `Sibling_Spouse` and `Parent_Children` columns. Perhaps knowing the value of those columns separately doesn't help that much for predicting survival status, but trying to figure out the whole onboard family size of passenger give us a better indication.

# In[35]:


# Calculating family size for each passenger
full['FamilySize'] = full['Sibling_Spouse'] +                      full['Parent_Children'] +                      1 # include the passeger itself


# In[36]:


# Adding a new column for indicating solo-travellers
full.loc[full.FamilySize > 1, 'IsAlone'] = 0
full.loc[full.FamilySize <= 1, 'IsAlone'] = 1
full['IsAlone'] = full['IsAlone'].astype(int)


# In[37]:


full.loc[:891,['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived', ascending=False)


# Interestingly, passengers who were members of a family in 4, had a chance to survive around **73%**.

# In[38]:


full.loc[:891,['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean().sort_values(by='Survived', ascending=False)


# Moreover, a really little chance for solo-travelers, only **30%**.

# In[39]:


# Dropping unnecessary columns after these columns engineerings
full.drop(['Sibling_Spouse','Parent_Children'], axis=1, inplace=True)


# In[40]:


# Check dataset again
full.head()


# So far, we have been working on continuous numeric data and you have also seen various techniques for engineering features from the same. We will now look at another structured data type, which is categorical data. Any attribute or feature that is categorical in nature represents discrete values that belong to a specific finite set of categories or classes. Category or class labels can be text or numeric in nature. Usually, there are two types of categorical variables—**nominal** and **ordinal**.
# 
# Nominal categorical features are such that there is *no concept of ordering among the values*, i.e., it does not make sense to sort or order them. Features like, `Sex`, `Embarked` and `Title` are some examples of nominal attributes. Ordinal categorical variables can be ordered and sorted on the basis of their values and hence these values have specific significance such that their order makes sense. Examples of ordinal attributes in our dataset are:  `Age` and `TicketFare`.
# 
# We will start with nominal features. We can do it either manually by just mapping categories to numeric values or using **scikit-learn** labelEncoder method.

# In[41]:


for nom_feature in ['Sex','Embarked','Title']:
    gle = LabelEncoder()
    labels = gle.fit_transform(full[nom_feature])
    report = {index: label for index,label in enumerate(gle.classes_)}
    full[nom_feature] = labels
    print(nom_feature,':',report,'\n','-'*50)


# Now we can deal with Ordinal features. Ordinal features are similar to nominal features except that order matters and is an inherent property with which we can interpret the values of these features.
# 
# Unfortunately, since there is a specific logic or set of rules involved in case of each ordinal variable, there is no generic module or function to map and transform these features into numeric representations. Hence we need to hand-craft this using our own logic, which is depicted in the following code snippet.

# In[42]:


# Mapping ordinal values in Age column
age_ord_map = {'infant':0, 'kid':1, 'young':2, 'mid-age':3, 'old':4}
full.Age = full.Age.map(age_ord_map)

# Mapping ordinal values in TicketFare column
tf_ord_map = {'Cheap':0, 'Regular':1, 'Premium':2}
full.TicketFare = full.TicketFare.map(tf_ord_map)


# In[43]:


# Checking dataset
full.head(10)


# We have mentioned several times in the past that Machine Learning algorithms usually work well with **numerical values**. You might now be wondering we already transformed and mapped the categorical variables into numeric representations in the previous sections so why would we need more levels of encoding again? The answer to this is pretty simple. If we directly fed these transformed numeric representations of categorical features into any algorithm, the model will essentially try to interpret these as raw numeric features and hence the **notion of magnitude** will be wrongly introduced in the system.
# 
# There are several schemes and strategies where dummy features are created for each unique value or label out of all the distinct categories in any feature. We are going to use a method which is called **One Hot Encoding Scheme**.
# 
# Considering we have numeric representation of any categorical feature with m labels, the one hot encoding scheme, encodes or transforms the feature into m binary features, which can only contain a value of 1 or 0. Each observation in the categorical feature is thus converted into a vector of size m with only one of the values as 1 (indicating it as active).

# In[44]:


# encode all the categorical features using one-hot encoding scheme
list_category_features = ['TicketClass','Sex','Age','TicketFare','Embarked','Title']
dummy_features = pd.get_dummies(full[list_category_features], columns=list_category_features)

# Drop all the features before transforming to dummy variables
full.drop(list_category_features, axis=1,inplace=True)

# Merging remaining columns with dummy variables
full = pd.concat([full, dummy_features], axis=1)

# Checking dataset
full.sample(10)


# ## Modeling
# 
# Models can be differentiated on a variety of categories and nomenclatures. A lot of this is based on the learning algorithm or method itself, which is used to build the model. Examples can be the model is linear or nonlinear, what is the output of model, whether it is a parametric model or a non-parametric model, whether it is supervised, unsupervised, or semi-supervised, whether it is an ensemble model or even a Deep Learning based model.
# In our case what we already now is the problem would be a **Logistic Regression** or/and **Classification**, moreover because of having labeled data we know that we're dealing with **Supervised learning**. By having these information we can pick some of the related models and evaluate each of them in order to making best model.
# 

# In[45]:


train_df_new = full.iloc[:891]
y = train_df_new['Survived']
X = train_df_new.drop(['Survived'], axis=1)

test_df_new = full.iloc[891:]
test_df_new = test_df_new.drop(['Survived'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=42) 
print(X_train.shape, X_test.shape)


# Now, we are going to try couple of models which could work on our problem:
# 
# We will start with Logistic Regression and we'll discuss about how to evaluate model and get reports then picking better models and tuning them.
# 
# **Logistic Regression**:
# Logistic Regression is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the logistic regression model predicts P(Y=1) as a function of X.

# In[46]:


# train and build the model
logistic = LogisticRegression()
logistic.fit(X_train, y_train)

# Make the prediction values
y_pred = logistic.predict(X_test)

# Checking model score
print('Logistic Regression model score:',
      np.round(logistic.score(X_test, y_test), 3))


# #### Evaluate model:
# 
# 
# 
# **Confusion matrix** is one of the most popular ways to evaluate a classification model. Although the matrix by itself is not a metric, the matrix representation can be used to define a variety of metrics, all of which become important in some specific case or scenario. A confusion matrix can be created for a binary classification as well as a multi-class classification model.
# 
# 
# **Accuracy**: This is one of the most popular measures of classifier performance. It is defined as the overall accuracy or proportion of correct predictions of the model.
# `NOTE`: Scikit-learn's models score method are representing model accuracy. 
# 
# **Precision**: Precision, also known as positive predictive value, is another metric that can be derived from the confusion matrix. It is defined as the number of predictions made that are actually correct or relevant out of all the predictions based on the positive class.
# 
# 
# **Recall**: Recall, also known as sensitivity, is a measure of a model to identify the percentage of relevant data points. It is defined as the number of instances of the positive class that were correctly predicted.
# 
# **F1 Score**: There are some cases in which we want a balanced optimization of both precision and recall. F1 score is a metric that is the harmonic mean of precision and recall and helps us optimize a classifier for balanced precision and recall performance.

# In[47]:


def model_report(y_test, y_pred):
    print('Confusion Matrix:\n',
          metrics.confusion_matrix(y_true=y_test,
                                   y_pred=y_pred,
                                   labels=[0, 1]))
    print('{:-^30}'.format('|'))

    print('{:15}{:.3f}'.format('Accuracy:', 
          metrics.accuracy_score(y_test,
                                 y_pred)))

    print('{:-^30}'.format('|'))

    print('{:15}0:{:.3f}|1:{:.3f}'.format('Precision:', 
          metrics.precision_score(y_test,y_pred,average=None)[0],
          metrics.precision_score(y_test,y_pred,average=None)[1]))

    print('{:-^30}'.format('|'))

    print('{:15}0:{:.3f}|1:{:.3f}'.format('Recall:',
          metrics.recall_score(y_test,y_pred,average=None)[0],
          metrics.recall_score(y_test,y_pred,average=None)[1]))


    print('{:-^30}'.format('|'))

    print('{:15}0:{:.3f}|1:{:.3f}'.format('f1-score:',
          metrics.f1_score(y_test,y_pred,average=None)[0],
          metrics.f1_score(y_test,y_pred,average=None)[1]))
    
    print('{:-^30}'.format('|'))
    
model_report(y_test, y_pred)


# An Alternative way to evaluate models is simply using **classification_report** from metrics package:

# In[48]:


from sklearn.metrics import classification_report

model = LogisticRegression()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
report = classification_report(y_test, predicted, digits=3)
print(report)


# #### Cross-Validation for models:
# 
# There is always a need to validate the stability of your machine learning model. **I mean you just can’t fit the model to your training data and hope it would accurately work for the real data it has never seen before.** You need some kind of assurance that your model has got most of the patterns from the data correct, and its not picking up too much on the noise, or in other words its low on bias and variance.
# 
# ##### K-Fold Cross Validation
# As there is never enough data to train your model, removing a part of it for validation poses a problem of underfitting. By reducing the training data, we risk losing important patterns/ trends in data set, which in turn increases error induced by bias. So, what we require is a method that provides ample data for training the model and also leaves ample data for validation. K Fold cross validation does exactly that.
# 
# In K Fold cross validation, the data is divided into k subsets. Now the holdout method is repeated k times, such that each time, one of the k subsets is used as the test set/ validation set and the other k-1 subsets are put together to form a training set. The error estimation is averaged over all k trials to get total effectiveness of our model. As can be seen, every data point gets to be in a validation set exactly once, and gets to be in a training set k-1 times. This significantly reduces bias as we are using most of the data for fitting, and also significantly reduces variance as most of the data is also being used in validation set. Interchanging the training and test sets also adds to the effectiveness of this method. As a general rule and empirical evidence, K = 5 or 10 is generally preferred, but nothing’s fixed and it can take any value.

# In[ ]:


from sklearn import model_selection

kfold = model_selection.KFold(n_splits=10, random_state=0)
model = LogisticRegression()

scoring = 'accuracy'
results = model_selection.cross_val_score(model, 
                                          X, 
                                          y, 
                                          cv=kfold, 
                                          scoring=scoring)

print("Average Accuracy: {:.3f}".format(results.mean()))


# Less accuracy than what we got in non-cv approach, right? but we have more confidence than or model would have less bias or virance in dealing with never seen data and we could consider it as a more realiable model than simple non-cv model.

# ### Moving ahead by constructing models and comparing them 

# Now we know what we should do in order to have a more reliable model and what parameters we need to check to pick best models. Let's make a list of all possible models and evaluate each of them first:

# In[ ]:


# Models names
names = ["Nearest Neighbors", 
         "Linear SVM", 
         "RBF SVM", 
         "Gaussian Process",
         "Decision Tree", 
         "Random Forest", 
         "Neural Net", 
         "AdaBoost",
         "Naive Bayes", 
         "Logistic Regression"]

# Models instances
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LogisticRegression()]

# A placeholder for all results
results = {}

# Iterate over all models and put the score and confusion matrix into the results dictionary.
for name, clf in zip(names, classifiers):
    results[name] = {}
    results[name]['Score'] = model_selection.cross_val_score(clf,
                                                             X, 
                                                             y, 
                                                             cv=kfold, 
                                                             scoring=scoring).mean()
    
    results[name]['Confusion_matrix'] = metrics.confusion_matrix(y,
                                                                 model_selection.cross_val_predict(clf,
                                                                                                   X,
                                                                                                   y,
                                                                                                   cv=kfold))


# In[ ]:


# Sort models based on accuracy
for clf_name in sorted(names, key=lambda x: (results[x]['Score'])):
    print('{:19} :{:.3f}'.format(clf_name, results[clf_name]['Score']))


# Between all models, `Decision Tree` and `Logistic Regression` performed slightly better than others. As you might recall based on what we discussed before, accuracy wouldn't be always a good choice if we have imbalance outcomes. So, we need to consider confusion matrix of each classifier to figure out which model performs better to identifying dead passnegers and which survived passengers. 

# In[ ]:


fig, axes = plt.subplots(5,2, figsize=(10,15))
# Adjust padding between plots
plt.tight_layout()

counter=0
for clf_name in sorted(names, key=lambda x: (results[x]['Score'])):
    sns.heatmap(results[clf_name]['Confusion_matrix'],
                ax=axes[counter % 5, math.floor(counter / 5)],
                annot=True,
                fmt='2.0f',
                square=True,
                annot_kws={"size": 20},
                cmap="coolwarm")
    axes[counter % 5, math.floor(counter / 5)].set_title(clf_name)
    counter += 1


# Heatmaps showed us, `Decision Tree` model has the best performance in predicting survivals and `Neural Net` and `Random Forest` models both have a higher chance in correctly predicting dead passengers.
# 
# What we are going to do next is selecting `Random Forest` and tuning their hyperparameters by using `GridSearchCV`.

# In[ ]:


# Setting values for hyperparameters
hyper_params={'max_depth':range(5,21,5),
              'min_samples_split':range(2,9,2),
              'min_samples_leaf':range(1,6,1),
              'max_leaf_nodes':range(2,11,1)}

grid = GridSearchCV(RandomForestClassifier(random_state=1),
                   param_grid=hyper_params)

grid.fit(X,y)
print('Best Score: {:.4f}'.format(grid.best_score_))

print('Best Parameters setting:',grid.best_params_)


# We boosted the model and reached to the accuracy of 83%. Now we got ready to use the model on the competition test dataset and export the dataframe:

# In[ ]:


ranfor_model = RandomForestClassifier(random_state=1,
                                      max_depth=10, 
                                      max_leaf_nodes=10, 
                                      min_samples_leaf=2, 
                                      min_samples_split=2)

ranfor_model.fit(X,y)

y_pred = ranfor_model.predict(test_df_new)


# In[ ]:


submission = pd.DataFrame({
    "PassengerId": range(892,1310),
    "Survived": y_pred
})
submission.to_csv('titanic.csv', index=False)

