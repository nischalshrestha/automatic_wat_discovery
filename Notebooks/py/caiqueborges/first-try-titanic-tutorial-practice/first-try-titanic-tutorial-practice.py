#!/usr/bin/env python
# coding: utf-8

# # Kaggle Competition: Titanic: Machine Learning from Disaster

# ### A start in Kaggle and Data Science Competitions

# This is my first attemp to participate in a Kaggle Competition. I'm studying Data Science and machine learning for a few months, watching courses in Udemy, Youtube, Alura, and know it's time for test and put into practise some of the things that I've learned and most important: learn even more!
# 
# I've been learning from brazilian teachers Guilherme Silveira (Alura) and Jones Granatyr (Udemy), and also from some other countries people like sentdex, Siraj and Jose Portilla. Now I'm going apply most of the things that I've learned from them. Thanks for all these people that shows us the way for learning Data Science and Machine Learning!
# 
# In this notebook, I'm also really inspired by <a href='https://www.kaggle.com/startupsci/titanic-data-science-solutions/code'>Manav Sehgal's Titanic Data Science Solutions</a> and <a href='https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python/code'>Anisotropic's Introduction to Ensembling/Stacking in Python</a>. They made really good tutorials of how to play with this dataset features and make them valious for Machine Learning models.

# ### Competition description

# "<i>The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.</i>"

# ### Imports 

# Importing necessary (for now) packages for applying Data Science

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns


# ### Reading the csv files and checking the dataframe

# Ok, now let's import the CSV files as dataframes, and take a look at how the data is structured

# In[2]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[3]:


train_df.head()


# ### A function to transform the dataframe

# Here is what I'm thinking: what about build a function, that is going to process the dataframe imported from the csv, to the dataframe that will be used for training and testing the model?
# 
# If all the transformations remain in the same function, it's going to be easier for loading a new dataset and train/test/predict it.

# In[4]:


def transform_df(df):
    #do all the crazy stuff, magic, polymerizations with df
    
    return df


# Maybe that should be fine! But let's build this function little by little (during the feature analysis and transformation), and then implement it in the end of the process.

# ### Let's start the feature analysis

# #### A general look into the data

# In[5]:


train_df.describe()


# In[6]:


train_df.head(20)


# In[7]:


train_df.info()


# Age, Embarked and Cabin have null values.

# #### Checking the correlation between some categorical variables and 'Survived'

# In[8]:


train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[9]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[10]:


train_df[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# #### Removing unnecessary columns

# The columns PassengerId, Ticket and Cabin are not going to be important or revelant to the model.<br/>
# Cabin has a lot of NaN values, so, I think that the best way is to drop it.

# In[11]:


train_df.drop(['PassengerId','Ticket', 'Cabin'], 1, inplace=True)
train_df.head()


# In[12]:


test_df.drop(['PassengerId', 'Ticket', 'Cabin'], 1, inplace=True)
test_df.head()


# In[13]:


test_df.info()


# Fare has 1 missing value in the test dataset!

# ### Let's fill and transform some data

# #### AGE

# In[14]:


train_df['Age'].hist()


# In[15]:


# There are 177 null values, practically 1/5 of the dataset have nan ages!
print('Count:', train_df.shape[0])
print('Null age count:', train_df['Age'].isnull().sum())
print(train_df['Age'].isnull().sum() / train_df.shape[0] * 100,'% of ages are NAN', sep="")


# Let's fix that, by using mean and standard deviation together

# In[16]:


age_mean = train_df['Age'].mean()
age_std = train_df['Age'].std()
age_nan_count = train_df['Age'].isnull().sum()


# In[17]:


age_mean, age_std, age_nan_count


# This code will generate age_nan_count (177) random integer values between age_mean - age_std (15.17) and age_mean + age_std (44.22)

# In[18]:


rand_ages = np.random.randint(age_mean - age_std, age_mean + age_std, age_nan_count)


# In[19]:


fig, (axis1, axis2) = plt.subplots(1,2, figsize=(15,4))
axis1.set_title = 'Original values of ages'
axis2.set_title = 'New values of ages'

# Original values without NAN
train_df['Age'].dropna().astype(float).hist(bins=70, ax=axis1)

# New values with random ages
train_df['Age'][np.isnan(train_df['Age'])] = rand_ages
train_df['Age'].hist(bins=70, ax=axis2)


# In[20]:


print('Now there are {} null values!'.format( train_df['Age'].isnull().sum()) )


# #### Transforming age from continuous to categorical values

# The hist() function brought to us 8 bins on the 'Age' column, transforming our continuous values to 8 categorial values. Let's test this by comparing the relation of these new values to survived

# In[21]:


train_df['AgeCategorical'] = pd.cut(train_df['Age'], bins=8)
train_df['AgeCategorical'].value_counts()

relation_age_survived = train_df[['AgeCategorical', 'Survived']].groupby('AgeCategorical', as_index=False).mean()


# Plotting the results, is visible that children between 0 and 10 years are most likely to survive

# In[22]:


fig, axis1 = plt.subplots(1,1,figsize=(15,5))
sns.barplot(x='AgeCategorical',y='Survived', data=relation_age_survived)


# In[23]:


train_df.loc[train_df['Age'] < 10.367, 'Age'] = 0
train_df.loc[(train_df['Age'] >= 10.367) & (train_df['Age'] < 20.315), 'Age'] = 1
train_df.loc[(train_df['Age'] >= 20.315) & (train_df['Age'] < 30.263), 'Age'] = 2
train_df.loc[(train_df['Age'] >= 30.263) & (train_df['Age'] < 40.21), 'Age'] = 3
train_df.loc[(train_df['Age'] >= 40.21) & (train_df['Age'] < 50.157), 'Age'] = 4
train_df.loc[(train_df['Age'] >= 50.157) & (train_df['Age'] < 60.105), 'Age'] = 5
train_df.loc[(train_df['Age'] >= 60.105) & (train_df['Age'] < 70.0525), 'Age'] = 6
train_df.loc[(train_df['Age'] >= 70.0525) & (train_df['Age'] <= 80), 'Age'] = 7

train_df['Age'] = train_df['Age'].astype(int)

# AgeCategorical isn't necessary anymore
train_df.drop('AgeCategorical', 1, inplace=True)

train_df.head()


# #### NAME

# We have one unique name for each passenger. The only way that we can use this feature, is by extracting the Title of each passenger, like "Mr" or "Miss". For that, we can use the extract() of a str from pandas.

# We're using the regular expression ' ([A-Za-z]+)\.', where the space in the beginning is purposeful (we want to match something that starts with a blank space), the [A-za-z] means any character from A-Z and a-z, and the '+' means any length of letters.

# We have any number of characters between A-Z-a-z, and that ends with a dot. That's our full regular expression for extracting titles from names in this dataset!

# In[24]:


train_df['Name'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[25]:


train_df['Name'].value_counts()


# Let's check the correlation between the title of the person and it's sex (personally, I dont know a lot of these titles, maybe because english isn't my first language, but at least this helps to visualize better, knowing the sex of these)

# In[26]:


pd.crosstab(train_df['Sex'], train_df['Name'])


# In[27]:


train_df['Name'] = train_df['Name'].replace(['Capt', 'Dr','Rev','Mile','Col','Major','Countess','Jonkheer','Mme',                                            'Don', 'Ms','Sir','Capt','Lady', 'Mlle'], 'Low Appearence')


# In[28]:


train_df['Name'].value_counts()


# In[29]:


train_df[['Name','Survived']].groupby('Name', as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[30]:


#in test_df, we need to fillna
train_df['Name'] = train_df['Name'].map({'Mr':0, 'Low Appearence': 1, 'Master': 2, 'Miss': 3, 'Mrs': 4})


# In[31]:


sns.barplot(x='Name', y='Survived', data=train_df)


# #### SIBSP, PARCH (FAMILY)

# Both sibsp and parch are related to the number of family members are with the passanger:

# <i>sibsp: The dataset defines family relations in this way...<br/>
# Sibling = brother, sister, stepbrother, stepsister<br/>
# Spouse = husband, wife (mistresses and fiancés were ignored)<br/>
# 
# parch: The dataset defines family relations in this way...<br/>
# Parent = mother, father<br/>
# Child = daughter, son, stepdaughter, stepson<br/>
# Some children travelled only with a nanny, therefore parch=0 for them.</i>

# Fusing these two columns, we can reduce dimensionally our dataset

# In[32]:


train_df['Family'] = train_df['SibSp'] + train_df['Parch']
train_df.drop(['SibSp','Parch'], 1, inplace=True)


# In[33]:


family_survived_relation = train_df[['Family','Survived']].groupby('Family', as_index=False).mean()


# In[34]:


family_survived_relation.sort_values(by='Survived', ascending=False)


# In[35]:


sns.barplot(x='Family', y='Survived', data=family_survived_relation)


# Let's create another feature, to see if the passenger is alone.

# In[36]:


train_df['Alone'] = train_df['Family'].copy()
train_df['Alone'].loc[train_df['Alone'] == 0] = -1
train_df['Alone'].loc[train_df['Alone'] > 0] = 0

train_df['Alone'].loc[train_df['Alone'] == -1] = 1


# In[37]:


relation_alone_survived = train_df[['Alone','Survived']].groupby('Alone', as_index=False).mean().sort_values(by='Alone', ascending=False)


# In[38]:


train_df['Alone'].value_counts()


# In[39]:


sns.barplot(x='Alone', y='Survived', data=relation_alone_survived)


# <b>Note</b>: After some tests, I finded out that the Family size is better than the Alone feature (using cross validation), increasing the score in the tests by 2%. I decided to remove the 'Alone' feature and keep 'Family'.

# In[40]:


train_df.drop('Alone', 1, inplace=True)
train_df.head()


# #### EMBARKED

# The embarked feature, is about the port of embarkation: <br/>
# C = Cherbourg<br/>
# Q = Queenstown<br/>
# S = Southampton<br/>
# 
# 
# As we saw earlier, Embarked have two null values that we need to fix:

# In[41]:


train_df.info()


# In[42]:


print('There are {} null values in Embarked'.format(train_df['Embarked'].isnull().sum()))

#Seeing the values
train_df['Embarked'].value_counts()


# We'll set the most common value for those null values

# In[43]:


embarked_most_common = train_df['Embarked'].value_counts().idxmax()
train_df['Embarked'].fillna(embarked_most_common, inplace=True)

print('Now there are {} null values in Embarked'.format(train_df['Embarked'].isnull().sum()))


# In[44]:


train_df[['Embarked','Survived']].groupby('Embarked', as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[45]:


train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'Q': 1, 'C': 2}).astype(int)


# Embarked is ready for prediction!

# #### FARE

# 'Fare' has a null value in the test set. We'll give to it the median value of the column

# In[46]:


test_df.info()


# In[47]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)


# In[48]:


train_df['Fare'].hist(bins=4)


# In[49]:


facet = sns.FacetGrid(train_df, col='Survived', aspect=2)
facet.map(plt.hist, 'Fare')
facet.set(xlim=(0, train_df['Fare'].max()))
facet.add_legend


# Since we have a lot of fare with low values and just a few with high values, let's divide categorically using qcut, that's going to considerate this fact for separing values  

# In[50]:


train_df['FareCategorical'] = pd.qcut(train_df['Fare'], 4)
train_df['FareCategorical'].value_counts()


# In[51]:


train_df[['FareCategorical','Survived']].groupby('FareCategorical', as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[52]:


train_df.loc[train_df['Fare'] < 7.91, 'Fare'] = 0
train_df.loc[(train_df['Fare'] >= 7.91) & (train_df['Fare'] < 14.454), 'Fare'] = 1
train_df.loc[(train_df['Fare'] >= 14.454) & (train_df['Fare'] < 31), 'Fare'] = 2
train_df.loc[train_df['Fare'] >= 31, 'Fare'] = 3

train_df['Fare'] = train_df['Fare'].astype(int)
train_df.drop('FareCategorical', 1, inplace=True)


# In[53]:


train_df.head()


# #### SEX

# Actually, we only need to map the 'sex':

# In[54]:


train_df['Sex'] = train_df['Sex'].map( {'male': 0, 'female': 1} )


# In[55]:


train_df.head()


# ### Implementing transform_df

# We finally made all the feature stuff, and now the transform_df can be implemented!
# Let's fullfil it with all the transformations we did so far..

# In[56]:


def transform_df(df):
    #do all the crazy stuff, magic, polymerizations with df
    df.drop(['PassengerId','Ticket', 'Cabin'], 1, inplace=True)
    
    age_mean = df['Age'].mean()
    age_std = df['Age'].std()
    age_nan_count = df['Age'].isnull().sum()
    
    rand_ages = np.random.randint(age_mean - age_std, age_mean + age_std, age_nan_count)
    df['Age'][np.isnan(df['Age'])] = rand_ages
    
    #df['Age'] = pd.cut(df['Age'], bins=8)
    
    df.loc[df['Age'] < 10.367, 'Age'] = 0
    df.loc[(df['Age'] >= 10.367) & (df['Age'] < 20.315), 'Age'] = 1
    df.loc[(df['Age'] >= 20.315) & (df['Age'] < 30.263), 'Age'] = 2
    df.loc[(df['Age'] >= 30.263) & (df['Age'] < 40.21), 'Age'] = 3
    df.loc[(df['Age'] >= 40.21) & (df['Age'] < 50.157), 'Age'] = 4
    df.loc[(df['Age'] >= 50.157) & (df['Age'] < 60.105), 'Age'] = 5
    df.loc[(df['Age'] >= 60.105) & (df['Age'] < 70.0525), 'Age'] = 6
    df.loc[(df['Age'] >= 70.0525) & (df['Age'] <= 80), 'Age'] = 7

    df['Age'] = df['Age'].astype(int)
    
    df['Name'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    df['Name'] = df['Name'].replace(['Capt', 'Dr','Rev','Mile','Col','Major','Countess','Jonkheer','Mme',                                            'Don', 'Ms','Sir','Capt','Lady', 'Mlle'], 'Low Appearence')
    df['Name'] = df['Name'].map({'Mr':0, 'Low Appearence': 1, 'Master': 2, 'Miss': 3, 'Mrs': 4})
    
    df['Name'] = df['Name'].fillna(4).astype(int)
    
    df['Family'] = df['SibSp'] + df['Parch']
    df.drop(['SibSp','Parch'], 1, inplace=True)

    '''df['Alone'] = df['Family'].copy()
    df['Alone'].loc[df['Alone'] == 0] = -1
    df['Alone'].loc[df['Alone'] > 0] = 0

    df['Alone'].loc[df['Alone'] == -1] = 1'''

    #df.drop('Family', 1, inplace=True)
    
    embarked_most_common = df['Embarked'].value_counts().idxmax()
    df['Embarked'].fillna(embarked_most_common, inplace=True)
    
    df['Embarked'] = df['Embarked'].map({'S': 0, 'Q': 1, 'C': 2}).astype(int)
    
    df['Fare'].fillna(df['Fare'].dropna().median(), inplace=True)
    #df['Fare'] = pd.qcut(df['Fare'], 4)
    
    df.loc[df['Fare'] < 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] >= 7.91) & (df['Fare'] < 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] >= 14.454) & (df['Fare'] < 31), 'Fare'] = 2
    df.loc[df['Fare'] >= 31, 'Fare'] = 3

    df['Fare'] = df['Fare'].astype(int)
    
    df['Sex'] = df['Sex'].map( {'male': 0, 'female': 1} )
    
    return df


# And use against the train/test dataset!

# In[57]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[58]:


train_df = transform_df(train_df)
test_df = transform_df(test_df)


# In[59]:


train_df.head()


# In[60]:


test_df.head()


# ### Building the models

# Let's go for the reeeally funny stuff: <b>making predictions</b>!

# The magic begins from the imports:

# In[61]:


from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier


# Separing the training dataset into X (features) and y (label)

# In[62]:


X_train = train_df.drop('Survived', 1)
y_train = train_df['Survived']

#number of cross validation folds
cv_value = 4

model_results = {}

X_train.head()


# In[63]:


y_train.head()


# Converting data to a np matrix, to avoid errors between XGBoost / Pandas

# In[64]:


X_train = X_train.as_matrix()
y_train = y_train.as_matrix()
test_df = test_df.as_matrix()


# We are going to use the "<b>cross_val_score</b>" method, that will divide our X and y in 'cv' times. This method will make predictions 'cv' times, and each time, the test data will be one part that was divided. In our case here, 'cv' is equal 4.
# 
# ![crossvalidation.png](https://www.researchgate.net/profile/Juan_Buhagiar2/publication/322509110/figure/fig1/AS:583173118664704@1516050714606/An-example-of-a-10-fold-cross-validation-cro17.ppm)
# <a href='https://www.researchgate.net/profile/Kiret_Dhindsa/publication/323969239/figure/fig10/AS:607404244873216@1521827865007/The-K-fold-cross-validation-scheme-133-Each-of-the-K-partitions-is-used-as-a-test.ppm'>Image source link</a>
# 
# The blue square is the part that will be the test set in that iteration, and the remainder it's the train data. For each iteration, we change the train set.
# In each iteration we'll have the score (how good the algorithm was in that iteration). When we are finished, we get the mean of all the scores.
# 
# Doing that, we try to avoid 'overfitting'. A model that has overfitting, can only predict well the training data, but when it's in production, with real world data, it won't classify well because it has memorized the training set and can't generalize the predictions.
# 
# Overfitting is a common term in Data Science, and with cross validation, we can try to avoid our model to become overfitted.

# #### Zero R (Dummy classifier)

# Before starting to test serious algorithms to find the best model, let's build a 'dummy' classifier, a thing that I learned from Guilherme Silveira (Alura) and Jones Granatyr (Udemy). That shows us the dummiest prediction we could have: this model will always return the prediction as the class (label) that is most frequent (if in dataset, there are more 'survived' than 'not survived', the algorithm will always predict as 'survived'). 

# The point is that the dummy classifier score, will be our 'base' score. All others algorithms should be better than this Zero R (one rule algorithm) model.

# In[65]:


clf_dummy = DummyClassifier(strategy='most_frequent')

scores = cross_val_score(clf_dummy, X=X_train, y=y_train, cv=cv_value)

model_results['DummyClassifier'] = np.mean(scores)

np.mean(scores)


# Our base score is <b>61%</b>. Any model that the prediction score is less than 61%, has no worth. It would be easier just to guess always the same label, every time that we would like to predict new data.

# #### Multinomial Naive Bayes
# Multinomial Naive Bayes is based in the frequency that certain events occurs and it's probabilities. <a href='https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes'>Read more here for better explanation</a>

# In[66]:


clf_m_naive_bayes = MultinomialNB()
scores = cross_val_score(clf_m_naive_bayes, X=X_train, y=y_train, cv=cv_value)

model_results['MultinomialNB'] = np.mean(scores)

np.mean(scores)


# #### SVC (Support Vector Machines)
# The SVC implements the SVM (Support Vector Machines) algorithm. It's tries to find the best hyperplane to separe the labels (points in a plane). <a href='http://scikit-learn.org/stable/modules/svm.html'>Read here for more information</a> and also <a href='https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/'>here</a>
# 
# ![svm.png](https://openi.nlm.nih.gov/imgs/512/252/2731864/PMC2731864_kjr-10-464-g003.png)
# <a href='https://www.quora.com/How-can-I-use-a-Support-Vector-Machine-in-regression-tasks-SVM'>Image Link</a>

# In[67]:


clf_svc = SVC()
scores = cross_val_score(clf_svc, X=X_train, y=y_train, cv=cv_value)

model_results['SVC'] = np.mean(scores)

np.mean(scores)


# #### Random Forest

# In Data Science, there is an algorithm that create a decision tree for classification. For "Random Forest", it's going to create a lot of these decision trees, each one giving it's own classification, and then the most voted is our final answer:
# 
# ![randomforest.png](https://cdn-images-1.medium.com/max/592/1*i0o8mjFfCn-uD79-F1Cqkw.png)
# <a href='https://medium.com/@williamkoehrsen/random-forest-simple-explanation-377895a60d2d'>Link for image and explanation</a>

# In[68]:


# Estimators = number of trees
clf_forest = RandomForestClassifier(n_estimators=200, max_depth=4, n_jobs=-1)
scores = cross_val_score(clf_forest, X=X_train, y=y_train, cv=cv_value)

model_results['RandomForest'] = np.mean(scores)

np.mean(scores)


# #### AdaBoost

# AdaBoost is a metaheurisct algorithm, that iterates through the dataset, adjusting weights and trying to become even better each iteration (at least, that what I understood, this is the first time I'm seeing this algorithm haha, I'm going to dig deeper). <a href='http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html'>Check it here</a>

# In[69]:


clf_ada = AdaBoostClassifier()
scores = cross_val_score(clf_ada, X=X_train, y=y_train, cv=cv_value)

model_results['AdaBoost'] = np.mean(scores)

np.mean(scores)


# #### KNN (K-Nearest Neighbors)

# K-Nearest Neighbors, is based on the distance between the points (each passenger, in this case). We introduce a new passenger to the algorithm, and it's going to compare the 'k' (number of neighbors that are going to be analyzed) nearest neighbors (the other passengers from training data). We plot that new passenger, and compare with it's neighbors. If k is equal to 5, and 3 out of 5 survived (majority), it's classified as a surviver.
# ![knn.png](http://www.computacaointeligente.com.br/wp-content/uploads/2017/03/imgKnn.jpg)
# <a href='http://www.computacaointeligente.com.br/algoritmos/knn-k-vizinhos-mais-proximos/'>Image source page</a>
# 
# In this image example, the new point (star) is going to be classified as the orange point 'Rótulo A', because there are more orange points than blue, considering that we're looking for the 7 nearest points.

# In[70]:


clf_knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(clf_knn, X=X_train, y=y_train, cv=cv_value)

model_results['KNN'] = np.mean(scores)

np.mean(scores)


# #### Neural Network

# Artificial Neural Networks are inspired in human brain. It has nodes that imitates neurons.
# 
# These neurons (that are called perceptrons) are divided into layers: input, hidden and output. Input neurons are the data being received, the hidden layers do all the processing based in weights that are updated each iteration, and the output layer, our response. I highly suggest to search more if you never heard about it. This image is showing a normal neural network, there are other types like Convolutional NN (that can be used to classify images), Recurrent NN that can be used for building chatbots. It's the most famous algorithm today, because of it's variations and what they enable us to create.
# 
# I founded this page, that looks very describing about ANN, take a look: <a href='https://www.doc.ic.ac.uk/~nd/surprise_96/journal/vol4/cs11/report.html'>Page link</a>
# 
# MLP stands for Multi-layer perceptron.

# In[71]:


clf_nn = MLPClassifier(hidden_layer_sizes=(5,3), max_iter=300, learning_rate_init=0.01, random_state=101)
scores = cross_val_score(clf_nn, X=X_train, y=y_train, cv=cv_value)

model_results['Neural Network'] = np.mean(scores)

np.mean(scores)


# #### XGBoost

# XGBoost is also based in decision trees, but this time, each tree gives a score of how probable of being a passenger that survived or not(in our case). 
# 
# This algorithm I do not know well either, but I founded this video from <a href='http://xgboost.readthedocs.io/en/latest/model.html'>Two Minute Papers (click here)</a>, that gives a quick introduction on how XGBoost works

# In[72]:


clf_xgb = XGBClassifier(learning_rate=0.01)

scores = cross_val_score(clf_xgb, X=X_train, y=y_train, cv=cv_value)

model_results['XGBoost'] = np.mean(scores)

np.mean(scores)


# ### Let's see the results!

# Transforming the results from each algorithm into a DataFrame

# In[73]:


results = pd.DataFrame(list(model_results.items()), columns=['Model','Score'])
results.sort_values(by='Score', ascending=False)


# Now that we know the best algorithms for the job, we'll gather them in a list, and then train each one using all the train dataset:

# In[82]:


#There are algorithms that are really close to each other, like AdaBoost, MultinomialNB, NN and KNN. Let's pick these, thinking about
# diferent types algorithms trying to classify:
models_list = [clf_xgb, clf_forest, clf_nn, clf_svc, clf_knn]
for model in models_list:
    model.fit(X_train, y_train)


# Our submission CSV needs all the passengers id. Importing the test.csv, only to get the 'PassengerId' column, and then using it to identify the passengers after our prediction

# In[75]:


original_test_df = pd.read_csv('../input/test.csv')
passengers_id = original_test_df['PassengerId']
len(passengers_id)


# In[76]:


from collections import Counter

y_predict = []

y = []

#for each model in the list, we are going to predict ALL the test_df, and the append it to 'y'
for model in models_list:
    y.append( model.predict(test_df[:]))

y = np.array(y)

# 5 columns for each algorithm prediction, 418 lines for each passenger prediction
print(y.shape)


# Just for easy understanding, let's see what will be the final prediction of the first two passengers from test_df.

# In[77]:


# y[:,0] contains the predictions for the first passenger, by the algorithms from models_list.
# As we can see, all of them predicted as '0' (Not Survived)
# Counter(y[:,0]).most_common(3) shows the most commom value, and it's count [(value, count)]: 

print(y[:,0], Counter(y[:,0]).most_common(1))
# [(0, 5)] = We can say that the value '0' it's the most common, with 5 appearences in total.

print(y[:,1], Counter(y[:,1]).most_common(1))
# [(1, 3)] = We can say that the value '1' it's the most common, with 3 appearences in total.


# In[78]:


# for each prediction (5 different predictions for each passenger),
# we are counting which one (survived/not survived) has most votes.
for i in range(y.shape[1]):
    y_predict.append([passengers_id[i], Counter(y[:,i]).most_common(3)[0][0]]) 


# These are our first 10 predictions!

# In[79]:


y_predict[:11]


# ### Submission

# Alright! We have our predictions of the test dataset! Let's create our CSV to make the upload on Kaggle! Remember that it needs the header!

# In[80]:


submission = pd.DataFrame(data=y_predict, columns=['PassengerId', 'Survived'])
submission.head()


# In[81]:


submission.to_csv('submission.csv', header=True, index=False)


# ### That's it! 

# This was my first try in a 'serious' project, if you have any sugestions, critics, tips, feel free for sharing it!
