#!/usr/bin/env python
# coding: utf-8

# # Titanic Dataset
# A classification task, predict whether or not passengers in the test set survived. I started learning to code about 12 months ago in my spare time and thanks to some friends have become interested in data science, this is my first data science project

# ## Initial Exploratory Data Analysis
# 

# In[ ]:


import pandas as pd
train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


test = pd.read_csv('../input/test.csv')
test.head()


# The train and test set have identical features, the target column only being present in the training data.

# In[ ]:


train.info()


# In[ ]:


test.info()


# 891 entries in the training set, with 418 in the test set.

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


print('Age : %d  percent missing values in the training set' % (train['Age'].isna().sum()*100/len(train)))
print('Cabin : %d percent missing values in the training set' % (train['Cabin'].isna().sum()*100/len(train)))


# In[ ]:


train.Ticket.value_counts()[:10]


# In[ ]:


train.Cabin.value_counts()[:10]


# In[ ]:


train.Embarked.value_counts()[:10]


# In[ ]:


train.Pclass.value_counts()


# Ticket and Cabin are questionable features, ticket seems to be a generic alpha numeric value and cabin has over 70% of its values missing, without knowing information about the location of cabins I will discard both of these features.

# ### Data visualisation

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
survivors = train[train['Survived']==1]
not_survivors = train[train['Survived']==0]
survived_class = survivors['Pclass'].value_counts().sort_index()
died_class = not_survivors['Pclass'].value_counts().sort_index()
classes = survived_class.index

fig,ax = plt.subplots(figsize=(12,8))
bar_width = 0.35
bar0 = ax.bar(classes,died_class,bar_width,label='Died')
bar1 = ax.bar(classes+bar_width,survived_class,bar_width,label='Survived')
ax.set_title('Survivals by passenger class')
ax.set_xlabel('Passenger Class')
ax.set_xticklabels(classes)
ax.set_xticks(classes + bar_width / 2)
ax.set_ylabel('Count')
ax.legend()
plt.show()


# Survival rates are significantly worse for those in third class, with first class having the best chances.

# In[ ]:


survived = survivors['Sex'].value_counts().sort_index()
died = not_survivors['Sex'].value_counts().sort_index()

fig,ax = plt.subplots(figsize=(12,8))
bar_width = 0.35
index = pd.Index([1,2])
bar0 = ax.bar(index,died,bar_width,label='Died')
bar1 = ax.bar(index+bar_width,survived,bar_width,label='Survived')
ax.set_title('Survival by sex')
ax.set_xlabel('Sex')
ax.set_xticklabels(survived.index)
ax.set_xticks(index + bar_width / 2)
ax.set_ylabel('Count')
ax.legend()


# Being female gave you a considerably better chance of survivng.

# In[ ]:


import seaborn as sns
ax = plt.figure(figsize=(12,8))
ax = sns.swarmplot(x='Survived',y='Age',data=train)
ax.set_title('Suvival by Age')
ax.set_xticklabels(['Died','Survived'])
ax.set_xlabel('')


# Children were more likely to survive and the over 70s were very unlikely to make it. Young men look to have the worst survival rate.

# In[ ]:


survived_class = survivors['Embarked'].value_counts().sort_index()
died_class = not_survivors['Embarked'].value_counts().sort_index()
index = pd.Index([1,2,3])

fig,ax = plt.subplots(figsize=(12,8))
bar_width = 0.35
bar0 = ax.bar(index,died_class,bar_width,label='Died')
bar1 = ax.bar(index+bar_width,survived_class,bar_width,label='Survived')
ax.set_title('Survivals by port of embarkation')
ax.set_xlabel('Port of embarkation')
ax.set_xticklabels(survived_class.index)
ax.set_xticks(index + bar_width / 2)
ax.set_ylabel('Count')
ax.legend()
plt.show()


# Passengers embarking at port 'C' have the best survival rate of the training data.

# In[ ]:


s_sib = survivors['SibSp'].value_counts().sort_index()
d_sib = not_survivors['SibSp'].value_counts().sort_index()
s_sib = s_sib.reindex(d_sib.index).fillna(0.0)
fig,ax = plt.subplots(figsize=(12,8))
index = s_sib.index
bar_width = 0.25
bar0 = ax.bar(index,d_sib,bar_width,label='Died')
bar1 = ax.bar(index+bar_width,s_sib,bar_width,label='Survived')
ax.set_title('Survival by number of sibling/spouse on board')
ax.set_xlabel('Number of sibling/spouse')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(s_sib.index)
ax.set_ylabel('Count')
ax.legend()


# In[ ]:


sibsp = train['SibSp'].value_counts()
sibsp_survival = survivors['SibSp'].value_counts()
survival_pct_by_sibling = sibsp_survival / sibsp * 100

plt.bar(survival_pct_by_sibling.index,survival_pct_by_sibling)
plt.title('Survival % by number of sibling/spouse on board')
plt.xlabel('Number of sibling/spouse on board')
plt.ylabel('Survival %')


# In[ ]:


s_parch = survivors['Parch'].value_counts().sort_index()
d_parch = not_survivors['Parch'].value_counts().sort_index()
s_parch = s_parch.reindex(d_parch.index).fillna(0.0)

fig,ax = plt.subplots(figsize=(12,8))
index = s_parch.index
bar_width = 0.25
bar0 = ax.bar(index,d_parch,bar_width,label='Died')
bar1 = ax.bar(index+bar_width,s_parch,bar_width,label='Survived')
ax.set_title('Survival by number of parents/children on board')
ax.set_xlabel('Number of parents/children')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(s_parch.index)
ax.set_ylabel('Count')
ax.legend()


# In[ ]:


train['par_ch'] = np.where(train['Parch']==0,0,1)

survivors = train.loc[train['Survived']==1]
not_survivors = train.loc[train['Survived']==0]
d_parch = not_survivors['par_ch'].value_counts().sort_index()
s_parch = survivors['par_ch'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(12,8))
index = pd.Index([1,2])
bar_width = 0.4

bar0 = ax.bar(index,d_parch,bar_width,label='Died')
bar1 = ax.bar(index+bar_width,s_parch,bar_width,label='Survived')
ax.set_title('Survival by having parents/children onboard')
ax.set_xlabel('')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(['No parents/children','One or more'])
ax.legend()


# Having a relative or spouse on board greatly improved your chances of survival

# In[ ]:


ax = plt.figure(figsize=(12,8))
ax = sns.boxplot(x = 'Survived', y = 'Fare', data = train)
ax.set_title('Survival by fare')
ax.set_xticklabels(['Died','Survived'])
ax.set_xlabel('')


# Higher paying passengers were more likely to survive, easily inferred from the passenger class information earlier.

# In[ ]:


plt.figure(figsize=(12,8))
ax = sns.boxplot(x = 'Pclass', y = 'Fare', data = train)
ax.set_title('Fare paid for each Pclass')


# In[ ]:


cols = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare']
corr = train[cols].corr()
corr


# None of the columns have high collinearity, while not being independent variables they may all be important in the final model.

# ## Data Cleaning
# ### Missing values
# Embarked has two missing values, these can be filled with the most frequent value 'S' as we have no other information to go on.
# Fare has one missing value in the test set which can be imputed with the median.
# 
# The age column has a lot of null entries, as we do not wish to discard the column we can pick from a number of methods to impute these missing values.
# 
# 1. Impute with 0.0 : Not applicable in this dataset
# 2. Impute with the mean or median : A common method for values that are known to not be 0, however it risks skewing the dataset.
# 3. Custom imputation : Design a different method for imputing the missing values.
# 
# Visualising the distribution of ages will give us an insight into which method may be appropriate

# In[ ]:


plt.figure(figsize=(12,8))
plt.title('Age of passengers')
plt.xlabel('Age')
plt.ylabel('Count')
train['Age'].hist(bins=30)


# ### Mean and median imputation

# In[ ]:


age_mean = train['Age'].mean()
age_median = train['Age'].median()
print('Mean age : ', age_mean, ' Median age : ',age_median)


# In[ ]:


mean_imp_age = train['Age'].fillna(age_mean)
median_imp_age = train['Age'].fillna(age_median)

plt.figure(figsize=(12,8))
ax1 = plt.subplot(1,2,1)
ax1 = mean_imp_age.hist(bins=30)
plt.title('Mean imputation')
plt.xlabel('Age')

ax2 = plt.subplot(1,2,2)
plt.title('Median imputation')
plt.xlabel('Age')
ax2 = median_imp_age.hist(bins=30)


# This is clearly an unsatisfactory method that will skew the data.
# ### Custom imputation methods
# There are again many custom methods that could be applied to fill in the missing values. Two of which are outlined below.
# 
# Stratified imputation : Imputing values into age brackets in the same ratio of the values present in the age brackets in the training set.
# Name-based impuation : Using information available in the names of passengers, try and predict more accurately which age bracket the passenger falls in to.
# 
# Using a combination of the two methods above, we will look at the titles of passengers and whether they have parents or children ('Parch') on board.

# In[ ]:


male_children = train.loc[train['Name'].str.contains('Master')]
male_children['Age'].describe()


# With a max value of 12, males whose name contains 'Master' are all children.

# In[ ]:


train.loc[train['Sex']=='male'].drop(male_children.index).sort_values(by='Age')[:5]


# By filtering for males and removing those with the title 'Master' we can see here that the youngest remaining male is 11 years old.

# In[ ]:


male_children[male_children['Age'].isnull()]


# These four 'Master's with missing values for age were also travelling with at least one 'Parch', presumably a parent in these cases. We can therefore be confident of these four passengers being 12 or under. 
# 
# A similar inference can be made for females with the title 'Mrs', missing age values here will indicate these passengers not being children. Those females with the title 'Miss' are far more likely to have parents than children when having 0 in the 'Parch' column. We can confirm this hypothesis with a simple visulatisaton.

# In[ ]:


mrs = train['Name'].str.contains('Mrs.')
miss = train['Name'].str.contains('Miss.')
parch = train['Parch']>0
no_parch = train['Parch']==0

fig,axes = plt.subplots(figsize=(12,7))

ax = sns.distplot(train[mrs]['Age'].dropna(),axlabel='Age',label='"Mrs"',kde=False,bins=20)
ax.set_title('Comparison of ages with the titles "Mrs" and "Miss"')
ax = sns.distplot(train[miss]['Age'].dropna(),axlabel='',label="Miss",kde=False,bins=20)
ax.set_yticks([0,5,10,15,20])
plt.legend()


# In[ ]:


fig,axes = plt.subplots(figsize=(12,7))

ax1 = sns.distplot(train[miss & parch]['Age'].dropna(),axlabel='Age',label='"Miss with parch"',kde=False,bins=20)
ax1 = sns.distplot(train[miss & no_parch]['Age'].dropna(),axlabel='',label='"Miss without parch"',kde=False,bins=20)
ax1.set_title('Comparison of ages of "Miss" with and without "parch onboard')
plt.legend()


# While by no means concrete, males with the title 'Master' will be children, males with the title 'Mr' are 11 or over. Females with the title 'Mrs' can be assumed to be 14 or older and females with the title 'Miss' will be on average much younger if travelling with at least one 'Parch'.
# 
# This will give us a more accurate way of imputing the ages than just assigning an age on a simple stratified basis.

# In[ ]:


from sklearn.base import TransformerMixin

class Age_Imputer(TransformerMixin):
    def __init__(self):
        """
        Imputes ages of passengers in the Titanic, values to be imputed will be dependant 
        on passenger titles and the presence of parents or children on board
        """
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        def value_imp(passengers):
            """
            Imputes an age, based on a weighted random choice derived from the non
            null entries in the subsets of the dataset.
            """
            passengers=passengers.copy()
            # Create 3 year age bins
            bins = np.arange(0,passengers['Age'].max()+3,step=3)
            # Assign each passenger an age bin
            passengers['age_bins'] = pd.cut(passengers['Age'],bins=bins,labels=bins[:-1]+1.5)
            # Count totals of age bins
            count = passengers.groupby('age_bins')['age_bins'].count()
            # Assign each age bin a weight
            weights = count/len(passengers['Age'].dropna())
            null = passengers['Age'].isna()
            # For each missing value, give the passenger an age from the age bins available
            passengers.loc[passengers['Age'].isna(),'Age']=np.random.RandomState(seed=42).choice(weights.index,
                           p=weights.values,size=len(passengers[null]))
            return passengers
        master = X.loc[X['Name'].str.contains('Master')]
        mrs = X.loc[X['Name'].str.contains('Mrs')]
        miss = X.loc[X['Name'].str.contains('Miss')]
        no_parch = X.loc[X['Parch']==0]
        parch = X.loc[X['Parch']!=0]
        miss_no_parch = miss.drop([x for x in miss.index if x in parch.index])
        miss_parch = miss.drop([x for x in miss.index if x in no_parch.index])
        remaining_mr = X.loc[X['Name'].str.contains('Mr. ')]
        # Imputing 'Mrs' first, as in cases where passengers have the titles
        # 'Miss' and 'Mrs', they are married so will be in the older category
        name_cats = [master,mrs,miss_no_parch,miss_parch,remaining_mr]
        for name in name_cats:
            X.loc[name.index] = value_imp(name)
        return X


# As an example, I will impute the 'Mrs' values and compare mean imputation and my custom imputation.

# In[ ]:


def value_imp(passengers):
            """
            Imputes an age, based on a weighted random choice derived from the non
            null entries in the subsets of the dataset.
            """
            passengers = passengers.copy()
            bins = np.arange(0,passengers['Age'].max()+3,step=3)
            passengers['age_bins'] = pd.cut(passengers['Age'],bins=bins,labels=bins[:-1]+1.5)
            count = passengers.groupby('age_bins')['age_bins'].count()
            weights = count/len(passengers['Age'].dropna())
            null = passengers['Age'].isna()
            passengers.loc[passengers['Age'].isna(),'Age']=np.random.RandomState(seed=42).choice(weights.index,
                           p=weights.values,size=len(passengers[null]))
            return passengers


# In[ ]:


train2 = train.copy()
mrs = train2.loc[train2['Name'].str.contains('Mrs.')]
train2.loc[mrs.index] = value_imp(mrs)


# In[ ]:


fig,axes = plt.subplots(figsize = (12,7))

ax = sns.distplot(train2.loc[mrs.index]['Age'],label='Custom Imputation',kde=False,bins=20)
ax = sns.distplot(train.loc[mrs.index]['Age'].fillna(value=mrs['Age'].mean()),
                  label='Mean Imputation',kde=False,bins=20
                 )
ax.set_title('Comparison of custom and mean imputation')
plt.legend()


# In[ ]:


train3 = train.copy()
imp = Age_Imputer()
imp.fit_transform(train3)
fig,axes = plt.subplots(figsize = (12,7))

ax = sns.distplot(train.loc[mrs.index]['Age'].dropna(),label='No imputation',kde=False,bins=20)
ax = sns.distplot(train3.loc[mrs.index]['Age'],label='Full custom imputation',kde=False,bins=20)
ax.set_title('Comparison of full custom imputation and no imputation')
plt.legend()


# This method gives both a stratified imputation and by utilising some simple logic of titles of the era it has allowed us to more accurately predict the ages of those passengers with missing values.

# ### Initial modelling
# We will first run a selection of classifiers on the training data with their default values, then choosing the most promising to pursue further with hyperparameter tuning.
# #### Pipeline
# A pipeline is not an essential piece of a project, however it allows easy access to add or remove a feature or tweak a hyperparameter and quickly be able to reproduce results. It will also allow us to implement GridSearchCV and RandomizedSearchCV to automatically test out many different hyperparameters, imputation methods or features. It would also allow quick transformation of any additional training data added to the dataset.
# 
# Given the different scales of numeric values, we will use a standard scaler on all numeric columns. We will encode all categorical labels 

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder(TransformerMixin):
    def __init__(self,columns = None):
        self.columns = columns 

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output
    
class DataFrameSelector(TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names].values

class ValueImputer(TransformerMixin):
    """
    Imputes a fixed value
    """
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X[self.attribute_names] = X[self.attribute_names].fillna('S')
        return X[self.attribute_names]

numerical_atts = ['Age','SibSp','Parch','Fare']
cat_atts = ['Sex','Pclass','Embarked']

num_pipeline = Pipeline([
    ('imputer', Age_Imputer()),
    ('selector', DataFrameSelector(numerical_atts)),
    ('imp', Imputer(strategy='mean')),
    ('scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('cat_imputer', ValueImputer(cat_atts)),
    ('encoder', MultiColumnLabelEncoder(columns=cat_atts)),
    ('selector', DataFrameSelector(cat_atts)),
])


full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])

train_data_prepared = full_pipeline.fit_transform(train)
train_labels = train['Survived']

feature_list = numerical_atts + cat_atts


# In[ ]:


train_data_prepared.shape


# In[ ]:


feature_list


# Importing a selection of models, fitting to the train data and predicting the training labels, using the average score of a 3 fold cross validation to try and avoid overfitting to the training data.

# In[ ]:


from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import cross_val_score

classifiers = [RidgeClassifier(),KNeighborsClassifier(),
              SGDClassifier(
                  max_iter=1000),DecisionTreeClassifier(),RandomForestClassifier(),
              MLPClassifier(max_iter=1000)]
names = ['Ridge','KNN','SGD','Decision Tree','Random Forest','MLP']

for name, classifier in zip(names,classifiers):
    classifier.fit(train_data_prepared,train_labels)
    print('Scores for ',name,' : ',cross_val_score(classifier,train_data_prepared,train_labels,cv=3).mean())
    


# ### Hyperparameter tuning
# The MLP classifier has performed the best on the training data so we will focus on that with some hyperparameter tuning.
# 
# RandomizedSearchCV lets you input a selection of hyperparameters, select a number of searches to make and will randomly select the models hyperparameters. This is a very good option if you are not sure where to start looking for hyperparameter values as it will cover a wide selection of values. Performing many iterations of fitting and predicting this way is however very computationally expensive with large datasets.

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

mlp = RandomizedSearchCV(MLPClassifier(),cv=3,n_iter=20,param_distributions=(
    {'hidden_layer_sizes':[(100,),(200,),(500,),(1000,)],
    'activation' : ['identity', 'logistic', 'tanh', 'relu'],
    'solver' : ['lbfgs','sgd','adam'],
    'alpha' : np.linspace(0,0.001),
    'max_iter' : [200,500,1000,2000],
    }))
mlp.fit(train_data_prepared,train_labels)


# In[ ]:


print('Best score : {}'.format(mlp.best_score_))
mlp.best_params_


# The best parameters returned by the search. These can now be used to make a prediction on the test set, the pipeline now making the job of transforming the test data a simple one.

# In[ ]:


test_prepared = full_pipeline.fit_transform(test)
best_mlp = mlp.best_estimator_
predictions = best_mlp.predict(test_prepared)
predictions[:20]


# To submit the predictions to the kaggle leaderboard a csv must be created.
# 

# In[ ]:


submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})
submission.to_csv('submission.csv',index=False)
submission.head()


# This result scored 0.77033, nearly 80% of passengers correctly predicted.
# ## Next steps
# To try and improve the model further there are several different avenues to explore.
# #### Feature engineering
# Adapting the available features or creating entirely new ones out of the current data.

# In[ ]:


forest = RandomForestClassifier()
forest.fit(train_data_prepared,train_labels)
sorted(zip(forest.feature_importances_,feature_list),reverse=True)


# Given information like this about feature importance we can choose to adapt the age column into categories or make a new feature that is a combination of exisiting ones such as Age/Fare. We could also onehotencode all categorical features to remove any chance of the model inferring relationships between the numbers currently assigned.
# #### Hyperparameter tuning/model selection
# Further fine tuning the existing model or trying different ones, new features may lead to improved performance of different models.
# #### Error evaluation
# Given access to the answers we could categorise the errors that the model made, did it give too many false positives for young women for example. Using this information both the model and input features can be adapted to improve the models accuracy.
