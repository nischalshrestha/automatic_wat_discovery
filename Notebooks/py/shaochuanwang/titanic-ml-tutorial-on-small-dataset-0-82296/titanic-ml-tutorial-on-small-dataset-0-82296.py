#!/usr/bin/env python
# coding: utf-8

# 
# # Introduction
# Titanic is a very interesting dataset; it contains only 891 rows in its training dataset. This almost means that for all the proposed features and ML models, we must pay extra attention to overfitting. We need to find the simplest model that generalizes well. In fact, my best public score comes from my lowest cross validation accuracy. All of my previous submissions had much higher cross validation scores, which meant they were all overfitted.
# 
# For exploring the Titanic dataset, I found the following kernels useful:
# * [Divide and Conquer](https://www.kaggle.com/pliptor/divide-and-conquer-0-82296)
# * [Exploring Survival on the Titanic](https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic)
# 
# In this kernel, the contribution has two-fold:
# 1. I introduce a clustered-based feature related to ticket number, assuming its ticket system resembles how we book the ticket for an airplane.
# 2. I try to simplify features by inspecting their distributions in order to reduce the model complexity.
# 
# Here we go!

# In[1]:


# Standard library import for python.

import numpy as np 
import pandas as pd 

import os

import seaborn as sns

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

# Let's make sure data source files are there.
print(os.listdir("../input"))


# In[3]:


# Load in the train and test datasets from the CSV files.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data = pd.concat([train, test])
train.shape


# # Feature Preprocessing and Engineering
# When looking at the training dataset, it only contains 891 rows, which is a very small dataset. This almost means that this competition almost always will need to fight against the overfitting. If we have lots of data, algorithm such as GBDT would help us find a good split and thus no need to bucketize the features. However, since the training dataset is so small, the strategy is to find a **simplest** model which has lowest validation error.
# 
# 
# ### Here we mainly doing the following:
# 1. Handling missing data.
# 2. Categorical feature encoding. 
# 3. Extracting features from texts. 

# In[4]:


sex_label = LabelEncoder()
cabin_label = LabelEncoder()
embarked_label = LabelEncoder()
family_name_label = LabelEncoder()
title_label = LabelEncoder()
title_remap_label = LabelEncoder()

data['Sex_Code'] = sex_label.fit_transform(data.Sex)
data['Cabin_Prefix'] = data.Cabin.str.get(0).fillna('Z')
data['Cabin_Code'] = cabin_label.fit_transform(data.Cabin.str.get(0).fillna('Z'))
data['Has_Cabin'] = (data.Cabin.str.get(0).fillna('Z') != 'Z').astype('int32')
data['Embarked_fillZ'] = data.Embarked.fillna('Z')
data['Embarked_Code'] = embarked_label.fit_transform(data.Embarked.fillna('S')) # 'S' has highest occurrence. 
data['FamilySize'] = data.Parch + data.SibSp + 1
data['BigFamily'] = data.FamilySize.apply(lambda s: s if s < 5 else 5)
data['IsAlone'] = data.FamilySize == 1
data['FamilyName'] = data.Name.str.extract('(\w+),', expand=False)
data['FamilyName_Code'] = family_name_label.fit_transform(data.FamilyName)
data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand=False)
mapping = {
    'Mlle': 'Miss',
    'Ms': 'Miss', 
    'Dona': 'Mrs',
    'Mme': 'Miss',
    'Lady': 'Mrs', 
    'Capt': 'Honorable', 
    'Countess': 'Honorable', 
    'Major': 'Honorable', 
    'Col': 'Honorable', 
    'Sir': 'Honorable', 
    'Don': 'Honorable',
    'Jonkheer': 'Honorable', 
    'Rev': 'Honorable',
    'Dr': 'Honorable'
}
data['Title_Remap'] = data.Title.replace(mapping)
data['Title_Code'] = title_label.fit_transform(data.Title)
data['Title_Remap_Code'] = title_remap_label.fit_transform(data.Title_Remap)
data.Age = data.Age.fillna(data.Age.median())
data.Fare = data.Fare.fillna(data.Fare.median())


# In[5]:


data.head(3)


# # Feature Exploration
# The purpose of feature exploration is to gain the insight of the engineered features, and thus may be able to help us come up with better feature or drop unuseful features.
# 
# Here is the utility function which I draw the survival rate conditioned on feature numbers, and also plot their distribution with histogram. 

# In[9]:


def inspect_feature_plot(data, feat):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 6))
    sns.barplot(data[feat], data.Survived, ax=ax1)
    sns.countplot(data[feat], ax=ax2)


# # Sex
# It's not a secret that females were more likely to be rescued in Titanic, and thus a very good feature candidate.

# In[10]:


inspect_feature_plot(data, 'Sex')


# ## Cabin
# Here we have used Cabin prefix as features; the original idea was assuming that Cabin prefix could have mapped the corresponding location of the ship, and there determined the survival rate. However, the following chart shows **Cabin** prefix is too granular, and may suffer from overfitting. For example, 'G', 'A', 'T' are too few to be statistically meaningful. Instead, **Has_Cabin** looks like a better and simpler feature.

# In[13]:


inspect_feature_plot(data, 'Cabin_Prefix')


# In[14]:


inspect_feature_plot(data, 'Has_Cabin')


# ## Embarked
# Very few row has null 'Embarked', and those entries turned out to be survived. It is better to fill null with 'S', which occurred the frequently (and thus more likely to be guessed correctly when this feature is missing).

# In[16]:


inspect_feature_plot(data, 'Embarked_fillZ') 


# ## Family - Family Size, Family Name, Family Survived
# Family size looks like a good feature, but the number of big family is few, and thus let's squash it to create another feature **BigFamily** which squashes >=5, i.e. [1, 2, 3, 4, >=5], which is more granular than **IsAlone**. 
# 
# **FamilySurvived** feature gave an estimate that if you know one of your family member survives, how likely you'll survive? This assumes family tends to move together.

# In[21]:


inspect_feature_plot(data, 'FamilySize') 


# In[22]:


inspect_feature_plot(data[:891], 'BigFamily')
print(data[['PassengerId', 'BigFamily']].groupby('BigFamily').count().rename(columns={'PassengerId': 'Number'}))


# In[23]:


for feat in ('IsAlone', 'Parch', 'SibSp'):
    inspect_feature_plot(data[:891], feat)


# In[24]:


m = data[['FamilyName', 'Survived']].groupby('FamilyName').max()
c = data[['FamilyName', 'PassengerId']].groupby('FamilyName').count()
m = m.rename(columns={'Survived': 'FamilySurvived'})
c = c.rename(columns={'PassengerId': 'FamilyMemberCount'})
m = m.where(m.join(c).FamilyMemberCount > 1, other=-1, axis=1).fillna(-1).join(c)
m.FamilySurvived = m.FamilySurvived.astype('int32')

joined_data = data.join(m, on='FamilyName')


# ### Family Survived
# Family tends to move together, and thus, if one of the family member survivied, and how does that impact the person being survived? We fill with -1, if the family member count <=1, meaning, we don't know much about that family, and hence we try to tell the model little information when the member count is low.

# In[30]:


inspect_feature_plot(joined_data[:891], 'FamilySurvived')


# In[31]:


inspect_feature_plot(joined_data[:891], 'FamilyMemberCount')


# # Title
# Title are extracted from the name of the passenger. There are many rare titles which tell us little information. Thus, we'll need to re-map them. Out of rare titles, there are two categories: **foreign language** and **honorable titles**. For foreign language such as *Mlle*, we'll remap it into *Ms* (English), where notable titles like Rev, we remap them into *Honorable*.

# In[32]:


for feat in ('Title', 'Title_Remap'):
    inspect_feature_plot(data, feat)
    print(data[['PassengerId', feat]].groupby(feat).count().rename(columns={'PassengerId': 'Number'}))


# # Pclass
# Pclass is a feature that can capture if the passenger is a rich person. As shown below, the fare is much more expensive in class 1 than class 3. In class 3, Fare distribution looks non-separable for survival, whereas class 1, and 2 seems to be a good predictor, and hence a good feature candidate.

# In[33]:


feat = 'Pclass'
inspect_feature_plot(joined_data, feat)
print(data[['PassengerId', feat]].groupby(feat).count().rename(columns={'PassengerId': 'Number'}))


# ### Price distribution for Pclass

# In[34]:


facet = sns.FacetGrid( data, hue = 'Survived' , row = 'Pclass', aspect = 8)
facet.map( sns.distplot , 'Fare' )
facet.set( xlim=( 0 , data[ 'Fare' ].max() ) )
facet.add_legend()


# ## Ticket Number
# Most ticket system uses serial number. Here we assume that if ticket number is close to one another, then they are probably bought around the same time. Note that for those tickets bought around the same time, their location in ship would be likely to be close as well, and hence we're doing **hierachical clustering** of ticket number. We assign the cluter number as the new feature.

# In[35]:


ticket = data.Ticket.str.extract('(\d+$)', expand=False).fillna(0).astype(int).ravel()
n_cluster = []
for max_d in range(1,201,2):
    Z = linkage(ticket.reshape(data.shape[0], 1), 'single')
    clusters = fcluster(Z, max_d, criterion='distance')
    data['Ticket_Code'] = clusters
    n_cluster.append( data.Ticket_Code.unique().shape[0] )

len(n_cluster)
d = pd.concat([pd.Series(n_cluster, name="Cluster_Count", dtype='int32'),
               pd.Series(range(1,201,2), name="Distance_Threshold", dtype='int32')], axis=1)
sns.regplot('Distance_Threshold', 'Cluster_Count', d)


# In[36]:


optimal_d = 20
Z = linkage(ticket.reshape(data.shape[0], 1), 'single')
clusters = fcluster(Z, optimal_d, criterion='distance')
joined_data['Ticket_Code'] = clusters


# ### Ticket Number Remap
# Small ticket number cluster doesn't tell us too much information, let's squash them to reduce the complexity.

# In[38]:


import itertools
count = joined_data[['PassengerId', 'Ticket_Code']].groupby('Ticket_Code').count().rename(columns={'PassengerId': 'Number'})
joined_data['Ticket_Code_Remap'] = joined_data.Ticket_Code.replace(dict(zip(count.index[count.Number <= 10], itertools.cycle([0]))))

for feat in ('Ticket_Code_Remap', 'Ticket_Code'):
    inspect_feature_plot(joined_data, feat)


# ### Ticket Number Cluster
# Let's inspect intersting ticket clusters.
# 
# * Deadly ticket clusters: 
#   * Cluster 89. Looks like Sage family is all gone. :(
#   * Cluster 186. Pclass 3 - economic class, mostly single person.
# * Highly survived ticket cluster: 127. Only one man died.

# In[58]:


joined_data[['FamilyName',
      'Name',
      'Age', 
      'Fare', 
      'BigFamily', 
      'Pclass',
      'Has_Cabin',
      'Embarked',
      'Sex', 
      'Title',
      'Ticket_Code',
      'Ticket_Code_Remap',
      'Survived']][joined_data.Ticket_Code==89].sort_values(by='FamilyName')


# In[59]:


joined_data[['FamilyName',
      'Name',
      'Age', 
      'Fare', 
      'BigFamily', 
      'Pclass',
      'Has_Cabin',
      'Embarked',
      'Sex', 
      'Title',
      'Ticket_Code',
      'Ticket_Code_Remap',
      'Survived']][joined_data.Ticket_Code==186].sort_values(by='FamilyName')


# In[60]:


joined_data[['FamilyName',
      'Name',
      'Age', 
      'Fare', 
      'BigFamily', 
      'Pclass',
      'Has_Cabin',
      'Embarked',
      'Sex', 
      'Title',
      'Ticket_Code',
      'Ticket_Code_Remap',
      'Survived']][joined_data.Ticket_Code==127].sort_values(by='FamilyName')


# In[42]:


selected_features = ['Age', 
                     'Fare', 
                     'BigFamily', 
                     'Pclass',
                     'Has_Cabin',
                     'Embarked_Code',
                     'Sex_Code', 
                     'Title_Remap_Code',
                     'Ticket_Code_Remap',
                     'FamilySurvived',
                    ]
one_hot_features = ['Pclass',
                    'BigFamily',
                    'FamilySurvived',
                    'Embarked_Code',
                    'Title_Remap_Code',
                    'Ticket_Code_Remap',
                   ]
selected_data = joined_data[selected_features]
print('Does the following feature contain any NaN? ')
for f in selected_features:
    print('%s: %s' % (f, repr(selected_data[f].isna().any())))


# In[43]:


selected_data_one_hot = pd.get_dummies(selected_data,
                                       columns = one_hot_features)
rescaling_features = ['Age', 'Fare']
std_scaler = StandardScaler()
for f in rescaling_features:
    selected_data_one_hot[f] = std_scaler.fit_transform(selected_data_one_hot[f].values.reshape(-1, 1))

train_x = selected_data[:train.shape[0]]
test_x = selected_data[train.shape[0]:]

train_x_one_hot = selected_data_one_hot[:train.shape[0]]
test_x_one_hot = selected_data_one_hot[train.shape[0]:]
train_y = data[:train.shape[0]].Survived


# In[44]:


parameters = {'n_estimators': [10,50,100,200],
              'learning_rate': [0.05, 0.1],
              'max_depth': [2,3,4],
              'min_samples_leaf': [2,3],
              'verbose': [0]}

grid_obj = GridSearchCV(GradientBoostingClassifier(), parameters, scoring = 'roc_auc', cv = 4, n_jobs = 4, verbose = 1)
grid_obj = grid_obj.fit(train_x, train_y)
gb = grid_obj.best_estimator_              
gb


# In[45]:


model = gb.fit(train_x, train_y)
pred_y = gb.predict(train_x)
f1 = f1_score(train_y, pred_y)
acc = accuracy_score(train_y, pred_y)


# In[46]:


f1


# In[47]:


acc


# In[48]:


test_y = pd.Series(gb.predict(test_x), name="Survived", dtype='int32')
results = pd.concat([data[train.shape[0]:].PassengerId, test_y], axis=1)


# In[49]:


results.to_csv("gbdt_csv_to_submit.csv",index=False)


# ### Feature importance
# This gives us estimates that how well our engineered features look like for a GBDT.

# In[51]:


feat_importance = list(zip(train_x.columns.values, gb.feature_importances_))
feat_importance.sort(key=lambda x:x[1])
feat_importance


# # Iterating the Models
# Here, we try to make the iterations efficiently by abstracting the comparsion with a training config. So that we can easily add a new comparision without writing codes.

# In[52]:


training_config = {
    'gbdt': {
        'clf': GradientBoostingClassifier(),
        'parameters': {
            'n_estimators': [10,50,100,200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [2,3,4],
            'min_samples_leaf': [2,3],
        },
        'n_jobs': 4,
        'one_hot': False
    },
    'logit' : {
        'clf': LogisticRegression(),
        'parameters': {
            'penalty': ['l1', 'l2'],
            'C': list(np.arange(0.5, 8.0, 0.1))
        }
    },
    'svm': {
        'clf': LinearSVC(),
        'parameters': {
            'penalty': ['l2'],
            'loss': ['hinge', 'squared_hinge'],
            'C': list(np.arange(0.5, 8.0, 0.1))
        }
    },
    'rf': {
        'clf': RandomForestClassifier(),
        'parameters': {
            'n_estimators': [10,50,100,200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [2,3,4],
            'min_samples_leaf': [2,3],
        },
        'n_jobs': 4,
        'one_hot': False
    },
    'ada': {
        'clf': AdaBoostClassifier(),
        'parameters': {
            'n_estimators': [10,50,100,200],
            'learning_rate': [0.05, 0.1, 0.5, 1.0, 2.0],
        },
        'n_jobs': 4,
        'one_hot': False
    }
}

# Change the following line if you only want to re-run subset of experiments
exp_to_run = training_config.keys()


# In[53]:


results = { 'name': [], 'f1': [], 'accuracy': [] }
train_pred = {}
test_pred = {}
for name in exp_to_run:
    conf = training_config[name]
    clf = conf['clf']
    parameters = conf['parameters']
    n_jobs = conf.get('n_jobs', 1)
    one_hot = conf.get('one_hot', True)

    print('=' * 20)
    print('Starting training:', name)
    grid_obj = GridSearchCV(clf, parameters, scoring = 'roc_auc', cv = 4, n_jobs = n_jobs, verbose = 1)
    train_X = train_x_one_hot if one_hot else train_x
    
    print('Number of Features:', train_X.columns.shape[0])
    grid_obj = grid_obj.fit(train_X, train_y)
    best_clf = grid_obj.best_estimator_ 
    
    print('Best classifier:', repr(best_clf))
    model = best_clf.fit(train_X, train_y)
    pred_y = best_clf.predict(train_X)
    train_pred[name] = pred_y

    f1 = f1_score(train_y, pred_y)
    acc = accuracy_score(train_y, pred_y)
    results['name'].append(name)
    results['f1'].append(f1)
    results['accuracy'].append(acc)
    
    test_X = test_x_one_hot if one_hot else test_x
    test_y = pd.Series(best_clf.predict(test_X), name="Survived", dtype='int32')
    test_pred[name] = test_y
    output = pd.concat([test.PassengerId, test_y], axis=1)
    
    output_filename = name + "_csv_to_submit.csv"
    print('Writing submission file:', output_filename)
    output.to_csv(output_filename, index=False)


# In[54]:


# Hard voting classifier
pred_y = pd.DataFrame.from_dict(train_pred).mean(axis=1) > 0.5
f1 = f1_score(train_y, pred_y)
acc = accuracy_score(train_y, pred_y)
results['name'].append('voting')
results['f1'].append(f1)
results['accuracy'].append(acc)

test_y = pd.Series(pd.DataFrame.from_dict(test_pred).mean(axis=1) > 0.5,
                   name="Survived", 
                   dtype='int32')
output = pd.concat([test.PassengerId, test_y], axis=1)
output_filename = 'voting_csv_to_submit.csv'
print('Writing submission file:', output_filename)
output.to_csv(output_filename, index=False)


# # Results
# Here is a table that compares its accuracy and f1 score with different algorithm.

# In[57]:


pd.DataFrame.from_dict(results)


# In[ ]:




