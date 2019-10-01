#!/usr/bin/env python
# coding: utf-8

# **INTRODUCTION :**
# 
# This Notebook helps beginners visualize the Decision Boundaries for various classifiers.
# 
# This Notebook covers Decision Boundary Visualization of below classifiers:
# 
# **CLASSIFIERS :**
# 1. Logistic_Regression
# 2. K_Nearest_Neighbors
# 3. Support_Vector_Machines
# 4. Decision_Trees
# 5. Random_Forest
# 6. Extra_Trees
# 7. Ada_Boost
# 8. Gradient_Boost
# 
# Please do upvote if you find this helpful.
# 
# Suggestions are welcome :)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import re
import time
import math
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
#import classifiers from sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# **LOADING DATA :**
# 
# The first step is to read the data from the CSV file using pandas.
# 
# The current data_type is data frame. Difference between data frame and matrix is that data frame can store strings, numbers etc, whereas matrices can only store numbers.

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.head(2)


# **FEATURE ENGINEERING:**
# 
# Thanks to the author of https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# 
# Do check it out for more details on Feature Engineering.
# 
# The author explains detailed description of how to extract features from the dataset.

# In[ ]:


full_data = [train_data, test_data]

# Feature that tells whether a passenger had a cabin on the Titanic
train_data['Has_Cabin'] = train_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test_data['Has_Cabin'] = test_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Feature engineering steps taken from Sina
# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
# Remove all NULLS in the Fare column and create a new feature CategoricalFare
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train_data['Fare'].median())
train_data['CategoricalFare'] = pd.qcut(train_data['Fare'], 4,duplicates='drop')
# Create a New feature CategoricalAge
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
train_data['CategoricalAge'] = pd.cut(train_data['Age'], 5)
# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    # Mapping titles
    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']                               = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare']                                  = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age']                          = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4
 
# Feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']#'Parch','Fare','Embarked','IsAlone']
train_data = train_data.drop(drop_elements, axis = 1)
train_data = train_data.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
test_data  = test_data.drop(drop_elements, axis = 1)
train_data.head(2)


# In[ ]:


plt.figure(figsize=(15,7)) 
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_data.corr(),annot=True,cmap=plt.cm.winter) #draws  heatmap with input as the correlation matrix calculted by(iris.corr())
plt.show()


# In[ ]:


def plot(X_reduced,y,y_predict,title,f,axs):
    # create meshgrid
    #plot the negative points and positive points
    axs.set_title(title,fontsize=20)
    neg_val1 = X_reduced[np.where(y == 0), 0]
    neg_val2 = X_reduced[np.where(y == 0), 1]
    pos_val1 = X_reduced[np.where(y == 1), 0]
    pos_val2 = X_reduced[np.where(y == 1), 1]
    resolution = 500 # 100x100 background pixels
    X2d_xmin, X2d_xmax = np.min(X_reduced[:,0]), np.max(X_reduced[:,0])
    X2d_ymin, X2d_ymax = np.min(X_reduced[:,1]), np.max(X_reduced[:,1])
    xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))

    # approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    background_model = KNeighborsClassifier(n_neighbors=1).fit(X_reduced, y_predict) 
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((resolution, resolution))

    #plot
    axs.contourf(xx, yy, voronoiBackground)
    l1 = axs.scatter(neg_val1, neg_val2, marker='o', c='red')
    l2 = axs.scatter(pos_val1, pos_val2, marker='x', c='green')
    f.legend((l1, l2), ('Not_Survived', 'Survived'), 'upper left',fontsize=15)


# In[ ]:


train_data = train_data.as_matrix()
test_data = test_data.as_matrix()
X = train_data[:,1:]
y = train_data[:,:1]
X_test = test_data
X_train, X_val, y_train, y_val = train_test_split( X, y, test_size = 0.1)# in this our main data is split into train and test
# the attribute test_size=0.1 splits the data into 90% and 10% ratio. train=90% and test=10%
y_train = np.reshape(y_train,-1)
y_val = np.reshape(y_val,-1)
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Test data shape: ', X_val.shape)
print('Test labels shape: ', y_val.shape)


# In[ ]:


mean = np.mean(X_train, axis=0,dtype=np.int64)
std = np.std(X_train, axis=0)
X_train -= mean
X_train = X_train/std
X_val -= mean
X_val = X_val/std
X_test -= mean
X_test = X_test/std


# **DATA VISUALIZATION :**
# 
# **Use PCA to reduce the dimensions to 2 features.**

# In[ ]:


from sklearn.decomposition import TruncatedSVD
X_train_reduced = TruncatedSVD(n_components=2, random_state=0).fit_transform(X_train)
X_val_reduced = TruncatedSVD(n_components=2, random_state=0).fit_transform(X_val)


# **VISUALIZING TRAIN & VALIDATION DATA : **

# In[ ]:


#plot the negative points and positive points
f, axs = plt.subplots(1,2,figsize=(16,7))
axs[0].set_title('Training_Data',fontsize=20)
l1 = axs[0].scatter(X_train_reduced[np.where(y_train == 0), 0], X_train_reduced[np.where(y_train == 0), 1], marker='o', c='red')
l2 = axs[0].scatter(X_train_reduced[np.where(y_train == 1), 0], X_train_reduced[np.where(y_train == 1), 1], marker='x', c='green')
f.legend((l1, l2), ('Not_Survived', 'Survived'), 'upper left',fontsize=15)
axs[1].set_title('Validation_Data',fontsize=20)
l3 = axs[1].scatter(X_val_reduced[np.where(y_val == 0), 0], X_val_reduced[np.where(y_val == 0), 1], marker='o', c='red')
l4 = axs[1].scatter(X_val_reduced[np.where(y_val == 1), 0], X_val_reduced[np.where(y_val == 1), 1], marker='x', c='green')
f.legend((l3, l4), ('Not_Survived', 'Survived'), 'upper right',fontsize=15)


# In[ ]:


columns = ['Train_Accuracy','Validation_Accuracy']
index = ['Logistic_Regression','KNN','SVM','Decision_Tree','Random_Forest','Extra_Trees','Ada_Boost','Gradient_Boost']
data = np.zeros((8,2))


# In[ ]:


LR = LogisticRegression().fit(X_train,y_train)
svm = svm.SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0).fit(X_train, y_train)
knn = KNeighborsClassifier().fit(X_train,y_train)
DT = DecisionTreeClassifier().fit(X_train,y_train)
RF = RandomForestClassifier(n_estimators=500, max_depth=None,min_samples_split=2, random_state=0).fit(X_train,y_train)
ET = ExtraTreesClassifier(n_estimators=500, max_depth=None,min_samples_split=2, random_state=0).fit(X_train,y_train)
AB = AdaBoostClassifier(n_estimators=500).fit(X_train,y_train)
GB = GradientBoostingClassifier(n_estimators=500).fit(X_train,y_train)


# **DECISION_BOUNDARIES :**

# **LOGISTIC_REGRESSION :**

# In[ ]:


f, axs = plt.subplots(1,2,figsize=(16,7))
plot(X_train_reduced,y_train,LR.predict(X_train),'Logistic_Regression_Train',f,axs[0])
plot(X_val_reduced,y_val,LR.predict(X_val),'Logistic_Regression_Validation',f,axs[1])


# **K_NEAREST_NEIGHBORS :**

# In[ ]:


f, axs = plt.subplots(1,2,figsize=(16,7))
plot(X_train_reduced,y_train,knn.predict(X_train),'KNN_Train',f,axs[0])
plot(X_val_reduced,y_val,knn.predict(X_val),'KNN_Validation',f,axs[1])


# **SUPPORT_VECTOR_MACHINES : **

# In[ ]:


f, axs = plt.subplots(1,2,figsize=(16,7))
plot(X_train_reduced,y_train,svm.predict(X_train),'SVM_Train',f,axs[0])
plot(X_val_reduced,y_val,svm.predict(X_val),'SVM_Validation',f,axs[1])


# **DECISION_TREES :**

# In[ ]:


f, axs = plt.subplots(1,2,figsize=(16,7))
plot(X_train_reduced,y_train,DT.predict(X_train),'Decision_Tree_Train',f,axs[0])
plot(X_val_reduced,y_val,DT.predict(X_val),'Decision_Tree_Validation',f,axs[1])


# **RANDOM_FOREST :**

# In[ ]:


f, axs = plt.subplots(1,2,figsize=(16,7))
plot(X_train_reduced,y_train,RF.predict(X_train),'Random_Forest_Train',f,axs[0])
plot(X_val_reduced,y_val,RF.predict(X_val),'Random_Forest_Validation',f,axs[1])


# **EXTRA_TREES :**

# In[ ]:


f, axs = plt.subplots(1,2,figsize=(16,7))
plot(X_train_reduced,y_train,ET.predict(X_train),'Extra_Trees_Train',f,axs[0])
plot(X_val_reduced,y_val,ET.predict(X_val),'Extra_Trees_Validation',f,axs[1])


# **ADA_BOOSTING :**

# In[ ]:


f, axs = plt.subplots(1,2,figsize=(16,7))
plot(X_train_reduced,y_train,AB.predict(X_train),'Ada_Boost_Train',f,axs[0])
plot(X_val_reduced,y_val,AB.predict(X_val),'Ada_Boost_Validation',f,axs[1])


# **GRADIENT_BOOSTING :**

# In[ ]:


f, axs = plt.subplots(1,2,figsize=(16,7))
plot(X_train_reduced,y_train,GB.predict(X_train),'Gradient_Boost_Train',f,axs[0])
plot(X_val_reduced,y_val,GB.predict(X_val),'Gradient_Boost_Validation',f,axs[1])


# **ACCURACIES OF TRAIN AND VALIDATION SET:**

# In[ ]:


data[0,0] = LR.score(X_train,y_train)
data[0,1] = LR.score(X_val,y_val)
data[1,0] = knn.score(X_train,y_train)
data[1,1] = knn.score(X_val,y_val)
data[2,0] = svm.score(X_train,y_train)
data[2,1] = svm.score(X_val,y_val)
data[3,0] = DT.score(X_train,y_train)
data[3,1] = DT.score(X_val,y_val)
data[4,0] = RF.score(X_train,y_train)
data[4,1] = RF.score(X_val,y_val)
data[5,0] = ET.score(X_train,y_train)
data[5,1] = ET.score(X_val,y_val)
data[6,0] = AB.score(X_train,y_train)
data[6,1] = AB.score(X_val,y_val)
data[7,0] = GB.score(X_train,y_train)
data[7,1] = GB.score(X_val,y_val)


# In[ ]:


accuracy = pd.DataFrame(data, index=index, columns=columns)
print(accuracy)


# **Thank you!!!
# Have a Nice Day!!**
