#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import the necessary libraries
import numpy as np
import pandas as pd
import os
import time
import warnings
import os
from six.moves import urllib
import matplotlib
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# In[ ]:


#Add All the Models Libraries

# Scalers
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

# Models
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn.svm import SVC # Support Vector Classifier
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
from scipy.stats import reciprocal, uniform

from sklearn.ensemble import AdaBoostClassifier


# Cross-validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn.model_selection import cross_validate

# GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#Common data processors
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from scipy import sparse

#Accuracy Score
from sklearn.metrics import accuracy_score


# In[ ]:


# to make this notebook's output stable across runs
np.random.seed(123)

# To plot pretty figures
get_ipython().magic(u'matplotlib inline')
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


# In[ ]:


#merge the data for feature engineering and later split it, just before applying Data Pipeline
TrainFile = pd.read_csv("../input/train.csv") #read the data from the csv file.
TestFile = pd.read_csv("../input/test.csv")
passenger_id_test = TestFile["PassengerId"].copy()
DataFile = TrainFile.append(TestFile)


# In[ ]:


TrainFile.shape


# In[ ]:


TestFile.shape


# In[ ]:


DataFile.describe()


# In[ ]:


DataFile.info()


# In[ ]:


# First Split the names to gt Mr. or Miss or Mrs.

FirstName = DataFile["Name"].str.split("[,.]")


# In[ ]:


# now strip the white spaces from the Salutation
titles = [str.strip(name[1]) for name in FirstName.values]


# In[ ]:


DataFile["Title"] = titles


# In[ ]:


# Now first we replace the extra titles to Mr and Mrs

mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'the Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}

DataFile.replace({'Title': mapping}, inplace=True)


# In[ ]:


# get the imputed value for FARE
DataFile['Fare'].fillna(DataFile['Fare'].median(), inplace=True)

#based on the median taken to check fare the imputed value of Embarkment to be "S"
DataFile['Embarked'].fillna("S", inplace=True)

#impute the age based on Titles 
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
for title in titles:
    imputed_age = DataFile.groupby('Title')['Age'].median()[titles.index(title)]
    DataFile.loc[(DataFile['Age'].isnull()) & (DataFile['Title'] == title), 'Age'] = imputed_age


# In[ ]:


# Merge SibSp and Parch into one
DataFile["Family Size"] = DataFile["SibSp"] + DataFile["Parch"] + 1

#Make Family Bins
DataFile["Family Size"] = pd.cut(DataFile["Family Size"], bins=[0,1,4,20], labels=[1,2,3])


# In[ ]:


#Making Fare bins

DataFile['FareBin'] = pd.qcut(DataFile['Fare'], 5)

label = LabelEncoder()
DataFile['FareBin'] = label.fit_transform(DataFile['FareBin'])

#Making Age Bins
DataFile['AgeBin'] = pd.qcut(DataFile['Age'], 4)

label = LabelEncoder()
DataFile['AgeBin'] = label.fit_transform(DataFile['AgeBin'])


# In[ ]:


#create a dummy for male and female
DataFile['Sex'].replace(['male','female'],[0,1],inplace=True)


# In[ ]:


##ADDED NEW CODE TO THE EXISTING PROGRAM
DataFile['Last_Name'] = DataFile['Name'].apply(lambda x: str.split(x, ",")[0])

DEFAULT_SURVIVAL_VALUE = 0.5
DataFile['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in DataFile[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                DataFile.loc[DataFile['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                DataFile.loc[DataFile['PassengerId'] == passID, 'Family_Survival'] = 0

print("Number of passengers with family survival information:", 
      DataFile.loc[DataFile['Family_Survival']!=0.5].shape[0])


# In[ ]:


for _, grp_df in DataFile.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    DataFile.loc[DataFile['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    DataFile.loc[DataFile['PassengerId'] == passID, 'Family_Survival'] = 0
                        
print("Number of passenger with family/group survival information: " 
      +str(DataFile[DataFile['Family_Survival']!=0.5].shape[0]))


# In[ ]:


#drop SibSp and Parch
DataFile = DataFile.drop(['SibSp','Parch','Age','Fare',
                          'Last_Name','Name','PassengerId',
                          "Ticket","Cabin","Embarked"],axis=1)


# In[ ]:


#Now split Back The data to training and test set - before applying the pipeline

train_set, test_set = train_test_split(DataFile, test_size=0.3193,shuffle=False)


# In[ ]:


train_set.shape # This exactly matches the original training set


# In[ ]:


test_set.shape # This exactly matches the original test set


# In[ ]:


#Check for the missing values to check if any random extraction happened? Validate that shuffle was false

obs = train_set.isnull().sum().sort_values(ascending = False)
percent = round(train_set.isnull().sum().sort_values(ascending = False)/len(train_set)*100, 2)
pd.concat([obs, percent], axis = 1,keys= ['Number of Observations', 'Percent'])


# In[ ]:


#Check for the missing values to check if any random extraction happened? Validate that shuffle was false

obs = test_set.isnull().sum().sort_values(ascending = False)
percent = round(test_set.isnull().sum().sort_values(ascending = False)/len(test_set)*100, 2)
pd.concat([obs, percent], axis = 1,keys= ['Number of Observations', 'Percent'])


# In[ ]:


# Now define x and y.

#the Y Variable
train_set_y = train_set["Survived"].copy()
test_set_y = test_set["Survived"].copy()

#the X variables
train_set_X = train_set.drop("Survived", axis=1)
test_set_X = test_set.drop("Survived", axis=1)


# In[ ]:


# The CategoricalEncoder class will allow us to convert categorical attributes to one-hot vectors.

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out


# In[ ]:


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


# In[ ]:


cat_pipeline = Pipeline([
        ("selector", DataFrameSelector(['Title'])),
        ("cat_encoder", CategoricalEncoder(encoding='onehot-dense')),
    ])

no_pipeline = Pipeline([
        ("selector", DataFrameSelector(["Sex","Pclass","FareBin", "AgeBin","Family Size","Family_Survival"]))
    ])


# In[ ]:


full_pipeline = FeatureUnion(transformer_list=[
    ("cat_pipeline", cat_pipeline),
    ("no_pipeline", no_pipeline),
    ])

final_train_X = full_pipeline.fit_transform(train_set_X)
final_test_X = full_pipeline.transform(test_set_X)


# #### Now We Build the Models

# #### KNN Classifier

# In[ ]:


#Introduce KNN Classifier 

KNeighbours = KNeighborsClassifier()
leaf_size = list(range(1,25,5))
n_neighbors = list(range(4,30,2))

param_grid_KNeighbours = {'n_neighbors' : n_neighbors,
'algorithm' : ['auto'],
'weights' : ['uniform', 'distance'],
'leaf_size':leaf_size}

grid_search_KNeighbours = GridSearchCV(KNeighbours, param_grid_KNeighbours, cv = 4, scoring='roc_auc', 
                               refit = True, n_jobs = -1, verbose = 2)

grid_search_KNeighbours.fit(final_train_X, train_set_y)


# In[ ]:


neighbor_grid = grid_search_KNeighbours.best_estimator_

y_pred_neighbor_grid = neighbor_grid.predict(final_train_X)
accuracy_score(train_set_y, y_pred_neighbor_grid)


# In[ ]:


# now get the predictions
y_pred_neigh_rand = neighbor_grid.predict(final_test_X)

# Prepare the predictions file
result_test1 = pd.DataFrame()
passenger_id_test = TestFile["PassengerId"].copy()
result_test1["PassengerId"] = passenger_id_test
result_test1["Survived"] = y_pred_neigh_rand

# Export the predictions file
#result_test1.to_csv("Titanic_prediction_ashish_KNN.csv")


# #### Another KNN Approach

# In[ ]:


KNeighbours2 = KNeighborsClassifier()
leaf_size2 = list(range(8,50,1))
n_neighbors2 = list(range(5,20,1))

param_grid_KNeighbours = {'n_neighbors' : n_neighbors2,
'algorithm' : ['auto'],
'weights' : ['uniform', 'distance'],
'leaf_size':leaf_size2}

grid_search_KNeighbours2 = GridSearchCV(KNeighbours2, param_grid_KNeighbours, cv = 4, scoring='roc_auc', 
                               refit = True, n_jobs = -1, verbose = 2)

grid_search_KNeighbours2.fit(final_train_X, train_set_y)


# In[ ]:


neighbor_grid2 = grid_search_KNeighbours2.best_estimator_

y_pred_neighbor_grid2 = neighbor_grid2.predict(final_train_X)
accuracy_score(train_set_y, y_pred_neighbor_grid2)


# ##### Random Forest Classifier

# In[ ]:


forest_class = RandomForestClassifier(random_state = 42)

n_estimators = [50, 100, 400, 700, 1000]
max_features = [5, 7, 10]
max_depth = [10, 20] 
oob_score = [True, False]
min_samples_split = [2, 4, 10, 12, 16]
min_samples_leaf = [1, 5, 10] 
max_leaf_nodes = [2, 10, 20]


param_grid_forest = {'n_estimators' : n_estimators, 'max_features' : max_features,
                     'max_depth' : max_depth, 'min_samples_split' : min_samples_split,
                    'oob_score' : oob_score, 'min_samples_leaf': min_samples_leaf, 
                     'max_leaf_nodes' : max_leaf_nodes}


rand_search_forest = RandomizedSearchCV(forest_class, param_grid_forest, cv = 4, scoring='roc_auc', refit = True,
                                 n_jobs = -1, verbose=2)

rand_search_forest.fit(final_train_X, train_set_y)


# In[ ]:


random_estimator = rand_search_forest.best_estimator_

y_pred_random_estimator = random_estimator.predict(final_train_X)
accuracy_score(train_set_y, y_pred_random_estimator)


# In[ ]:


# now get the predictions
y_pred_forest_rand = random_estimator.predict(final_test_X)

# Prepare the predictions file
result_test2 = pd.DataFrame()
passenger_id_test = TestFile["PassengerId"].copy()
result_test2["PassengerId"] = passenger_id_test
result_test2["Survived"] = y_pred_forest_rand

# Export the predictions file
#result_test2.to_csv("Titanic_prediction_ashish_RF.csv")


# ###### Ada Boost Classifier

# In[ ]:


ada_boost = AdaBoostClassifier(random_state = 42)

n_estimators = [50, 100, 400, 700, 1000]
learning_rate = [0.001, 0.01, 0.05, 0.09]
algorithm = ['SAMME', 'SAMME.R']

param_grid_ada = {'n_estimators' : n_estimators, 'learning_rate' : learning_rate, 'algorithm' : algorithm}

rand_search_ada = RandomizedSearchCV(ada_boost, param_grid_ada, cv = 4, scoring='roc_auc', refit = True, n_jobs = -1, verbose = 2)

rand_search_ada.fit(final_train_X, train_set_y)


# In[ ]:


ada_estimator = rand_search_ada.best_estimator_

y_pred_ada_estimator = ada_estimator.predict(final_train_X)
accuracy_score(train_set_y, y_pred_ada_estimator)


# In[ ]:


# now get the predictions
y_pred_ada_rand = ada_estimator.predict(final_test_X)

# Prepare the predictions file
result_test3 = pd.DataFrame()
passenger_id_test = TestFile["PassengerId"].copy()
result_test3["PassengerId"] = passenger_id_test
result_test3["Survived"] = y_pred_ada_rand

# Export the predictions file
#result_test3.to_csv("Titanic_prediction_ashish_ada.csv")


# ###### Extra Trees Classifier

# In[ ]:


extra_classifier = ExtraTreesClassifier(random_state = 42)

n_estimators = [50, 100, 400, 700, 1000]
max_features = [5, 7, 10]
max_depth = [10, 20]
min_samples_split = [2, 4, 10, 12, 16]
min_samples_leaf = [1, 5, 10]  # Mhm, this one leads to accuracy of test and train sets being the same.

param_grid_extra_trees = {'n_estimators' : n_estimators, 'max_features' : max_features,
                         'max_depth' : max_depth, 'min_samples_split' : min_samples_split,
                         'min_samples_leaf' : min_samples_leaf}


rand_search_extra_trees = RandomizedSearchCV(extra_classifier, param_grid_extra_trees, cv = 4, scoring='roc_auc', 
                               refit = True, n_jobs = -1, verbose = 2)

rand_search_extra_trees.fit(final_train_X, train_set_y)


# In[ ]:


extra_estimator = rand_search_extra_trees.best_estimator_

y_pred_extra_estimator = extra_estimator.predict(final_train_X)
accuracy_score(train_set_y, y_pred_extra_estimator)


# #### Support Vector Classifier

# In[ ]:


SVC_Classifier = SVC(random_state = 42)

param_distributions = {"gamma": reciprocal(0.0001, 0.001), "C": uniform(100000, 1000000)}

rand_search_svc = RandomizedSearchCV(SVC_Classifier, param_distributions, n_iter=10, verbose=2, n_jobs = -1)

rand_search_svc.fit(final_train_X, train_set_y)


# In[ ]:


svc_estimator = rand_search_svc.best_estimator_

y_pred_svc_estimator = svc_estimator.predict(final_train_X)
accuracy_score(train_set_y, y_pred_svc_estimator)


# In[ ]:


# now get the predictions
y_pred_svc_rand = svc_estimator.predict(final_test_X)

# Prepare the predictions file
result_test4 = pd.DataFrame()
passenger_id_test = TestFile["PassengerId"].copy()
result_test4["PassengerId"] = passenger_id_test
result_test4["Survived"] = y_pred_svc_rand 

# Export the predictions file
#result_test4.to_csv("Titanic_prediction_ashish_svc.csv")


# #### Gradient Boosting Classifier

# In[ ]:


GB_Classifier = GradientBoostingClassifier(random_state = 42)

n_estimators = [50, 100, 400, 700, 1000]
learning_rate = [0.1, 0.5]
max_depth = [10, 20]
min_samples_split = [2, 4, 10, 12, 16]
min_samples_leaf = [1, 5, 10]
max_leaf_nodes = [2, 10, 20]
                            
param_grid_grad_boost_class = {'n_estimators' : n_estimators, 'learning_rate' : learning_rate,
                              'max_depth' : max_depth, 'min_samples_split' : min_samples_split,
                              'min_samples_leaf' : min_samples_leaf, 'max_leaf_nodes' : max_leaf_nodes}

rand_search_grad_boost_class = RandomizedSearchCV(GB_Classifier, param_grid_grad_boost_class, cv = 4, scoring='roc_auc', 
                               refit = True, n_jobs = -1, verbose = 2)

rand_search_grad_boost_class.fit(final_train_X, train_set_y)


# In[ ]:


gb_estimator = rand_search_grad_boost_class.best_estimator_

y_pred_gb_estimator = gb_estimator.predict(final_train_X)
accuracy_score(train_set_y, y_pred_gb_estimator)


# In[ ]:


# now get the predictions
y_pred_gb_rand = gb_estimator.predict(final_test_X)

# Prepare the predictions file
result_test5 = pd.DataFrame()
passenger_id_test = TestFile["PassengerId"].copy()
result_test5["PassengerId"] = passenger_id_test
result_test5["Survived"] = y_pred_gb_rand 

# Export the predictions file
#result_test5.to_csv("Titanic_prediction_ashish_gb.csv")


# #### Logistic Classifier

# In[ ]:


log_reg = LogisticRegression(random_state = 42)

C = np.array(list(range(1, 100)))/10
                            
param_grid_log_reg = {'C' : C}

rand_search_log_reg = RandomizedSearchCV(log_reg, param_grid_log_reg, cv = 4, scoring='roc_auc', 
                               refit = True, n_jobs = -1, verbose = 2)

rand_search_log_reg.fit(final_train_X, train_set_y)


# In[ ]:


log_estimator = rand_search_log_reg.best_estimator_

y_pred_log_estimator = log_estimator.predict(final_train_X)
accuracy_score(train_set_y, y_pred_log_estimator)


# In[ ]:


# now get the predictions
y_pred_log_rand = log_estimator.predict(final_test_X)

# Prepare the predictions file
result_test6 = pd.DataFrame()
passenger_id_test = TestFile["PassengerId"].copy()
result_test6["PassengerId"] = passenger_id_test
result_test6["Survived"] = y_pred_log_rand 

# Export the predictions file
#result_test6.to_csv("Titanic_prediction_ashish_log.csv")


# ##### MLP Classifier - Nueral Networks

# In[ ]:


mlp_clf = MLPClassifier(random_state = 42)

alpha = [.0001,.001,.01,1]
learning_rate_init= [.0001,.001,.01,1]
max_iter = [50,70,100,200]
tol = [.0001,.001,.01,1]

param_grid_mlp_clf = {'alpha':alpha, 'learning_rate_init':learning_rate_init, 'max_iter':max_iter,'tol':tol}

rand_search_mlp_clf = RandomizedSearchCV(mlp_clf, param_grid_mlp_clf, cv = 4, scoring='roc_auc', 
                               refit = True, n_jobs = -1, verbose = 2)

rand_search_mlp_clf.fit(final_train_X, train_set_y)


# In[ ]:


mlp_estimator = rand_search_mlp_clf.best_estimator_

y_pred_mlp_estimator = mlp_estimator.predict(final_train_X)
accuracy_score(train_set_y, y_pred_mlp_estimator)


# In[ ]:


# now get the predictions
y_pred_mlp_rand = mlp_estimator.predict(final_test_X)

# Prepare the predictions file
result_test7 = pd.DataFrame()
passenger_id_test = TestFile["PassengerId"].copy()
result_test7["PassengerId"] = passenger_id_test
result_test7["Survived"] = y_pred_mlp_rand 

# Export the predictions file
#result_test7.to_csv("Titanic_prediction_ashish_mlp.csv")


# #### Another Ensemble Technique : Bagging Model - Not the best model on the test data

# In[ ]:


Bag_Classifier = BaggingClassifier(DecisionTreeClassifier(random_state=42))

n_estimators = [50,70,100,200,500]
max_samples = [10,50,100]

param_grid_bag_clf = {'n_estimators':n_estimators, 'max_samples':max_samples}

rand_search_bag_clf = RandomizedSearchCV(Bag_Classifier, param_grid_bag_clf, cv = 4, scoring='roc_auc', 
                               refit = True, n_jobs = -1, verbose = 2)

rand_search_bag_clf.fit(final_train_X, train_set_y)


# In[ ]:


#Predict the y_pred to get accuracy score.
bag_estimator = rand_search_bag_clf.best_estimator_
y_pred_bag = bag_estimator.predict(final_train_X)
accuracy_score(train_set_y, y_pred_bag)


# ##### Voting Classifier - Ensemble the models.

# In[ ]:


voting_clf = VotingClassifier(
    estimators=[('lr', log_estimator),('rf',random_estimator), ('ada',ada_estimator),('gb', gb_estimator), ('knn', neighbor_grid),
                ('svc', svc_estimator), ('mlp', mlp_estimator), ('ext', extra_estimator),('bag',bag_estimator)],
    voting='hard')
voting_clf.fit(final_train_X, train_set_y)


# In[ ]:


#Ensemble out the best models to get better accuracy score.

for clf in (log_estimator, random_estimator, ada_estimator, gb_estimator, neighbor_grid, svc_estimator, mlp_estimator, extra_estimator, bag_estimator,voting_clf):
    clf.fit(final_train_X, train_set_y)
    y_pred = clf.predict(final_train_X)
    print(clf.__class__.__name__, accuracy_score(train_set_y, y_pred))


# SVC looks like a clear winner with 87% accuracy on training set. But Kaggle predicts Voting Classifier as the best classification algorithm on Test set.

# In[ ]:


#predict using voting
y_pred_voting = voting_clf.predict(final_test_X)


# In[ ]:


#Create the datafile for voting classifier 2
result_test8 = TestFile.copy()
result_test8["Survived"] = y_pred_voting.astype(int)


# In[ ]:


#Import Voting Classifier results
result_test8.to_csv('Titanic_prediction_ashish.csv', columns=['PassengerId', 'Survived'], index=False)

