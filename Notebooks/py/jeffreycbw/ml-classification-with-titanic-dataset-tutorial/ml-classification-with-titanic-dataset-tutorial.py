#!/usr/bin/env python
# coding: utf-8

# My 1st personal kernel on practising a Classification problem. My goal is to make it a weekly thing with various techniques and datasets. Your feedback and comments are much appreciated.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')
import warnings
warnings.filterwarnings("ignore")


# Load train file and test file into Dataframe

# In[ ]:


training = pd.read_csv("../input/train.csv")
testing = pd.read_csv("../input/test.csv")


# By using Dataframe.head(), we can glimpse some of the data in the Train and Test dataset

# In[ ]:


training.head()


# In[ ]:


testing.head()


# In[ ]:


training.describe()


# After looking at the dataset, we need to refurbish the dataset to remove NA value

# In[ ]:


training.isna().sum()


# In[ ]:


testing.isna().sum()


# * Droppping columns for:
#     1. [Cabin] because the missing value is high.
#     2. [Ticket] because the data is not helping on making the predictions

# In[ ]:


training.drop(labels=["Cabin", "Ticket"], axis = 1 , inplace = True)
testing.drop(labels= ["Cabin", "Ticket"], axis = 1, inplace  = True)


# * We can check again with Dataframe.isna().sum() to confirm if [Cabin] and [Ticket] are removed from the Training and Testing datasets

# In[ ]:


training.isna().sum()


# In[ ]:


testing.isna().sum()


# * Fill the remaining empty values in columns [Age] and [Embarked] with medians for both Training and Testing datasets.
# * Combine SibSp and Parch because they are related as the Family size in which
#     * [SibSP]: related to sibling and spouse
#     * [Parch]: related to Parent, Child and some children travelled with a nanny.
# 
#  and we can analyze later if they help on our predictions

# In[ ]:


copy = training.copy()
copy.dropna(inplace = True)
sns.distplot(copy["Age"])


# In[ ]:


training["Age"].fillna(training["Age"].median(), inplace = True)
testing["Age"].fillna(training["Age"].median(), inplace = True)
training["Embarked"].fillna("S", inplace = True)
testing["Fare"].fillna(testing["Fare"].median(), inplace = True)
training["Family_Size"] = training["SibSp"] + training["Parch"] + 1
testing["Family_Size"] = testing["SibSp"] + testing["Parch"] + 1


# After replacing NA with median(), the column [Age] is more normally distributed.

# In[ ]:


sns.distplot(training["Age"])


# Perhaps the individual title may help on the predictions Hence, we can add column [Title] from column [Name] for Training dataset

# In[ ]:


for name in training["Name"]:
    training["Title"] = training["Name"].str.extract("([A-Z a-z]+)\.", expand= True)

count = 0
for i in list(training["Title"]):
    if not i in [" Mr", " Mrs", " Miss", " Master"]:
        training.at[count, "Title"] = " Other"
    count += 1

title_list = list(training["Title"])
frequency_title = {}
for i in title_list:
    frequency_title.update({i :(title_list.count(i))})
frequency_title


# We then add the  column [Title] from Column [Name] for Testing dataset

# In[ ]:


for name in testing["Name"]:
    testing["Title"] = testing["Name"].str.extract("([A-Z a-z]+)\.", expand= True)
    
count = 0
for i in list(testing["Title"]):
    if not i in [" Mr", " Mrs", " Miss", " Master"]:
        testing.at[count, "Title"] = " Other"
    count += 1

title_list = list(testing["Title"])
frequency_title = {}
for i in title_list:
    frequency_title.update({i :(title_list.count(i))})
frequency_title


# We can validate again with Dataframe.isna().sum() if there is any missing values appear for both Training and Testing datasets

# In[ ]:


training.isna().sum()


# In[ ]:


testing.isna().sum() 


# * Pairplot is good to display relationship between variables.
# * Since Family_Size is from Parch and SibSp, there is a relationship that can be found from those graphs.

# In[ ]:


sns.pairplot(training)


# From the heatmap, there seems to be a  correlation between Survived and Fare

# In[ ]:


sns.heatmap(training.corr(), annot= True)
plt.show()


# * Next, we use pd.get_dummies to convert the categorical variables into dummy/indicator variables. The categorical variables that we can identify from the datasets are:
#     * Sex
#     * Embarked
#     * Title
#     * Pclass

# In[ ]:


train_enc = pd.get_dummies(training, columns = ["Sex","Embarked", "Title", "Pclass"])
test_enc = pd.get_dummies(testing, columns = ["Sex", "Embarked", "Title", "Pclass"])


# * We can load MinMaxScaler from sklearn.preprocessing to downscale the weights of numerical variables between 0 and 1 for Training and Testing datasets. The numerical values that are identified as below.
#     * Age
#     * Fare
#     * SibSp
#     * Parch
#     * Family_Size

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
train_enc[["Age", "Fare", "SibSp","Parch", "Family_Size"]] = scale.fit_transform(train_enc[["Age", "Fare", "SibSp","Parch", "Family_Size"]].as_matrix())
test_enc[["Age", "Fare", "SibSp","Parch", "Family_Size"]] = scale.fit_transform(test_enc[["Age", "Fare", "SibSp","Parch", "Family_Size"]].as_matrix())


# * Before moving to Validation and Training of the model. We remove column [PassengerId], [Name] and [Sex_ male] from both Training and Testing datasets. 
#     * The reason being is that, we are not required to know the passengers and their names. 
#     * For the Sex_ male, Sex_ female column would be sufficient enough to distinguish the only 2 categories (Male and Female).
#     
# * Furthermore, we remove column [Survived] from the X_train 
#     * because that is the one we are doing for our prediction. Hence, we would create a new dataframe Y_train for the column [Survived]

# In[ ]:


X_train = train_enc.drop(labels = ["PassengerId", "Survived", "Name", "Sex_male"], axis = 1)
Y_train = train_enc["Survived"]
X_test = test_enc.drop(labels = ["PassengerId", "Name", "Sex_male"], axis = 1)


# * We need to further split the Training dataset into Training and Validation datasets because we need a subset of our Training dataset to validate on our model results. 
#     * The Testing set cannot be used for our validation of the model when the predicted column [Survived] is not provided.
# 
# * Thus, we would split into Training and Validation datasets with train_test_split from sklearn.model_selection. 
#     * Then, we would specify on what is the size of the Validation set in which we have set as 20%
#     * The random_state is set to 0 (or any other numerical value of your choice) so we can reproduce the same results for every run
#     
# * After splitting, we need to balance the Training dataset with enough positive and negative samples. Hence, we would attempt to oversample the minority with SMOTE from imblearn.over_sampling

# In[ ]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
X_training, X_valid, y_training, y_valid = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 0)
sm = SMOTE(random_state = 2)
X_train, Y_train = sm.fit_sample(X_train, Y_train.ravel())


# We would import 
# * GridSearchCV
#     * It is used for exhaustive search over specified parameter values for an estimator. In other words, it can provide the most optimized parameter from a set of parameters provided.
# * maker_scorer
#     * This factory function wraps scoring functions for use in GridSearchCV and cross_val_score.
# * accuracy_score
#     * Accuracy classification score that we are using for the Titanic dataset
# * XGBClassifier, RandomForestClassifier, DecisionTreeClassifier 
#     * Those are the models we would use for our Training and Validation
# * Cross_val_score
#     * To estimate the expected accuracy of the model.
# * confusion_matrix
#     * Provide more presentable results by displaying TP, FP, TN,  FN    

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


# In[ ]:


model_xgb = XGBClassifier() #Identify the model for training
param_xgb = {"objective": ["reg:linear"], "n_estimators": [5, 10, 15, 20]} #Listing the parameters to find the best parameters 
grid_xgb = GridSearchCV(model_xgb, param_xgb, scoring = make_scorer(accuracy_score)) #Setup the function to find the best parameters
grid_xgb.fit(X_training, y_training) #To find the best parameters for the model
model_xgb = grid_xgb.best_estimator_ #Select the best parameters for the model
model_xgb.fit(X_training, y_training) #Fitting the best estimator parameters for Training
pred_xgb = model_xgb.predict(X_valid) #Predict the trained model with validation dataset
acc_xgb = accuracy_score(y_valid, pred_xgb) #Find the accuracy of the model
scores_xgb = cross_val_score(model_xgb, X_training, y_training, cv = 5) #Check on the cross validation score of the dataset with 5 splits
print("Best parameters: {} \nCross Validation Score: {} (+/- {} \nBest prediction accuracy: {})".format(model_xgb, scores_xgb.mean(), scores_xgb.std(), acc_xgb))


# In[ ]:


cm_xgb = confusion_matrix(y_valid, pred_xgb)
cm_df = pd.DataFrame(cm_xgb, index = ["Not Survived", "Survived"], columns = ["Not Survived", "Survived"])
sns.heatmap(cm_df, annot= True)
plt.title("XGB Classifier \nAccuracy: {0:.3f}".format(acc_xgb))
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


# In[ ]:


sns.barplot(model_xgb.feature_importances_, X_training.columns)
plt.title("Feature Importance for XGB Classifier")
plt.show()


# In[ ]:


model_rfc = RandomForestClassifier() #Identify the model for training
param_rfc = {"n_estimators" : [4, 5, 6, 7, 8, 9, 10, 15], "criterion" : ["gini", "entropy"], "max_features": ["auto", "sqrt", "log2"],
            "max_depth": [2, 3, 5, 10], "min_samples_split": [2, 3, 5, 10]} #Listing the parameters to find the best parameters 
grid_rfc = GridSearchCV(model_rfc, param_rfc, scoring = make_scorer(accuracy_score)) #Setup the function to find the best parameters
grid_rfc.fit(X_training, y_training) #To find the best parameters for the model 
model_rfc = grid_rfc.best_estimator_ #Select the best parameters for the model
model_rfc.fit(X_training, y_training) #Fitting the best estimator parameters for Training
pred_rfc = model_rfc.predict(X_valid) #Predict the trained model with validation dataset
acc_rfc = accuracy_score(y_valid, pred_rfc) #Find the accuracy of the model
scores_rfc = cross_val_score(model_rfc, X_training, y_training, cv = 5) #Check on the cross validation score of the dataset with 5 splits
print("Best parameters: {} \nCross Validation Score: {} (+/- {}) \nBest prediction accuracy: {}".format(model_rfc, scores_rfc.mean(), scores_rfc.std(), acc_rfc))


# In[ ]:


cm_rfc = confusion_matrix(y_valid, pred_rfc)
cm_df = pd.DataFrame(cm_rfc, index = ["Not Survived", "Survived"], columns = ["Not Survived", "Survived"])
sns.heatmap(cm_df, annot = True)
plt.title("Random Forest Classifier \nAccuracy: {0:.3f}".format(acc_rfc))
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


# In[ ]:


sns.barplot(model_rfc.feature_importances_, X_training.columns)
plt.title("Feature Importance for Random Forest Classifier")
plt.show()


# In[ ]:


model_dtc = DecisionTreeClassifier() #Identify the model for training
param_dtc = {"criterion": ["gini", "entropy"], "splitter": ["best","random"], "max_features": ["auto", "sqrt", "log2"]} #Listing the parameters to find the best parameters 
grid_dtc = GridSearchCV(model_dtc, param_dtc, scoring = make_scorer(accuracy_score)) #Setup the function to find the best parameters
grid_dtc.fit(X_training, y_training) #To find the best parameters for the model 
model_dtc = grid_dtc.best_estimator_ #Select the best parameters for the model
model_dtc.fit(X_training, y_training) #Fitting the best estimator parameters for Training
pred_dtc = model_dtc.predict(X_valid) #Predict the trained model with validation dataset
acc_dtc = accuracy_score(y_valid, pred_dtc) #Find the accuracy of the model
scores_dtc = cross_val_score(model_dtc, X_training, y_training, cv = 5) #Check on the cross validation score of the dataset with 5 splits
print("Best parameters: {} \nCross Validation Score: {} (+/- {}) \nBest prediction accuracy: {} ".format(model_dtc, scores_dtc.mean(), scores_dtc.std(), acc_dtc ))


# In[ ]:


cm_dtc = confusion_matrix(y_valid, pred_dtc)
cm_df = pd.DataFrame(cm_dtc, index = ["Not Survived", "Survived"], columns = ["Not Survived", "Survived"])
sns.heatmap(cm_df, annot = True)
plt.title("Decision Tree Classifier \nAccuracy: {0:.3f}".format(acc_dtc))
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


# In[ ]:


sns.barplot(model_dtc.feature_importances_, X_training.columns)
plt.title("Feature Importance for Decision Tree Classifier")
plt.show()


# We are going to submit the best model which is random forest classifier since the prediction is the highest among the 3.

# In[ ]:


model_rfc.fit(X_train, Y_train)


# In[ ]:


submission_predictions = model_rfc.predict(X_test)


# In[ ]:


submission = pd.DataFrame({
    "PassengerId": testing["PassengerId"],
    "Survived": submission_predictions
})

submission.to_csv("titanic.csv", index = False)
print(submission.shape)

