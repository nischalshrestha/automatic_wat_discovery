#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction

# The main task in this project is to predict if a particular person would have survived in the titanic crush. We will use the given training data to develop a high performance predictive model.In this project, one can learn retrieving csv file, how to drop and add features to the dataset, identifying the numerical and categorical variables, identifying missing data, replacing missing data, transformation (mapping) of categorical variables into equivalent numerical values, transforming variable ranges into discrete bins, correlation between all features and survival, correlation between each features (Heat map), feature engineering, 10 predictive models, model evaluation and selection, parameter tuning for the selected model, KFold cross-validation for the selected model, variable importance for the proposed model, predicting the target value (Survival of the test data), and submitting the final results.

# ## 1.1 Goal

# The goal of this project:
# 
#  1. To study the correlation between the survival rate and the features
#  2. To identify the most correlated and least correlated features
#  3. To propose a predictive model
#  4. To practice machine learning algorithms

# Some notes:
# 
# - On the outset some of the features may not have direct correlation to survival such as:  
#     - name of a person, passenger ID, ticket 
# - Some of the features may not have full/major set of data such as Cabin with multiple null 
# 
# As Feature Engineering:
# 
# - We will consider the correlation between survival and the size of family. Hence, we will create a new FamilySize feature
# - We will consider the correlation between title of the passenger and survival. We will also create a new Title feature
# 
# Transforming of categorical variables into their corresponding numeric values:
# - Dividing the age into range of ages.
# - Similarly dividing the fare into range (number of bins)  

# # 1.2 ASSUMPTIONS

# 1. Children are more likely to survive more
# 2. Female are more likely to survive 
# 3. First class passengers are more likely to survive 

# ## 1.3 Import Libraries

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "whitegrid", color_codes = True)
np.random.seed(sum(map(ord, "palettes")))

from sklearn.metrics import roc_auc_score

#Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold


# ## 1.4 Data Retriving and Exploration

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.describe()


# For the train data, the numerical features are PassengerId, pclass, Age, sibSp, Parch, and Fare.

# In[ ]:


test.describe()


# The test data has same numerical features: PassengerId, pclass, Age, sibSp, Parch, and Fare.

# ####  df.info()  help us to see how many instances does the datasethave and the overall missing data

# In[ ]:


train.info()
print("++++++++++++++++++++++++++++++++++++++")
print()
test.info()


# #### The train data has 891 instances (rows) and Age, Cabin, Embarked  have missing data. The test data has 418 instances (rows) and Age, Fare, and Cabin have missing data

# In[ ]:


train.head()


# In[ ]:


test.head()


# ###  Information about the categorical variables

# In[ ]:


train.describe(include = ['O'])


# In[ ]:


test.describe(include = ['O'])


# #### Get familiarize with the features information of the dataset

# - Categorical or discrete variables (features):
#     - Sex (male or female) the most frequent being male
#     - Embarked: (C, Q or S) the most frequent being S
#     - Cabin (this feature has several duplicates 147 unique and missing data (this could be a candidate to remove from the features)
#     - Ticket with unique 681

# # 2. Preparing Data for Model Prediction and Data Analysis

# - Filling missing Data 
# - Feature engineering
# - Feature vs Survival correlation
# - Transforming categorical variables into equivalent numerical values

# First let us combine the train and test data for preprocessing. But this combination is not used for scaling or identifying the outliers (data leakage). We will start by removing the data which are not important for data analysis and model prediction. The cabin has more missing data than available data, the PassengerId and Name wouldn't have relationship with survival. Moreover, class of the passengers is relevant to survival but I am assuming that the ticket will not have an effect. Hence, I am not considering the Ticket for further analysis.

# As we stated above, there is no importance of the Name and PassengerId for the data analysis. however, we need Name to generate the Title feature in the feature engineering. So, we will not drop the Name feature for now. We also need the PassengerId for submitting the final result. 

# In[ ]:


test_PassengerId = test["PassengerId"]  # save the id for submiting the final results

train.drop(['PassengerId', "Ticket", 'Cabin'], axis = 1, inplace = True)
test.drop(['PassengerId', "Ticket", 'Cabin'], axis=1, inplace = True)
train_test_data = [train, test] 


# ## 2. 1. Sex

# - No missing data
# - Categorical variable
# - Transform the Sex categorical variable into equivalent discrete numerical value (Sex: male being 1 and female = 0). 

# In[ ]:


for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map({'male': 1, 'female': 0}).astype(int)


# In[ ]:


train_test_data[0].head() # train data


# In[ ]:


train_test_data[1].head()  # test data


# - Sex feature correlation with survival by pivoting on Sex feature

# In[ ]:


train[['Sex', 'Survived']].groupby(['Sex'], 
                                        as_index = False).mean().sort_values(by = 'Survived', ascending = True)


# One of our assumption was the female passengers are more likely to survive than male and the correlation coefficient for the female shown in the table as 0.742. Hence, our assumption is supported by the data analysis.
# Females are more likely to survive 

# ## 2. 2. Pclass

# - No missing data
# - Categorical variable
# - Pclass feature correlation with survival by pivoting on Pclass feature
# 

# In[ ]:


train[['Pclass', 'Survived']].groupby(['Pclass'], 
                                        as_index = False).mean().sort_values(by = 'Survived', ascending = True)


# One of our assumption was the first class passengers are more likely to survive and the correlation coefficient for the first class is shown in the table as 0.63. Hence, our assumption is supported by the data analysis.

# ## 2. 3. Age

# From the .describe() and .info() we can see that there are missing data. If we look the "Age" column there are 891-714 = 177 missing age data. One way to replace these missing data is to fill them using the average value, second approach is sampling from a normal distribution using mean and standard deviation of the available data in the training and test data respectively. Age_mean = Age.mean(), Age.std(), Age_add = rnd.uniform(age_mean - age_std, age_mean + age_std), third method that we use here is to use the median based on Sex and Pclass.(From Kaggle computation project)

# In[ ]:


age_fill = np.zeros((2,3)) # 2 for sex and 3 for Pclass
print(age_fill)


# In[ ]:


age_fill = np.zeros((2,3)) 
for dataset in train_test_data:
    for s in range(0, 2):
        for p in range(0, 3):
            age_fill_df = dataset[(dataset['Sex'] == s) &                               (dataset['Pclass'] == p + 1)]['Age'].dropna()
            age_to_fill = age_fill_df.median()

            # Convert random age float to nearest .5 age
            age_fill[s,p] = int( age_to_fill/0.5 + 0.5 ) * 0.5
            
    for s in range(0, 2):
        for p in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == s) & (dataset.Pclass == p + 1),                    'Age'] = age_fill[s,p]

    dataset['Age'] = dataset['Age'].astype(int)

train.head()


# In[ ]:


test.head()


# Divide age into groups of bins:  min is 0 and max 80 so let us divide it into 8 and determine the correlation with Survival. I use 10 years of age gap

# In[ ]:


min(train['Age']), max(train['Age'])


# In[ ]:


train['AgeBins'] = pd.cut(train['Age'], 8)


# - Age feature correlation with survival by pivoting on Age feature

# In[ ]:


train[['AgeBins', 'Survived']].groupby(['AgeBins'], 
                                       as_index = False).mean().sort_values(by = 'Survived', ascending = True)


# One of our assumption was children are more likely to survive and the correlation coefficient for under age 10 is 0.594. Hence, the assumption is validated by the data analysis. 

# ###### Transforming the Age categorical feature into ordinal numerical values based in the AgeBins

# In[ ]:


for dataset in train_test_data:    
    dataset.loc[dataset['Age'] <= 10, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 20), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 30), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 60) & (dataset['Age'] <= 70), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 70, 'Age'] = 7


# In[ ]:


fig = sns.barplot(x="Sex", y="Survived", hue="Pclass", data=train)
fig.get_axes().set_xticklabels(["Female", "Male"])
fig.get_axes().legend(["First Class", "Second Class", "Third Class"], 
                    loc='upper right');


# - From the baplot we can see that female of all class survived more than the male.  This can be seen also using the point plot below. 

# In[ ]:


fig = sns.pointplot(x="Sex", y="Survived", hue="Pclass", data=train);

fig.get_axes().set_xlabel("Sex")
fig.get_axes().set_xticklabels(["Female", "Male"])
fig.get_axes().set_ylabel("Mean(Survived")
fig.get_axes().legend(["First Class", "Second Class", "Third Class"], 
                    loc='upper left')


# In[ ]:


sns.countplot(x="AgeBins", data = train, palette = "GnBu_d");


# In[ ]:


sns.countplot( x ="AgeBins", hue="Pclass", data = train, palette="PuBuGn_d");


# In[ ]:


train.head()


# Because we have transformed the age values into 8 categorical values, we don't need the AgeBins feature that we have created above. so we can drop it

# In[ ]:


train = train.drop(['AgeBins'], axis = 1)
train_test_data = [train, test]
train.head()


# ##  2. 4. Family Size (SibSp + Parch)

# - No missing data
# - Discrete variable
# - If we assume the survival is dependent on the family size and to analyze this assumption, we will combine SibSp (# of siblings / Spouses aboard ) and Parch (# of parents  / children aboard ) features together.

# In[ ]:


for dataset in train_test_data:
    dataset["FamilySize"] = dataset['SibSp'] + dataset['Parch']
train, test = train_test_data[0], train_test_data[1]
train.head()


# - FamilySize feature correlation with survival by pivoting on FamilySize feature

# In[ ]:


train[['FamilySize', 'Survived']].groupby(['FamilySize'], 
                                        as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# In[ ]:


sns.countplot(x="FamilySize", data = train, palette = "GnBu_d");


# -We are considering the FamilySize feature. So we don't need the SibSp, and Parch. We will drop them next in favor of FamilySize

# In[ ]:


train = train.drop(['Parch', 'SibSp'], axis = 1)
test = test.drop(['Parch', 'SibSp'], axis = 1)
train_test_data = [train, test]
train.head()


# In[ ]:


test.head()


# ##  2. 5. Embarked

# - 2 missing data
# - Categorical variable
# - The embarking feature takes S, Q and C categorical values for port embarkation. 
# - The missing values are filled using the most frequent value

# In[ ]:


Embarking_freq = train.Embarked.dropna().mode()[0]
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna(Embarking_freq)
train, test = train_test_data[0], train_test_data[1]
train.head()   


# - Embarked feature correlation with survival by pivoting on embarked feature

# In[ ]:


train[['Embarked', 'Survived']].groupby(['Embarked'], 
                                       as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# #### Transform the Embarked categorical values into discrete numeric values   (S = 0, C = 1, and Q = 2 )

# In[ ]:


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
train.head()


# ##  2. 6. Fare

# - The train data has no missing data
# - The test data has one missing value and we will replace it with the most frequent
# - This is a continuous feature 
# - We will lamp the fare feature into bins to develop the predictive model
# - We use qcut method from pandas to divide the fare into ranges

# In[ ]:


Fare_freq = test.Fare.dropna().mode()[0]
for dataset in train_test_data:
    dataset['Fare'] = dataset['Fare'].fillna(Fare_freq)


# - Fare feature correlation with Survival by pivoting on Fare feature

# In[ ]:


train['FareBins'] = pd.qcut(train['Fare'], 5)
train[['FareBins', 'Survived']].groupby(['FareBins'], 
                                        as_index = False).mean().sort_values(by = 'Survived', ascending = True)


# - Transforming the Fare feature into ordinal values based in the FareBins

# In[ ]:


for dataset in train_test_data:    
    dataset.loc[dataset['Fare']  <=7.854, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.84)   & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5)   & (dataset['Fare'] <= 21.679), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 39.688) & (dataset['Fare'] <= 5512.329), 'Fare'] = 4


# In[ ]:


train, test = train_test_data[0], train_test_data[1]
train = train.drop(['FareBins'], axis = 1)
train.head(6)


# In[ ]:


test.head(6)


# ##  2. 7. Title

# ##### For our data analysis and developing the predictive model, the full name doesn't have importance but I assume the title does.  So, let us extract the title of each person and add a new  feature  "Title"

# In[ ]:


def extract_title(df):
    # the Name feature includes last name, title, and first name. After splitting 
    # the title is in the second column or at index 1
    df["Title"] = df.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip()) 
    return df
train = extract_title(train)
test = extract_title(test)


# In[ ]:


fig = sns.countplot(x = 'Title', data = train, palette = "GnBu_d")
fig = plt.setp(fig.get_xticklabels(), rotation = 45)


# ![](http://)- Looking at the figure, some of the titles were used only once and we will lamped them 

# In[ ]:


#for dset in train:
train_test_data = [train, test]
for dset in train_test_data:
    dset["Title"] = dset["Title"].replace(["Melkebeke", "Countess", "Capt", "the Countess", "Col", "Don",
                                         "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"] , "Lamped")
    dset["Title"] = dset["Title"].replace(["Lady", "Mlle", "Ms", "Mme"] , "Miss")


# In[ ]:


fig2 = sns.countplot(x = 'Title', data = train, palette = "GnBu_d")
fig2 = plt.setp(fig2.get_xticklabels(), rotation = 45)


# - Title has Mr, Mrs, Miss, Master, and Lamped categorical variables with the most frequent being Mr
# - Title feature correlation with survival by pivoting on title feature

# In[ ]:


train[['Title', 'Survived']].groupby(['Title'], 
                                        as_index = False).mean().sort_values(by = 'Survived', ascending = False)


# - Map the categorical title feature into numerical values (Mr = 1, Miss = 2, Mrs = 3, Master = 4, Lamped = 5) 

# In[ ]:


for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map({'Mr': 1, 'Miss': 2, 'Mrs': 3, 
                                             'Master': 4, 'Lamped': 5}).astype(int)
train.head()


# In[ ]:


train.drop(['Name'], axis = 1, inplace = True)
test.drop(['Name'], axis=1, inplace = True)
train.head()


# In[ ]:





# ## 2.8 Heat map for correlation between features

# In the next figure, we will see the correlation among all features using a heat map.

# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize=(16,16))
plt.title('Correlation between Features', y=1.05, size = 20)
sns.heatmap(train.corr(),
            linewidths=0.1, 
            vmax=1.0, 
            square=True, 
            cmap=colormap, 
            linecolor='white', 
            annot=True)


# The figure shows the linear relationship between the individual features vs survival and also between each feature. Example, Pclass vs Survival corr_coef is - 0.34 that means they have -ve leaner relationship. Pclass = 1 has higher survival rate than Pclass = 3.  +Ve corr_coef_ mean the two variables have +ve linear relationship and -ve mean they have -ve linear relationship (slope = -ve), close to zero mean they are not correlated, close to +1 or -1 mean they are strongly positively and negatively correlated, respectively. 

# # 3. Machine Learning Algorithms

# - The first step is to divide the training and test dataset into features dataset and target dataset

# In[ ]:


y_train = train["Survived"]
X_train = train.drop(["Survived"], axis = 1 )

X_test = test
X_train.shape, y_train.shape, X_test.shape


# ##   3.1. Logistic Regression

# In[ ]:


LR = LogisticRegression(random_state = 0)
LR.fit(X_train, y_train)
y_pred_lr = LR.predict(X_test)
LR_score = LR.score(X_train, y_train)
print("LR Accuracy  score = {:.2f}".format(LR_score*100))


# ##  3.2. Support Vector Machine

# In[ ]:


svc = SVC(random_state = 0)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
SVC_score = svc.score(X_train, y_train)
print("SVC Accuracy  score = {:.2f}".format(SVC_score*100))


# ##  3.3. K-Nearest Neignbors Classifier

# In[ ]:


KNN = KNeighborsClassifier(n_neighbors = 5)
KNN.fit(X_train, y_train)
y_pred_knn = KNN.predict(X_test)
KNN_score = KNN.score(X_train, y_train)
print("KNN accuracy score = {:.2f}".format(KNN_score*100))


# ##  3.4. Naive Bayes Classifier

# In[ ]:


GNB = GaussianNB()
GNB.fit(X_train, y_train)
y_pred_gnb = GNB.predict(X_test)
GNB_score = GNB.score(X_train, y_train)
print("GNB accuracy score = {:.2f}".format(GNB_score*100))


# ##  3.5. Linear SVC

# In[ ]:


LSVC = LinearSVC()
LSVC.fit(X_train, y_train)
y_pred_lsvc = LSVC.predict(X_test)
LSVC_score = LSVC.score(X_train, y_train)
print("GNB accuracy score = {:.2f}".format(LSVC_score*100))


# ##  3.6. Perceptron

# In[ ]:


perceptron = Perceptron()
perceptron.fit(X_train, y_train)
y_pred_perceptron = perceptron.predict(X_test)
perceptron_score = perceptron.score(X_train, y_train)
print("perceptron accuracy score = {:.2f}".format(perceptron_score*100))


# ##  3.7. Stochastic Gradient Descent

# In[ ]:


SGD = SGDClassifier()
SGD.fit(X_train, y_train)
y_pred_sgd = SGD.predict(X_test)
SGD_score = SGD.score(X_train, y_train)
print("Stochastic Gradient Descent accuracy score = {:.2f}".format(SGD_score*100))


# ##  3.8. Decision Tree

# In[ ]:


DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
y_pred_dt = DT.predict(X_test)
DT_score = DT.score(X_train, y_train)
print("Decision Tree accuracy score = {:.2f}".format(DT_score*100))


# ##  3.9. Random Forest Regressor 

# In[ ]:


RF = RandomForestRegressor(n_estimators = 1000)
RF.fit(X_train, y_train)
y_pred_rf = RF.predict(X_test)
RF_score = RF.score(X_train, y_train)
print("Random forest regressor accuracy score = {:.2f}".format(RF_score*100))


# # 4. Evaluating Predictive Models

# In[ ]:


Predictive_models = pd.DataFrame({
    'Model': ['SVM', 'KNN', 'LR', 'RF', 'GNB', 
              'Perceptron','SGD', 'LSVC', 'DT'],
    'Score': [SVC_score, KNN_score, LR_score, RF_score, GNB_score, 
              perceptron_score, SGD_score, LSVC_score, DT_score]})
Predictive_models.sort_values(by ='Score', ascending=True)


# #### Note: From this the decision tree model has higher accuracy rate and hence this classification method is selected for further KFold cross-validation, parameter tuning, and variable importance

# ## 4.1. Cross Validation

# - KFold cros-validation is used 

# In[ ]:


DT = DecisionTreeClassifier()
y_train = train.loc[:,"Survived"]
X_train = train.drop(["Survived"], axis = 1)


kfold = KFold(n=len(train), n_folds = 5, shuffle = True, random_state = 0)
kfold_score = cross_val_score(DT, X_train, y_train, cv = kfold)

kfold_score_mean = np.mean(kfold_score)
  
print("Decision Tree accuracy score per fold : ", kfold_score, "\n")
print("Average accuracy score : {:.4f}".format(kfold_score_mean))


# ## 4.2 Parameter tuning

# In[ ]:


DT = DecisionTreeClassifier()
y_train = train.loc[:,"Survived"]
X_train = train.drop(["Survived"], axis = 1)

parameters = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
             'max_features' : ["auto", None, "sqrt", "log2"],
             'random_state': [0, 25, 75, 125, 250],
             'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
clf = GridSearchCV(DT, parameters)
clf.fit(X_train, y_train)

DT_Model = clf.best_estimator_
print (clf.best_score_, clf.best_params_)
print(DT_Model)


# # 5. Final Model 

# In[ ]:


DT = DecisionTreeClassifier(max_depth = 6, 
                            max_features = 'auto', 
                            min_samples_leaf = 2,
                            random_state = 125)
X_train = train.drop(["Survived"], axis = 1)
y_train = train.loc[:,"Survived"]

kfold = KFold(n=len(train), n_folds = 5, shuffle = True, random_state = 125)
DT.fit(X_train, y_train)
kfold_score = cross_val_score(DT, X_train, y_train, cv = kfold)
kfold_score_mean = np.mean(kfold_score)

y_pred_dt = DT.predict(X_test)

  
print("Decision Tree accuracy score per fold : ", kfold_score, "\n")
print("Average accuracy score : {:.4f}".format(kfold_score_mean)) 


# ## 5.1 Variable importance measures

# In[ ]:


DT.feature_importances_


# In[ ]:


feature_importances = pd.Series(DT.feature_importances_, index = X_train.columns).sort_values()
#feature_importances.sort()
feature_importances.plot(kind = "barh", figsize = (7,6));
plt.title(" feature ranking", fontsize = 20)
plt.show()


# In[ ]:


Titanic_submission = pd.DataFrame({
        "PassengerId": test_PassengerId,
        "Survived": y_pred_dt
    })


# In[ ]:


Titanic_submission.to_csv("Titanic_compet_submit.csv", index = False)


# References: Manav Sehgal, Omar El Gabry, Sina, Antonello, Jeff Delaney

# # 6. Pipeline

# For optimal performance, most learning algorithms need input features on the same scale therefore the first step is that the features need to standardize the columns of the given dataset before we can feed them to an estimator. As this project is part of my learning and excerise, I want to compress the training data from the initial 13 to a lower three-dimensional subspace via principal component analysis (PCA), a feature extraction technique for dimensionality reduction. Instead of going through the fitting and transformation steps for the training and test dataset separately, we can chain the StandardScaler, PCA, and Estimator objects in a pipeline.  The next step is cross validation and model selection. A good approach for model selection is to separate a dataset into three parts: a training set, a validation set, and a test set. The training set is used to fit the different models, and the performance on the validation set is then used for the model selection. The advantage of having a test set that the model hasn't seen before during the training and model selection steps is that we can obtain a less biased estimate of its ability to generalize to new data. In stratified cross-validation, the class proportions are preserved in each fold to ensure that each fold is representative of the class proportions in the training dataset.

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
pipe_svm = Pipeline([('scl', StandardScaler()),
            ('pca', PCA(n_components=3)),
            ('clf', SVC(random_state=0))])
scores = cross_val_score(estimator=pipe_svm, 
                          X=X_train, y=y_train, 
                          cv=10, n_jobs=-1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))) 


# In[ ]:


from sklearn.learning_curve import learning_curve
pipe_svm = Pipeline([('scl', StandardScaler()),            
                     ('pca', PCA(n_components=3)),
                    ('clf', SVC(random_state = 0))])
train_sizes, train_scores, valid_scores = learning_curve(estimator=pipe_svm, 
                       X=X_train, 
                       y=y_train, 
                       train_sizes=np.linspace(0.1, 1.0, 10), 
                       cv=10,
                       n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)
plt.plot(train_sizes, train_mean, 
          color='blue', marker='o', 
          markersize=5, 
          label='training accuracy')
plt.fill_between(train_sizes, 
                  train_mean + train_std,
                  train_mean - train_std, 
                  alpha=0.15, color='blue')
plt.plot(train_sizes, valid_mean, 
          color='green', linestyle='--', 
          marker='s', markersize=5, 
          label='validation accuracy')
plt.fill_between(train_sizes, 
                  valid_mean + valid_std,
                  valid_mean - valid_std, 
                  alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.65, 0.9])
plt.show()


# In[ ]:


from sklearn.learning_curve import validation_curve
pipe_svm = Pipeline([('scl', StandardScaler()),            
#                    ('pca', PCA(n_components=3)),
                    ('clf', SVC(random_state = 0))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 1000.0, 10000.0]
train_scores, vald_scores = validation_curve(
                estimator=pipe_svm, 
                 X=X_train, 
                 y=y_train, 
                 param_name='clf__C', 
                 param_range=param_range,
                 cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)
plt.plot(param_range, train_mean, 
          color='blue', marker='o', 
          markersize=5, 
          label='training accuracy')
plt.fill_between(param_range, train_mean + train_std,
                  train_mean - train_std, alpha=0.15,
                  color='blue')
plt.plot(param_range, valid_mean, 
          color='green', linestyle='--', 
          marker='s', markersize=5, 
          label='validation accuracy')
plt.fill_between(param_range, 
                  valid_mean + valid_std,
                  valid_mean - valid_std, 
                  alpha=0.15, color='green')
plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.5, 0.9])
plt.show()


# In[ ]:


pipe_svm = Pipeline([('scl', StandardScaler()),
#                     ('pca', PCA(n_components = 2)),
                      ('clf', SVC(random_state=1))])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'clf__C': param_range, 
                'clf__kernel': ['linear']},
               {'clf__C': param_range, 
                'clf__gamma': param_range, 
                'clf__kernel': ['rbf']}]
gs = GridSearchCV(estimator=pipe_svm, 
                   param_grid=param_grid, 
                   scoring='accuracy', 
                   cv=10,
                   n_jobs=-1)
clf = gs.fit(X_train, y_train)
print(clf.best_score_) 
print(clf.best_params_)


# In[ ]:


y_pred_pip = clf.predict(X_test)
print('y_pred_pip: {:.3f}',format(y_pred_pip),"\n")


# In[ ]:


gs = GridSearchCV(estimator=pipe_svm, 
                   param_grid=param_grid,
                   scoring='accuracy', 
                   cv=2, 
                   n_jobs=-1)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV accuracy scores: {:.3f}',format(scores))
print('CV accuracy: {:.3f} +/- {:.3f}',format((np.mean(scores), np.std(scores))))


# In[ ]:


Titanic_submission = pd.DataFrame({
        "PassengerId": test_PassengerId,
        "Survived": y_pred_pip
    })


# In[ ]:


Titanic_submission.to_csv("Titanic_compet_submit_3.csv", index = False)


# In[ ]:




