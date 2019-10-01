#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import nltk
import plotly
import re
          
plotly.offline.init_notebook_mode() # run at the start of every notebook
import cufflinks as cf

cf.go_offline()
cf.getThemes()
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
get_ipython().magic(u'matplotlib inline')
import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from IPython.display import display

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
full_data = [df_train,df_test]


# In[ ]:


df_train.info()


# In[ ]:


# Function to calculate no. of null values with percentage in the dataframe
def null_values(DataFrame_Name):
    
    sum_null = DataFrame_Name.isnull().sum()
    total_count = DataFrame_Name.isnull().count()
    percent_nullvalues = sum_null/total_count * 100
    df_null = pd.DataFrame()
    df_null['Total_values'] = total_count
    df_null['Null_Count'] = sum_null
    df_null['Percent'] = percent_nullvalues
    df_null = df_null.sort_values(by='Null_Count',ascending = False)

    return(df_null)


# In[ ]:


null_values(df_train)


# In[ ]:


null_values(df_test)


# In[ ]:


df_train.describe()


# In[ ]:


df_train.head(5)


# # Correlation

# In[ ]:


## get the most important variables. 
corr = df_train.corr()**2
corr.Survived.sort_values(ascending=False)


# In[ ]:


## heatmeap to see the correlation between features. 
# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(df_train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize = (15,12))
sns.heatmap(df_train.corr(), 
            annot=True,
            mask = mask,
            cmap = 'RdBu_r',
            linewidths=0.1, 
            linecolor='white',
            vmax = .9,
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20);


# # Pclass

# In[ ]:


# Lets start with Pclass column - Already an integer - good
# Lets check the impact of this column on the survived column in the train dataset.
# We will calculate mean of survived people in each class - This will tell us how many survived out of total for each class
df_train[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean()


# From the above calculation, it seems like passenger from class1 ( mostly rich ) survived with a maximum percentage
# and passengers from a lower class survived least

# In[ ]:


sns.barplot('Pclass','Survived', data=df_train)


# In[ ]:


# Kernel Density Plot
fig = plt.figure(figsize=(15,8),)
## I have included to different ways to code a plot below, choose the one that suites you. 
ax=sns.kdeplot(df_train.Pclass[df_train.Survived == 0] , 
               color='gray',
               shade=True,
               label='not survived')
ax=sns.kdeplot(df_train.loc[(df_train['Survived'] == 1),'Pclass'] , 
               color='g',
               shade=True, 
               label='survived')
plt.title('Passenger Class Distribution - Survived vs Non-Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Passenger Class", fontsize = 15)
## Converting xticks into words for better understanding
labels = ['Upper', 'Middle', 'Lower']
plt.xticks(sorted(df_train.Pclass.unique()), labels);


# # Sex

# In[ ]:


females=df_train['Sex'].apply(lambda x: x.count('female')).sum()
print('Total males=',891-females)
print('Total females=',females)


# In[ ]:


# Now lets focus on the Sex column and evaluate its impact on the survived column
df_train[['Sex','Survived']].groupby(['Sex'],as_index = False).mean()


# From the above calculation it is clear that the female survival rate is much higher than the male survivor rate

# In[ ]:


sns.barplot(x='Sex', y='Survived', data=df_train)


# # Embarked

# In[ ]:


df_train[['Embarked','Survived']].groupby(['Embarked'],as_index = False).mean()


# People with destination C = Cherbourg (C = Cherbourg, Q = Queenstown, S = Southampton) survived with highest percentage

# In[ ]:


sns.barplot(x='Embarked', y='Survived', data=df_train)


# # SibSp & Parch = Family_members

# Lets create a new feature column by combining sibling/spouse & parent/children column

# In[ ]:


df_train['Family_members'] = df_train['SibSp'] + df_train['Parch']
df_test['Family_members'] = df_test['SibSp'] + df_test['Parch']
df_train[['Family_members','Survived']].groupby(['Family_members'],as_index=False).mean()


# From the above calculation, we can conclude that - Survival percentage is higher when Family members are #1,2,3
# It is less when you are alone or have family members > 3

# In[ ]:


sns.barplot(x='Family_members', y='Survived', data=df_train)


# In[ ]:


df_train = df_train.drop(['PassengerId'],axis=1)
#df_test = df_test.drop(['PassengerId'],axis=1)


# In[ ]:


full_data = [df_train,df_test]


# # Lets focus on the Missing Values
# Cabin - Removing this column from the dataset- 80% missing values

# In[ ]:


df_train = df_train.drop(['Cabin','Ticket'],axis=1)

df_test = df_test.drop(['Cabin','Ticket'],axis=1)


# In[ ]:


full_data = [df_train,df_test]


# In[ ]:


for dataset in full_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(df_train['Title'], df_train['Sex'])


# In[ ]:


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
df_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


sns.barplot(x='Title', y='Survived', data=df_train)


# In[ ]:


df_train


# In[ ]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder,LabelBinarizer
#cat_features = df_train['Title']
#encoder = LabelBinarizer()
#new_cat_features = encoder.fit_transform(cat_features)
#new_cat_features

#pd.get_dummies(df_train, columns=['Title'], prefix=['Title'])


# In[ ]:


df_train = df_train.drop(['Name'],axis = 1)
df_test = df_test.drop(['Name'],axis = 1)


# # Age
# We will find the average age for every category in the age column and then impute the mean value for the respective category

# In[ ]:



df_train[['Title','Age']].groupby(['Title'],as_index = False).mean().sort_values(by='Age')


# In[ ]:


Mean_Age = df_train[['Title','Age']].groupby(['Title'],as_index = False).mean().sort_values(by='Age')
sns.barplot(x='Title', y='Age', data=Mean_Age)


# In[ ]:


df_train['Age'] = df_train['Age'].fillna(-1)
df_test['Age'] = df_test['Age'].fillna(-1)  
full_data = [df_train,df_test]


# # Age
# Null Values - 20% - Imputing the mean value per category as calculated above

# In[ ]:



for dataset in full_data:
    
    dataset.loc[(dataset['Age'] == -1) &(dataset['Title'] == 'Master'), 'Age'] = 4.57
    dataset.loc[(dataset['Age'] == -1) &(dataset['Title'] == 'Miss'), 'Age'] = 21.84
    dataset.loc[(dataset['Age'] == -1) &(dataset['Title'] == 'Mr'), 'Age'] = 32.36
    dataset.loc[(dataset['Age'] == -1) &(dataset['Title'] == 'Mrs'), 'Age'] = 35.78
    dataset.loc[(dataset['Age'] == -1) &(dataset['Title'] == 'Rare'), 'Age'] = 45.54
    dataset['Age'] = dataset['Age'].astype(int)   
    


# Now creating different age bands...

# In[ ]:


full_data = [df_train, df_test]
for dataset in full_data:
    
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 7


# In[ ]:


df_train[['Sex','Age','Survived']].groupby(['Sex','Age'],as_index=False).mean()


# In[ ]:


agesexsurv = df_train[['Sex','Age','Survived']].groupby(['Sex','Age'],as_index=False).mean()
sns.factorplot('Age','Survived','Sex', data=agesexsurv
                ,aspect=3,kind='bar')
plt.suptitle('AgeBand,Sex vs Survived')


# # I want to create different categories for family members
# as calculated above
# Family_members vs Survived
# 
# 
# category 0 = person is alone - survival chance = 30%,
# category 1 = person has family members = 1,2 - survival chance = 56%,
# category 2 = person has family members = 3 - survival chance = 72%
# category 3 = person has family members = 4,5 - survival chance = approx 17%,
# category 4 = person has family members = 6 - survival chance = 33%
# category 5 = person has family members = 7,10 - survival chance = 0%
# 

# In[ ]:


full_data = [df_train, df_test]
for dataset in full_data:
    
    dataset.loc[ dataset['Family_members'] == 0, 'Family_members_Band'] = 0
    dataset.loc[(dataset['Family_members'] == 1)|(dataset['Family_members'] == 2),'Family_members_Band'] = 1
    dataset.loc[ dataset['Family_members'] == 3, 'Family_members_Band'] = 2
    dataset.loc[(dataset['Family_members'] == 4)|(dataset['Family_members'] == 5),'Family_members_Band'] = 3
    dataset.loc[ dataset['Family_members'] == 6, 'Family_members_Band'] = 4
    dataset.loc[(dataset['Family_members'] == 7)|(dataset['Family_members'] == 10),'Family_members_Band'] = 5
    dataset['Family_members_Band'] = dataset['Family_members_Band'].astype(int)


# # Creating Categories for Fare column

# In[ ]:


df_train['FareBand'] = pd.qcut(df_train['Fare'], 4)
df_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[ ]:


FarePlot = df_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand')
sns.barplot(x='FareBand', y='Survived', data=FarePlot)


# In[ ]:


df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].dropna().mean()) # df_test has one null value


# In[ ]:


full_data = [df_train,df_test]
for dataset in full_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare_Band'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare_Band'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare_Band'] = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare_Band'] = 3
    dataset['Fare_Band'] = dataset['Fare_Band'].astype(int)


# In[ ]:


sns.factorplot('Fare_Band','Survived','Sex', data=df_train
                ,aspect=3,kind='bar')
plt.suptitle('FareBand,Sex vs Survived')


# From the above graph, thing to notice is that: the males survival rate increases as the fare for the ticket increases but for the females, the survival rate is almost similar for all the fare bands

# # Embarked Column

# In[ ]:


most_frequent = df_train['Embarked'].mode()[0]
most_frequent


# In[ ]:


full_data = [df_train,df_test]
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna(most_frequent)
    
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


embarkedgraph = df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Embarked',y='Survived',data=embarkedgraph)


# Dropping Values

# In[ ]:


df_train = df_train.drop(['SibSp','Parch','Fare','Family_members','FareBand'],axis = 1)


# In[ ]:


df_test = df_test.drop(['SibSp','Parch','Fare','Family_members'],axis = 1)


# # One Hot Encoding

# In[ ]:


X_train = pd.get_dummies(df_train, columns=['Pclass','Sex','Age','Embarked','Title','Family_members_Band','Fare_Band'], prefix=['Pclass','Sex'
                                                                ,'Age','Embarked','Title','Family_members_Band','Fare_Band'])


# In[ ]:


Y_train = X_train['Survived']
X_train = X_train.drop('Survived', axis=1)


# In[ ]:


X_train.shape


# In[ ]:


X_test = pd.get_dummies(df_test, columns=['Pclass','Sex','Age','Embarked','Title','Family_members_Band','Fare_Band'], prefix=['Pclass','Sex','Age','Embarked','Title','Family_members_Band','Fare_Band'])


# In[ ]:


X_test.shape


# In[ ]:


X_test=X_test.drop(['PassengerId'],axis=1)


# # Testing Machine Learning Models

# In[ ]:


# stochastic gradient descent (SGD) learning
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)

sgd.score(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)


print(round(acc_sgd,2,), "%")


# In[ ]:


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")


# In[ ]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(round(acc_log,2,), "%")


# In[ ]:


# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(round(acc_knn,2,), "%")


# In[ ]:


# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print(round(acc_gaussian,2,), "%")


# In[ ]:


# Perceptron
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print(round(acc_perceptron,2,), "%")


# In[ ]:


# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(round(acc_linear_svc,2,), "%")


# In[ ]:


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(round(acc_decision_tree,2,), "%")


# In[ ]:


results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)

result_df.head(9)


# # Best Model

# In[ ]:


bestmodelgraph = result_df.head(9)
ax = sns.factorplot("Model", y="Score", data=bestmodelgraph,
                palette='Blues_d',aspect=3.5,kind='bar')


# # K-FOLD Validation

# In[ ]:


from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")


# In[ ]:


print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# In[ ]:


# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
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
             label="Validation score")

    plt.legend(loc="best")
    return plt


# In[ ]:


# Plot learning curves
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

title = "Learning Curves (Random Forest)"
cv = 10
plot_learning_curve(rf, title, X_train, Y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=1);


# # Feature Importance

# In[ ]:


importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False)


# In[ ]:


importances_most = importances.head(10) # 10 most important features
axes = sns.factorplot('feature','importance', 
                      data=importances_most, aspect = 4, )


# In[ ]:


importances_least = importances.tail(10) # least 10 important features
axes = sns.factorplot('feature','importance', 
                      data=importances_least, aspect = 4,)


# In[ ]:


# Random Forest , Testing with oob score

random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")


# # OOB Score

# Our random forest model predicts as good as it did before. A general rule is that, the more features you have, the more likely your model will suffer from overfitting and vice versa. But I think our data looks fine for now and hasn't too much features.
# 
# There is also another way to evaluate a random-forest classifier, which is probably much more accurate than the score we used before. What I am talking about is the out-of-bag samples to estimate the generalization accuracy. I will not go into details here about how it works. Just note that out-of-bag estimate is as accurate as using a test set of the same size as the training set. Therefore, using the out-of-bag error estimate removes the need for a set aside test set.

# In[ ]:


print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


# # Hyperparameter Tuning
# Below you can see the code of the hyperparamter tuning for the parameters criterion, min_samples_leaf, min_samples_split and n_estimators.
# 
# I put this code into a markdown cell and not into a code cell, because it takes a long time to run it. Directly underneeth it, I put a screenshot of the gridsearch's output.
# 
# param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10, 25, 50, 70], "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100, 400, 700, 1000, 1500]}
# 
# from sklearn.model_selection import GridSearchCV, cross_val_score
# 
# rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
# 
# clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)
# 
# clf.fit(X_train, Y_train)
# 
# clf.bestparams

# # Testing new parameters from hypertuning

# In[ ]:


# Random Forest
random_forest = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


# Now that we have a proper model, we can start evaluating it's performace in a more accurate way. Previously we only used accuracy and the oob score, which is just another form of accuracy. The problem is just, that it's more complicated to evaluate a classification model than a regression model. We will talk about this in the following section.

# # Confusion Matrix

# In[ ]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
confusion_matrix(Y_train, predictions)


# The first row is about the not-survived-predictions: 489 passengers were correctly classified as not survived (called true negatives) and 60 were wrongly classified as not survived (false positives).
# 
# The second row is about the survived-predictions: 100 passengers where wrongly classified as survived (false negatives) and 242 were correctly classified as survived (true positives).

# In[ ]:


conf_mat = confusion_matrix(Y_train, predictions)
TP = conf_mat[0][0]
FP = conf_mat[0][1]
FN = conf_mat[1][0]
TN = conf_mat[1][1]

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
print('Sensitivity, hit rate, recall, or true positive rate=',TPR)

# Specificity or true negative rate
TNR = TN/(TN+FP) 
print('Specificity or true negative rate=',TNR)

# Precision or positive predictive value
PPV = TP/(TP+FP)
print('Precision or positive predictive value=',PPV)

# Negative predictive value
NPV = TN/(TN+FN)
print('Negative predictive value=',NPV)

# Fall out or false positive rate
FPR = FP/(FP+TN)
print('Fall out or false positive rate=',FPR)

# False negative rate
FNR = FN/(TP+FN)
print('False negative rate=',FNR)

# False discovery rate
FDR = FP/(TP+FP)
print('False discovery rate=',FDR)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)
print('Overall accuracy=',ACC)


# In[ ]:


positives = pd.DataFrame({
    'Factor': ['True Positives', 'False Positives', ],
    'Score': [TP, FP]})


# In[ ]:


sns.barplot(x='Factor',y='Score',data=positives)


# In[ ]:


negatives = pd.DataFrame({
    'Factor':['True Negative', 'False Negative'],
    'Score':[TN, FN]
})


# In[ ]:


sns.barplot(x='Factor',y='Score',data=negatives)


# In[ ]:


from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(Y_train, predictions))
print("Recall:",recall_score(Y_train, predictions))


# Our model predicts 80% of the time, a passengers survival correctly (precision). 
# The recall tells us that it predicted the survival of 71 % of the people who actually survived.

# # F-Score

# You can combine precision and recall into one score, which is called the F-score. The F-score is computed with the harmonic mean of precision and recall. Note that it assigns much more weight to low values. 
# As a result of that, the classifier will only get a high F-score, if both recall and precision are high.

# In[ ]:


from sklearn.metrics import f1_score
print('F1score',f1_score(Y_train, predictions))


# There we have it, a 75 % F-score. The score is not that high, because we have a recall of 70%.
# 
# But unfortunately the F-score is not perfect, because it favors classifiers that have a similar precision and recall. 
# This is a problem, because you sometimes want a high precision and sometimes a high recall. 
# The thing is that an increasing precision, sometimes results in an decreasing recall and vice versa (depending on the threshold). 
# This is called the precision/recall tradeoff. We will discuss this in the following section.

# # Precision Recall Curve

# For each person the Random Forest algorithm has to classify, it computes a probability based on a function 
# and it classifies the person as survived (when the score is bigger the than threshold) or 
# as not survived (when the score is smaller than the threshold). 
# That's why the threshold plays an important part.

# In[ ]:


from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = random_forest.predict_proba(X_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(Y_train, y_scores)


# In[ ]:


def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()


# Above you can clearly see that the recall is falling of rapidly at a precision of around 84%. 
# Because of that you may want to select the precision/recall tradeoff before that - maybe at around 75 %.
# 
# You are now able to choose a threshold, that gives you the best precision/recall tradeoff 
# for your current machine learning problem. If you want for example a precision of 80%, 
# you can easily look at the plots and see that you would need a threshold of around 0.4. 
# Then you could train a model with exactly that threshold and would get the desired accuracy.
# 
# Another way is to plot the precision and recall against each other

# In[ ]:


def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g--", linewidth=2.5)
    plt.ylabel("recall", fontsize=19)
    plt.xlabel("precision", fontsize=19)
    plt.axis([0, 1.5, 0, 1.5])

plt.figure(figsize=(14, 7))
plot_precision_vs_recall(precision, recall)
plt.show()


# # ROC AUC Curve

# Another way to evaluate and compare your binary classifier is provided by the ROC AUC Curve. 
# This curve plots the true positive rate (also called recall) against the false positive rate (ratio of incorrectly classified negative instances), instead of plotting the precision versus the recall.

# In[ ]:


from sklearn.metrics import roc_curve
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train, y_scores)


# In[ ]:


# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()


# The red line in the middel represents a purely random classifier (e.g a coin flip) and therefore your classifier should be as far away from it as possible. Our Random Forest model seems to do a good job.
# 
# Of course we also have a tradeoff here, because the classifier produces more false positives, the higher the true positive rate is.

# # ROC AUC Score

# The ROC AUC Score is the corresponding score to the ROC AUC Curve. It is simply computed by measuring the area under the curve, which is called AUC.
# 
# A classifiers that is 100% correct, would have a ROC AUC Score of 1 and a completely random classiffier would have a score of 0.5.

# In[ ]:


from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(Y_train, y_scores)
print("ROC-AUC-Score:", r_a_score)


# **Submission**

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_prediction
    })
submission.to_csv('submission.csv', index=False)


# **More to come...
# Please provide suggestions if there is need of improvement.
# Please upvote if the kernel was useful**
