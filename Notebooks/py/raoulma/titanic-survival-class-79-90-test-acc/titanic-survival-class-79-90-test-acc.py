#!/usr/bin/env python
# coding: utf-8

# **Author:** Raoul Malm  
# **Description:** Given is a training set of samples listing passengers who survived or did not survive the Titanic disaster. The goal is to construct a model that can predict from a test dataset not containing the survival information if these passengers in the test dataset survived or not. This is a supervised classification task. The individual steps for the solution are:
# - Analyze data
# - Manipulate data: complete, convert, create, delete features
# - Model data: kNN, SVC, Decision Tree, Random Forest, Neural Networks
# 
# **Results:** Using a split of 90%/10% on the labeled training data this implementation, training on data of 801 passengers, achieves a 86% accuracy on the validation set of 90 passengers.  
# **Reference:** [Titanic Data Science Solutions by Manav Sehgal](https://www.kaggle.com/startupsci/titanic-data-science-solutions?scriptVersionId=1145136)
# 
# 

# # Libraries and Settings

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.neighbors
import sklearn.naive_bayes
import sklearn.tree
import sklearn.neural_network
from subprocess import check_output
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tensorflow as tf
get_ipython().magic(u'matplotlib inline')

train_set_size = 891;
valid_set_size = 0;

#display parent directory and working directory
print(os.path.dirname(os.getcwd())+':', os.listdir(os.path.dirname(os.getcwd())));
print(os.getcwd()+':', os.listdir(os.getcwd()));


# # Analyze Data
# 
# The train/test sets have 891/418 rows with 12/11 columns. The features are:
# - Survived: 0 = No, 1 = Yes 
# - Pclass: Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd 
# - Name: Name of the passenger
# - Sex: male, female 
# - Age: Age in years. Is fractional if less than 1. If the age is estimated, it is in the form of xx.5.
# - SibSp: # of siblings / spouses aboard the Titanic (Sibling = brother, sister, stepbrother, stepsister, Spouse = husband, wife). Mistresses and fiancés were ignored
# - Parch: # of parents / children aboard the Titanic (Parent = mother, father, Child = daughter, son, stepdaughter, stepson). Some children travelled only with a nanny, therefore Parch=0 for them.
# - Ticket: Ticket number 
# - Fare: Passenger fare 
# - Cabin: Cabin number 
# - Embarked: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton
# 
# The features can be characterized by different types:
# - numerical: Age (continuous, float64), Fare (continuous, float64), SibSp (discrete, int64), Parch (discrete, int64)
# - categorial: Sex (string), Pclass (int64), Embarked (character), Survived (int64), Ticket (alphanumeric, string), Cabin (alphanumeric, string), Name (string)
# 

# In[ ]:


# read data and have a first look at it
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
train_df.info()
print('_'*40)
test_df.info()


# In[ ]:


# look at the first five rows
train_df.head()


# In[ ]:


# look at the first five rows
test_df.head() 


# In[ ]:


# describe numerical data
train_df.describe()


# In[ ]:


# describe numerical data
test_df.describe()


# In[ ]:


# describe object data
train_df.describe(include=['O'])


# In[ ]:


# describe object data
test_df.describe(include=['O'])


# In[ ]:


# check Pclass - Survived correlation
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# check Sex - Survived correlation
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# check SibSp - Survived correlation
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# check Parch - Survived correlation
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# Age histograms depending on Survived
grid = sns.FacetGrid(train_df, col='Survived');
grid.map(plt.hist, 'Age', bins=20);


# In[ ]:


# Age histograms depending on Survived, Pclass
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


# In[ ]:


# Survived values depending on Embarked, Sex
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6);
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep');
grid.add_legend();


# In[ ]:


# Fare depending on Embarked, Survived, Sex
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# # Manipulate Data
# 
# By having analyzed the data we will perform the following steps:
# 
# - create new feature: Title
# - delete features: Ticket, Cabin, Name, PassengerId
# - convert features: Sex
# - complete and convert feature: Age
# - create new features: IsAlone, Age*Class
# - complete and convert feature: Embarked 
# - complete and convert feature: Fare

# ### Create new feature: Title

# In[ ]:


# extract title from Name and then create new feature: Title  
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

# reduce the number of titles
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                                 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
#train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#print('train_df.shape=',train_df.shape)

# map the title to int64
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()


#  ### Delete features: Ticket, Cabin, Name, PassengerId

# In[ ]:


# delete columns: Ticket, Cabin, Name, PassengerId
train_df = train_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
print("train_df = ", train_df.shape)
print("test_df = ", test_df.shape)


#  ### Convert features: Sex

# In[ ]:


# convert variable 'Sex' into type int64
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


# ### Complete and convert feature: Age

# In[ ]:


# complete missing age entries by using information on Sex, Pclass
guess_ages = np.zeros((2,3));

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & 
                               (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()
            guess_ages[i,j] = int(age_guess/0.5 + 0.5 ) * 0.5
            #print(age_guess)
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & 
                        (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()


# In[ ]:


# create new feature AgeBand
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[ ]:


# Replace Age with ordinals based on the bands in AgeBand
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()


# In[ ]:


# remove AgeBand
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()


# ### Create new features: IsAlone, Age*Class

# In[ ]:


# create new feature FamilySize
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# create new feature IsAlone
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# In[ ]:


# remove features: Parch, SibSp, FamilySize
#train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
#test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
combine = [train_df, test_df]
train_df.head()


# In[ ]:


# create new feature Age*Class
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

#train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
train_df.head()


# ### Complete and convert feature: Embarked 

# In[ ]:


# most frequent occurence of Embarked value
freq_port = train_df.Embarked.dropna().mode()[0]
print(freq_port);

# replace na entries with most frequent value of Embarked
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# map Embarked values to integer values
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


# ### Complete and convert feature: Fare

# In[ ]:


# complete feature Fare
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


# In[ ]:


# create feature FareBand
train_df['FareBand'] = pd.qcut(train_df['Fare'], 6)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[ ]:


# replace feature Fare by ordinals based on FareBand
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head()


# 
# # Model Data
# 
# Supervised learning plus cassification limits the number of machine learning algorithms to: 
#     - Logistic Regression
#     - kNN (k-Nearest Neighbors)
#     - SVM (Support Vector Machine) with different kernels
#     - Gaussian Naive Bayes
#     - Decision Tree
#     - Random Forrest
#     - Perceptron
#     - Multi-layer Perceptron
#     - Deep Neural Network

# In[ ]:


# create subsets for training, validation and testing
X_train = train_df.drop("Survived", axis=1)[0:train_set_size]
Y_train = train_df["Survived"][0:train_set_size]
if valid_set_size > 0:
    X_valid = train_df.drop("Survived", axis=1)[train_set_size:train_set_size+valid_set_size]
    Y_valid = train_df["Survived"][train_set_size:train_set_size+valid_set_size]
else:
    X_valid = train_df.drop("Survived", axis=1)[801:]
    Y_valid = train_df["Survived"][801:]
    
X_test  = test_df.drop("PassengerId", axis=1).copy()

print("training data: ", 'X_train.shape = ', X_train.shape, 'Y_train.shape = ', Y_train.shape)
print("validation data: ", 'X_valid.shape = ', X_valid.shape, 'Y_valid.shape = ', Y_valid.shape)
print("test data: ", 'X_test.shape = ', X_test.shape)


# In[ ]:


# normalize features
X_train_norm = (X_train)/(X_train.max()-X_train.min());
X_valid_norm = (X_valid)/(X_valid.max()-X_valid.min());
X_test_norm = (X_test)/(X_test.max()-X_test.min());


# In[ ]:


"""
# skip features
feature = 'Age*Class';
if feature in X_train.columns:
    X_train = X_train.drop(feature, axis=1)
if feature in X_valid.columns:
    X_valid = X_valid.drop(feature, axis=1)
if feature in X_test.columns:
    X_test = X_test.drop(feature, axis=1)
"""   
print('X_train.columns = ', X_train.columns.values)
print('X_valid.columns = ', X_valid.columns.values)
print('X_test.columns = ', X_test.columns.values)


# In[ ]:


## Logistic Regression as a benchmark model

logreg = sklearn.linear_model.LogisticRegression()
logreg.fit(X_train_norm, Y_train)
Y_log_pred = logreg.predict(X_test_norm)
acc_log_train = np.round(logreg.score(X_train_norm, Y_train), 4)
acc_log_valid = np.round(logreg.score(X_valid_norm, Y_valid), 4)
print('Logistic Regression: train/valid Acc = %.4f/%.4f'%(acc_log_train, acc_log_valid))
coeff_df = pd.DataFrame(X_train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)


# In[ ]:


## Further Machine Learning Algorithms

# support vector machine with rbf kernel
svc_rbf = sklearn.svm.SVC(kernel='rbf')
svc_rbf.fit(X_train_norm, Y_train)
Y_pred_svc_rbf = svc_rbf.predict(X_test_norm)
acc_svc_rbf_train = np.round(svc_rbf.score(X_train_norm, Y_train), 4)
acc_svc_rbf_valid = np.round(svc_rbf.score(X_valid_norm, Y_valid), 4)
print('SVC rbf kernel: train/valid Acc = %.4f/%.4f'%(acc_svc_rbf_train, acc_svc_rbf_valid))

# support vector machine with linear kernel
svc_linear = sklearn.svm.SVC(kernel='linear')
svc_linear.fit(X_train_norm, Y_train)
Y_pred_svc_linear = svc_linear.predict(X_test_norm)
acc_svc_linear_train = np.round(svc_linear.score(X_train_norm, Y_train), 4)
acc_svc_linear_valid = np.round(svc_linear.score(X_valid_norm, Y_valid), 4)
print('SVC linear kernel: train/valid Acc = %.4f/%.4f'%(acc_svc_linear_train, acc_svc_linear_valid))

# k-Nearest-Neighbour Algorithm
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_norm, Y_train)
Y_pred_knn = knn.predict(X_test_norm)
acc_knn_train = np.round(knn.score(X_train_norm, Y_train), 4)
acc_knn_valid = np.round(knn.score(X_valid_norm, Y_valid), 4)
print('kNN: train/valid Acc = %.4f/%.4f'%(acc_knn_train, acc_knn_valid))

# Gaussian Naive Bayes
gaussianNB = sklearn.naive_bayes.GaussianNB()
gaussianNB.fit(X_train_norm, Y_train)
Y_pred_gaussianNB = gaussianNB.predict(X_test_norm)
acc_gaussianNB_train = np.round(gaussianNB.score(X_train_norm, Y_train), 4)
acc_gaussianNB_valid = np.round(gaussianNB.score(X_valid_norm, Y_valid), 4)
print('Gaussian Naive Bayes: train/valid Acc = %.4f/%.4f'%(acc_gaussianNB_train, acc_gaussianNB_valid))

# Decision Tree
decision_tree = sklearn.tree.DecisionTreeClassifier()
decision_tree.fit(X_train_norm, Y_train)
Y_pred_decision_tree = decision_tree.predict(X_test_norm)
acc_decision_tree_train = np.round(decision_tree.score(X_train_norm, Y_train), 4)
acc_decision_tree_valid = np.round(decision_tree.score(X_valid_norm, Y_valid), 4)
print('Decision Tree: train/valid Acc = %.4f/%.4f'%(acc_decision_tree_train, acc_decision_tree_valid))

# Random Forest
random_forest = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
random_forest.fit(X_train_norm, Y_train)
Y_pred_random_forest = random_forest.predict(X_test_norm)
random_forest.score(X_train_norm, Y_train)
acc_random_forest_train = np.round(random_forest.score(X_train_norm, Y_train), 4)
acc_random_forest_valid = np.round(random_forest.score(X_valid_norm, Y_valid), 4)
print('Random Forest: train/valid Acc = %.4f/%.4f'%(acc_random_forest_train, acc_random_forest_valid))

# Perceptron
perceptron = sklearn.linear_model.Perceptron(max_iter = 10000, tol = 1e-6, shuffle = True)
perceptron.fit(X_train_norm, Y_train)
Y_pred_perceptron = perceptron.predict(X_test_norm)
acc_perceptron_train = np.round(perceptron.score(X_train_norm, Y_train), 4)
acc_perceptron_valid = np.round(perceptron.score(X_valid_norm, Y_valid), 4)
print('Perceptron: train/valid Acc = %.4f/%.4f'%(acc_perceptron_train, acc_perceptron_valid))

# Multi Layer Perceptron
mlp = sklearn.neural_network.MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                                           max_iter = 10000, tol = 1e-6, shuffle = False, 
                                           hidden_layer_sizes=(64,32,16), solver ='adam',
                                           learning_rate = 'adaptive',
                                           learning_rate_init=0.001, verbose=False);
mlp.fit(X_train_norm, Y_train)
Y_pred_mlp = mlp.predict(X_test_norm)
acc_mlp_train = np.round(mlp.score(X_train_norm, Y_train), 4)
acc_mlp_valid = np.round(mlp.score(X_valid_norm, Y_valid), 4)
print('MLP: train/valid Acc = %.4f/%.4f'%(acc_mlp_train, acc_mlp_valid))


# In[ ]:


## Deep Neural Network

x_size = X_train_norm.shape[1]; # number of features
y_size = 1; # binary variable
n_n_fc1 = 128; # number of neurons of first layer
n_n_fc2 = 64; # number of neurons of second layer
n_n_fc3 = 32; # number of neurons of third layer

# variables for input and output 
x_data = tf.placeholder('float', shape=[None, x_size])
y_data = tf.placeholder('float', shape=[None, y_size])

# 1.layer: fully connected
W_fc1 = tf.Variable(tf.truncated_normal(shape = [x_size, n_n_fc1], stddev = 0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape = [n_n_fc1]))  
h_fc1 = tf.nn.relu(tf.matmul(x_data, W_fc1) + b_fc1)

# dropout
tf_keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, tf_keep_prob)

# 2.layer: fully connected
W_fc2 = tf.Variable(tf.truncated_normal(shape = [n_n_fc1, n_n_fc2], stddev = 0.1)) 
b_fc2 = tf.Variable(tf.constant(0.1, shape = [n_n_fc2]))  
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) 

# dropout
h_fc2_drop = tf.nn.dropout(h_fc2, tf_keep_prob)

# 3.layer: fully connected
W_fc3 = tf.Variable(tf.truncated_normal(shape = [n_n_fc2, n_n_fc3], stddev = 0.1)) 
b_fc3 = tf.Variable(tf.constant(0.1, shape = [n_n_fc3]))  
h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3) 

# dropout
h_fc3_drop = tf.nn.dropout(h_fc3, tf_keep_prob)

# 3.layer: fully connected
W_fc4 = tf.Variable(tf.truncated_normal(shape = [n_n_fc3, y_size], stddev = 0.1)) 
b_fc4 = tf.Variable(tf.constant(0.1, shape = [y_size]))  
z_pred = tf.matmul(h_fc3_drop, W_fc4) + b_fc4  

# cost function
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_data, logits=z_pred));

# optimisation function
tf_learn_rate = tf.placeholder(dtype='float', name="tf_learn_rate")
train_step = tf.train.AdamOptimizer(tf_learn_rate).minimize(cross_entropy)

# evaluation
y_pred = tf.nn.sigmoid(z_pred);
y_pred_class = tf.cast(tf.greater(y_pred, 0.5),'float')
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_class, y_data ), 'float'))

# start TensorFlow session and initialize global variables
sess = tf.InteractiveSession() 
sess.run(tf.global_variables_initializer())  

keep_prob = 0.2; # dropout regularization with keeping probability
learn_rate_range = [0.01,0.005,0.0025,0.001];
learn_rate_step = 500;

x_train_batch = X_train_norm.iloc[:,:].values.astype('float');
y_train_batch = Y_train.iloc[:].values.reshape(Y_train.shape[0],1).astype('float');

x_valid_batch = X_valid_norm.iloc[:,:].values.astype('float');
y_valid_batch = Y_valid.iloc[:].values.reshape(Y_valid.shape[0],1).astype('float');

x_test_batch = X_test_norm.iloc[:,:].values.astype('float');

n_epoch = 1000; # number of epochs
train_loss, train_acc, valid_loss, valid_acc = np.array([]), np.array([]), np.array([]), np.array([]);
n_step = -1;

# training model
for i in range(0,n_epoch):
    
    if i%learn_rate_step == 0:
        n_step += 1;
        learn_rate = learn_rate_range[n_step];
        print('set learnrate = ', learn_rate)
        
    sess.run(train_step, feed_dict={x_data: x_train_batch, y_data: y_train_batch, tf_keep_prob: keep_prob, 
                                    tf_learn_rate: learn_rate})
    
    if i%100==0:
        train_loss = np.append(train_loss, sess.run(cross_entropy, feed_dict={x_data: x_train_batch, 
                                                                              y_data: y_train_batch, 
                                                                              tf_keep_prob: 1.0}));
    
        train_acc = np.append(train_acc, accuracy.eval(feed_dict={x_data: x_train_batch, 
                                                                  y_data: y_train_batch, 
                                                                  tf_keep_prob: 1.0}));      
    
        valid_loss = np.append(valid_loss, sess.run(cross_entropy, feed_dict={x_data: x_valid_batch, 
                                                                              y_data: y_valid_batch, 
                                                                              tf_keep_prob: 1.0}));
    
        valid_acc = np.append(valid_acc, accuracy.eval(feed_dict={x_data: x_valid_batch, 
                                                                  y_data: y_valid_batch, 
                                                                  tf_keep_prob: 1.0}));      
    
        print('%d epoch: train/val loss = %.4f/%.4f, train/val acc = %.4f/%.4f'%(i+1,
                                                                                 train_loss[-1],
                                                                                 valid_loss[-1],
                                                                                 train_acc[-1],
                                                                                 valid_acc[-1]))

acc_DNN_train = train_acc[-1];
acc_DNN_valid = valid_acc[-1];
# prediction for test set
Y_pred_DNN = y_pred_class.eval(feed_dict={x_data: x_test_batch,tf_keep_prob: 1.0}).astype('int').flatten()

sess.close();


# In[ ]:


# model summary
models = pd.DataFrame({
    'Model': ['SVC with rbf kernel', 'kNN', 'Logistic Regression', 
              'Random Forest', 'Gaussian Naive Bayes', 'Perceptron', 
              'MLP', 'SVC with linear kernel', 'Decision Tree', 'Deep Neural Network'],
    'Train Acc': [acc_svc_rbf_train, acc_knn_train, acc_log_train, 
              acc_random_forest_train, acc_gaussianNB_train, acc_perceptron_train, 
              acc_mlp_train, acc_svc_linear_train, acc_decision_tree_train, acc_DNN_train],
    'Valid Acc': [acc_svc_rbf_valid, acc_knn_valid, acc_log_valid, 
              acc_random_forest_valid, acc_gaussianNB_valid, acc_perceptron_valid, 
              acc_mlp_valid, acc_svc_linear_valid, acc_decision_tree_valid, acc_DNN_valid]})
models.sort_values(by='Valid Acc', ascending=False)


# In[ ]:


# submit the best results
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred_DNN
    })

#if not os.path.exists(os.path.dirname(os.getcwd())+'/output'): 
#    print('create directory ', os.path.dirname(os.getcwd())+'/output')
#    os.makedirs(os.path.dirname(os.getcwd())+'/output')
#submission.to_csv('../output/submission.csv', index=False)
submission.to_csv('submission.csv', index=False)


# In[ ]:




