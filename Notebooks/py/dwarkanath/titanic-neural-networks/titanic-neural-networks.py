#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.saved_model import tag_constants
from sklearn.metrics import confusion_matrix


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.head()


# In[ ]:


train.info()


# As it can be seen, there are several missing data points. Specifically, in age and cabin. 
# 
# Since, we need to predict the 'Survived' column based on other columns, we will need as much data as possible. Steps to do this:
# 
# 1. See correlation between survived and other columns
# 2. Combine train and test dataset
# 3. Extract titles from names
# 4. Fill in missing data
#     1. Age
#     2. Embarked
#     3. Fare
# 5. Add is_alone variable
# 6. Convert data to categorical variables
# 7. Normalize data to be between 0 and 1
# 8. Separate labels (y) from data
# 9. Create training and dev sets
# 10. Set up then neural networks to predict y
# 11. Create submission file and submit
# 12. Try Other Models - SVM and Random Forest
# 13. Average All Predictions

# ## Step 1: See correlation between survived and other columns

# In[ ]:


train.columns
m = train.shape[0]


# Since PassengerId and name is unique to an individual, there is no relation that can be drawn with 'Survived'. 
# 
# However, Pclass, Sex, SibSp, Parch, and Embarked are categorical variables with distinct values that can impact 'Survived'.

# In[ ]:


catCols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
j = 0
for i in catCols:
    plt.figure(j)
    sns.barplot(x = i, y = 'Survived', data = train)
    plt.show()
    j+=1


# We can use all these variables as they all seem to have an impact on the survival of the passenger.

# ## Step 2: Combine train and test dataset
# 
# We will be transforming the training set to make it suitable for use in a machine learning model. Eg. convert string variables to categorical variables. The words 'male' and 'female' cannot be directly used for input. Instead they have to be replaced by 0 and 1 to feed to the model.
# 
# The same transformations also need to be applied on the test set because the trained model expects the input in this format. The best way to achieve this is to combine the two dataset and apply all transformations together. 

# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:


dfs = [train,test]
data = pd.concat(dfs,ignore_index=True)


# In[ ]:


data.head()


# ## Step 3: Extract titles from names

# The names begin with certain titles - Master/Miss for children and Mr. / Mrs. for adults. These can be extracted to create a new feature using RegEx

# In[ ]:


def getTitles(names):
    titleRegex = re.compile(r',.\w+\.')    
    title = []
    for str in names:
        titlePat = re.search(titleRegex,str)
        if titlePat is None:
            title.append(str)
        else:
            x = titlePat.group()
            x = x[2:len(x)-1]
            title.append(x)
    return title


title = getTitles(data['Name'])


# In[ ]:


set(title)


# In[ ]:


def getCleanTitles(title):
    for i in range(len(title)):
        if title[i] in ['Don', 'Sir', 'Jonkheer']:
            title[i] = 'Noble'
        elif title[i] in ['Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)', 'Lady', 'Dona']:
            title[i] = 'Noble'
        elif title[i] in ['Mlle', 'Ms']:
            title[i] = 'Miss'
        elif title[i] == 'Mme':
            title[i] = 'Mrs'
        elif title[i] in ['Capt', 'Col', 'Dr', 'Major', 'Rev']:
            title[i] = 'Other'
    return title

data['Title'] = getCleanTitles(title)


# In[ ]:


data.head()


# ## Step 4: Fill in missing data
# 
# The more data we have available, the better the predictions can be. Hence, instead of discarding rows with data missing in some columns, we will try to fill it with some reasonable value so that the rest of the data in the row does not go to waste.

# In[ ]:


data.info()


# Since Cabin has several missing values, we will not include it in our list of features. However, we can try filling in missing values in Age, Embarked and Fare.

# ### A. Age
# 
# Age is closely associated with Title, hence we will use the mean of ages for titles to replace missing values for rows with those titles.

# In[ ]:


data.groupby('Title').Age.mean()


# In[ ]:


data['Age'].fillna(data.groupby('Title')['Age'].transform("mean"), inplace=True)


# ### B. Embarked
# 

# In[ ]:


data.loc[pd.isnull(data['Embarked'])]


# There are 2 missing values in Embarked which can be found online

# In[ ]:


data.loc[61,'Embarked'] = 'S'
data.loc[829,'Embarked'] = 'S'


# As per this source:
# 
# https://www.encyclopedia-titanica.org/titanic-survivor/amelia-icard.html
# 
# https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html

# ### C. Fare
# 
# Fare still has one missing value. Fill it with the mean

# In[ ]:


data['Fare'].fillna(data['Fare'].mean(), inplace = True)


# ## Step 5: Add is_alone variable
# 
# The chances of survival go up for families. To find if someone is alone, we check if the sum of SibSp and Parch is 0.

# In[ ]:


is_alone = (data['Parch'] + data['SibSp'] == 0).astype(int)


# In[ ]:


data['is_alone'] = is_alone


# In[ ]:


sns.barplot(x = 'is_alone', y = 'Survived', data = data[:m])
plt.show()


# ## Step 6: Convert data to categorical variables

# In[ ]:


data.columns


# In[ ]:


catCols.extend(['Title', 'is_alone'])


# In[ ]:


catCols


# In[ ]:


def convertCatValToNum(catVal):
    le = LabelEncoder()
    le.fit(catVal)
    catVal = le.transform(catVal)
    return catVal


for i in catCols:
    data[i] = convertCatValToNum(data[i])


# ## Step 7: Normalize data to be between 0 and 1
# 
# Since our output is binary i.e. either 0 or 1, training a machine learning algorithm is faster when inputs are also in the same range.

# In[ ]:


data.columns


# Out of the above columns we will select only those we think affect the outcome i.e. 'Survived' column. 

# In[ ]:


Xcols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'is_alone']


# In[ ]:


data[Xcols].info()


# In[ ]:


scaler = MinMaxScaler()
scaler.fit(data[Xcols])
X = scaler.transform(data[Xcols])


# In[ ]:


# Check if the features have been correctly scaled

X_stats = pd.DataFrame()
X_stats['Min'] = np.min(X, axis = 0)
X_stats['Max'] = np.max(X, axis = 0)
X_stats['Mean'] = np.mean(X, axis = 0)
X_stats


# ## Step 8: Separate labels (y) from data

# In[ ]:


y = np.expand_dims(data[:m].Survived.values,1)


# In[ ]:


y.shape


# In[ ]:


# Save preprocessed data to file

X_file = 'X.npy'
#np.save(X_file, X)

y_file = 'y.npy'
#np.save(y_file, y)


# In[ ]:


# Load preprocessed data from file


#X = np.load(X_file)

#y = np.load(y_file)


# ## Step 9: Create training and dev sets
# 
# Validation (dev) set is useful to know how the model performs on data it has not seen. We will randomly select 10% of the data as a validation set. The test set is already provided separately by Kaggle.

# In[ ]:


# Set random seed

seed = 5
np.random.seed(seed)

# Get random training index

train_index = np.random.choice(m, round(m*0.9), replace=False)
dev_index = np.array(list(set(range(m)) - set(train_index)))

test_index = range(m, data.shape[0])
# Make training and dev


X_train = X[train_index]
X_dev = X[dev_index]
X_test = X[test_index]

y_train = y[train_index]
y_dev = y[dev_index]


# In[ ]:


y_dev.shape


# ## Step 10: Set up neural network to predict y

# In[ ]:


# Initialize placeholders for data
n = X.shape[1]
x = tf.placeholder(dtype=tf.float32, shape=[None, n], name = 'inputs_ph')
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name = 'labels_ph')


# In[ ]:


# number of neurons in each layer

input_num_units = n
hidden_num_units = 120
output_num_units = 1


# In[ ]:


# Build Neural Network Weights
initializer = tf.contrib.layers.xavier_initializer()
weights = {
    'hidden1': tf.Variable(initializer([input_num_units, hidden_num_units])),
    'hidden2': tf.Variable(initializer([hidden_num_units, hidden_num_units])),
    'hidden3': tf.Variable(initializer([hidden_num_units, hidden_num_units])),
    'output': tf.Variable(initializer([hidden_num_units, output_num_units])),
}

biases = {
    'hidden1': tf.Variable(initializer([hidden_num_units])),
    'hidden2': tf.Variable(initializer([hidden_num_units])),
    'hidden3': tf.Variable(initializer([hidden_num_units])),
    'output': tf.Variable(initializer([output_num_units])),
}


# In[ ]:


# Build model 

keep_prob = tf.placeholder(dtype=tf.float32, name = 'keep_prob')

hidden_1_layer = tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1'])
hidden_1_layer = tf.nn.dropout(tf.nn.relu(hidden_1_layer),keep_prob = keep_prob)
hidden_2_layer = tf.add(tf.matmul(hidden_1_layer, weights['hidden2']), biases['hidden2'])
hidden_2_layer = tf.nn.dropout(tf.nn.relu(hidden_2_layer),keep_prob = keep_prob)
hidden_3_layer = tf.add(tf.matmul(hidden_2_layer, weights['hidden2']), biases['hidden3'])
hidden_3_layer = tf.nn.dropout(tf.nn.relu(hidden_3_layer),keep_prob = keep_prob)

output_layer = tf.matmul(hidden_3_layer, weights['output']) + biases['output']


# In[ ]:


# Set hyperparameters

learning_rate = 3e-4
epochs = 4000


# In[ ]:


# Set loss function and goal i.e. minimize loss

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_layer, labels=y))
opt = tf.train.AdamOptimizer(learning_rate)
goal = opt.minimize(loss)


# In[ ]:


prediction = tf.round(tf.nn.sigmoid(output_layer), name = 'prediction')
correct = tf.cast(tf.equal(prediction, y), dtype=tf.float32)
recall = tf.metrics.recall(labels = y, predictions = prediction)
accuracy = tf.reduce_mean(correct)


# In[ ]:


# Initialize lists to store loss and accuracy while training the model

loss_trace = []
train_acc = []
dev_acc = []


# In[ ]:


# Start tensorflow session

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# In[ ]:


# Train the model

for epoch in range(epochs):
    
    sess.run(goal, feed_dict={x: X_train, y: y_train, keep_prob: 0.5})

    # calculate results for epoch

    temp_loss = sess.run(loss, feed_dict={x: X_train, y: y_train, keep_prob: 1})
    temp_train_acc = sess.run(accuracy, feed_dict={x: X_train, y: y_train, keep_prob: 1})
    temp_dev_acc = sess.run(accuracy, feed_dict={x: X_dev, y: y_dev, keep_prob: 1})

    # save results in a list

    loss_trace.append(temp_loss)
    train_acc.append(temp_train_acc)
    dev_acc.append(temp_dev_acc)

    # output

    if (epoch + 1) % 200 == 0:
        print('epoch: {:4d} loss: {:5f} train_acc: {:5f} dev_acc: {:5f}'.format(epoch + 1, temp_loss, temp_train_acc, temp_dev_acc))



# ## Step 11: Create submission file and submit

# In[ ]:


plt.plot(loss_trace)


# In[ ]:


y_train_preds_nn = sess.run(prediction, feed_dict ={x: X_train, keep_prob: 1})
y_dev_preds_nn = sess.run(prediction, feed_dict ={x: X_dev, keep_prob: 1})
y_test_preds_nn = sess.run(prediction, feed_dict ={x: X_test, keep_prob: 1})


# In[ ]:


# Save model for future predictions

inputs_dict = {'inputs_ph': x, 'labels_ph': y, 'keep_prob': keep_prob}
outputs_dict = {'prediction': prediction}
tf.saved_model.simple_save(sess, 'simple', inputs_dict, outputs_dict)


# In[ ]:


sess.close()


# In[ ]:


# Restore saved model

graph = tf.get_default_graph()
with tf.Session(graph = graph) as sess:
    # Restore saved values
    print('\nRestoring...')
    tf.saved_model.loader.load(sess, [tag_constants.SERVING], 'simple')
    print('Ok')
    # Get restored placeholders
    x = graph.get_tensor_by_name('inputs_ph:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    
    # Get restored model output
    prediction = graph.get_tensor_by_name('prediction:0')

    # Initialize restored dataset
    y_train_preds_nn_saved = sess.run(prediction, feed_dict={x: X_train, keep_prob: 1})
    y_test_preds_nn_saved = sess.run(prediction, feed_dict={x: X_test, keep_prob: 1})


# In[ ]:


# Check if original and restored predictions are same

confusion_matrix(y_train_preds_nn_saved, y_train_preds_nn)


# Since the confusion matrix only has diagonal elements, the predictions match.

# In[ ]:


def get_recall(labels, preds):
    tp = int(np.dot(labels.T,preds))
    fn = int(np.dot(labels.T,1-preds))
    recall = tp/(tp+fn)
    return recall


# In[ ]:


recall_nn = get_recall(y_train, y_train_preds_nn)
recall_nn


# In[ ]:


test['Survived_nn'] = y_test_preds_nn.astype(int)


# In[ ]:


test[['PassengerId', 'Survived_nn']].to_csv('submission_nn.csv', index = False, header = ['PassengerId', 'Survived'])


# ## Step 12: Try Other Models - SVM and Random Forest

# In[ ]:


svclassifier = SVC()
svclassifier.fit(X_train, y_train) 


# In[ ]:


y_train_preds_svm = np.expand_dims(svclassifier.predict(X_train),1)
train_acc_svm = np.mean(y_train == y_train_preds_svm)
print('Train Accuracy for SVM: {:5f}'.format(train_acc_svm))


# In[ ]:


y_test_preds_svm = svclassifier.predict(X_test)


# In[ ]:


recall_svm = get_recall(y_train, y_train_preds_svm)
recall_svm


# In[ ]:


test['Survived_svm'] = y_test_preds_svm.astype(int)


# In[ ]:


test[['PassengerId', 'Survived_svm']].to_csv('submission_svm.csv', index = False, header = ['PassengerId', 'Survived'])


# In[ ]:


rfclassifier = RandomForestClassifier(n_estimators = 100, max_features = 4)
rfclassifier.fit(X_train, y_train) 


# In[ ]:


y_train_preds_rf = np.expand_dims(rfclassifier.predict(X_train),1)
train_acc_rf = np.mean(y_train == y_train_preds_rf)
print('Train Accuracy for Random Forest: {:5f}'.format(train_acc_rf))


# In[ ]:


y_dev_preds_rf = np.expand_dims(rfclassifier.predict(X_dev),1)
dev_acc_rf = np.mean(y_dev == y_dev_preds_rf)
print('Dev Accuracy for Random Forest: {:5f}'.format(dev_acc_rf))


# In[ ]:


recall_rf = get_recall(y_train, y_train_preds_rf)
recall_rf


# In[ ]:


y_test_preds_rf = rfclassifier.predict(X_test)


# In[ ]:


test['Survived_rf'] = y_test_preds_rf.astype(int)


# In[ ]:


test[['PassengerId', 'Survived_rf']].to_csv('submission_rf.csv', index = False, header = ['PassengerId', 'Survived'])


# # Step 13: Average All Predictions

# In[ ]:


test.columns


# In[ ]:


test['Survived_nn_wtd'] = test['Survived_nn']*recall_nn
test['Survived_svm_wtd'] = test['Survived_svm']*recall_svm
test['Survived_rf_wtd'] = test['Survived_nn']*recall_rf


# In[ ]:


y_test_preds_avg = np.round(np.mean(test[['Survived_nn', 'Survived_svm','Survived_rf']],axis = 1))


# In[ ]:


y_test_preds_wtd_avg = np.round(np.mean(test[['Survived_nn_wtd', 'Survived_svm_wtd','Survived_rf_wtd']],axis = 1))


# In[ ]:


test['Survived_avg'] = y_test_preds_avg.astype(int)


# In[ ]:


test['Survived_wtd_avg'] = y_test_preds_wtd_avg.astype(int)


# In[ ]:


test[['PassengerId', 'Survived_avg']].to_csv('submission_avg.csv', index = False, header = ['PassengerId', 'Survived'])


# In[ ]:


test[['PassengerId', 'Survived_wtd_avg']].to_csv('submission_wtd_avg.csv', index = False, header = ['PassengerId', 'Survived'])

