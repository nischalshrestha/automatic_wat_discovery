#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sb
import sklearn as skl
from matplotlib import pyplot as mpl

train_set = pd.read_csv("../input/train.csv")
test_set = pd.read_csv("../input/test.csv")

full_set = train_set.append(test_set)
full_set.head()



# The following shows the number of items in each column that are missing within the training set.  There are several ways to deal with these missing values.  For the missing age values, we will try to maintain the distribution (assuming it is fairly normal), by generating replacement values from a normal distribution.

# In[2]:


null_counts = full_set.isnull().sum()
null_counts


# We'll start cleaning up these missing entries by looking at how age is distributed in the current dataset.  First, it might be worth it to check if there is a strong correlation between missing entries and dead passengers in the training data (as you can assume people who survived would have their age, cabin and embarkation recorded) . 

# In[3]:


t_n = train_set.shape[0]
age_missing_survived = train_set['Survived'][train_set['Age'].isnull()]
cabin_missing_survived = train_set['Survived'][train_set['Cabin'].isnull()]
embarked_missing_survived = train_set['Survived'][train_set['Embarked'].isnull()]
known_survivors = train_set['Survived']#.where(pd.notnull(train_set['Survived']))
[[str(known_survivors.sum()) + ' / ' + str(known_survivors.shape[0]) + " known passengers survived or " + str(100 * round(1 - known_survivors.sum() / known_survivors.shape[0], 2)) + "% did not survive"],
[str(age_missing_survived.sum()) + ' / ' + str(age_missing_survived.shape[0]) + " passengers with missing ages survived or " + str(100 * round(1 - age_missing_survived.sum() / age_missing_survived.shape[0], 2)) + "% did not survive" ],
[str(cabin_missing_survived.sum()) + ' / ' + str(cabin_missing_survived.shape[0]) + " passengers with missing cabins survived or " + str(100 * round(1 - cabin_missing_survived.sum() / cabin_missing_survived.shape[0], 2)) + "% did not survive"],
[str(embarked_missing_survived.sum()) + ' / ' + str(embarked_missing_survived.shape[0]) + " passengers with missing embarkments survived or " + str(100 * round(1 - embarked_missing_survived.sum() / embarked_missing_survived.shape[0], 2)) + "% did not survive"]]


# Although there is a fairly high correlation between dead passengers and missing ages (71.0%) or missing cabin numbers (70%), it's not enough to just assume a missing age or cabin = a missing passenger.  All the passengers who did not have an embarkation location survived, so it should be easy to fill these values in with useful data. 
# 
# We'll continue by looking at the known distribution of ages in the dataset to get an idea of how 'normal' it is before generating a placeholder distribution.

# In[4]:


ages_present = full_set[pd.notnull(full_set['Age'])]['Age'].sort_values()
ages_present.plot.hist(100)


# The ages within the dataset appear to be approximately normally distributed.  This means we should be able to replace values using interpolation.  Once we replace missing values with interpolated values, we can check if the distribution looks similar.  Ideally the new age values will be from the same distribution, and will hold information that could be useful in identifying the status of the passenger.

# In[5]:


full_set['Age'] = full_set['Age'].interpolate()
full_set.sort_values(['Age'], ascending = True).head(3)


# Next we can look at how the interpolated data compares with the original age distribution.  Since most of the age data was already there and the missing values are randomly distributed within the data, we can expect that the new data will only reinforce the data from the distribution of known ages, as this is the best that interpolation can ever do.  We see that this is indeed the case.

# In[6]:


full_set[:]['Age'].plot.hist(100)


# Now we should deal with the missing cabin values.  Since most of the dataset is missing, not a lot will be lost by dropping this metric from the dataset (and the risk of creating inaccurate correlations from unknowable data is probably higher than the risk of losing useful information by removing the metric entirely).
# 
# We'll also drop the ticket and passenger ID at this point, since the values for ticket and ID don't hold much useful information about the fate of the passengers, and because the data for tickets is pretty mixed up in the set with duplicates and different data formats throughout the set.  We can also expect that if there was any useful information contained in the specific ticket or passenger ID it would have been redundant as a marker for the embarkation point, class and fare of the passenger.

# In[7]:


full_set = full_set.drop(['Cabin', 'Ticket', 'Name'], axis = 1)
full_set.isnull().sum()


# Now the only missing values are the 4 embarkation points.  Looking at the examples where the embarkations are missing, we find that the same passengers are duplicated (likely to be present in both the training and testing datasets).  It was already discovered early on that all the passengers missing embarkation points were also survivors.  Why they couldn't be tracked down for comment is something we will truly never know...

# In[8]:


miss_embark = full_set[full_set['Embarked'].isnull()]
miss_embark


# To find the most likely embarkation point for these passengers, we'll look at the common features they share and find the embarkation point of most passengers with those features.  Specifically, we'll look at the most common embarkation point for female survivors from 1st class.  This query shows that most of the passengers that fit this description embarked from Southampton, with slightly fewer embarking from Cherbourg. 

# In[9]:


embark_class = full_set[:][['Embarked', 'Pclass']].where(full_set['Survived'] == 1).where(full_set['Pclass'] == 1).where(full_set['Sex'] == 'female').groupby('Embarked').agg('count')
embark_class


# In[10]:


full_set.loc[full_set[:]['Embarked'].isnull(), 'Embarked'] = ['S','C']
full_set['Fare'] = full_set['Fare'].interpolate()
full_set.isnull().sum()


# Now that the missing data has been taken care of, we'll reformat the full set to make it as efficient as possible to train on.  Since there are several categorical variables (embarkment, sex, and class), we'll tranform them into one-hot encoding.  One-hot encoding is a better mechanism for training than simply mapping the categories to numeric values, because the magnitude of the numeric value has no meaning but will effect the behavior of the model.  

# In[11]:


from sklearn.preprocessing import LabelBinarizer
from collections import OrderedDict
import string


Emb_Coder = LabelBinarizer()
Emb_Vals = Emb_Coder.fit_transform(full_set['Embarked'])

Sex_Coder = LabelBinarizer()
Sex_Vals = Sex_Coder.fit_transform(full_set['Sex'])

PC_Coder = LabelBinarizer()
PC_Vals = PC_Coder.fit_transform(full_set['Pclass'])

data_dict = OrderedDict()
for i in range(3):
    data_dict["PC" + str(PC_Coder.classes_[i])] = PC_Vals[:, i]

for j in range(3):
    data_dict[str(Emb_Coder.classes_[j])] = Emb_Vals[:, j]
    
data_dict['gender_f/m'] = Sex_Vals[:,0]

data_dict
encodeVals = pd.DataFrame(data = data_dict)
encodeVals.head(10)


# Now the data can be aggregated back together, the categorical data can be dropped and the new full set of data can be seperated out into components to be used in training and testing the model.

# In[12]:


ind = [i for i in range(full_set.shape[0])]
full_set.insert(0, "ind_", ind)
encodeVals.insert(0, "ind_", ind)
full_set = full_set.drop(['Pclass', 'Sex', 'Embarked'], axis = 1)


# In[13]:




full = full_set.join(encodeVals, on = 'ind_', how = "left", lsuffix = 'indx')

f_n = full.shape
t_n = train_set.shape

# collect training data
train_set = full[0:t_n[0]][full.columns[0:(f_n[1])]]
y_train = full_set[0:t_n[0]]['Survived']
p_train = full_set[0:t_n[0]]['PassengerId']
train_set = train_set.drop(["Survived", "ind_indx", "ind_", "PassengerId"], axis = 1)

# collect testing data
test_set = full[t_n[0]:full.size][full.columns[0:(f_n[1])]]
y_test = full_set[t_n[0]:f_n[0]]['Survived']
p_test = full_set[t_n[0]:f_n[0]]['PassengerId']
test_set = test_set.drop(["Survived", "ind_indx", "ind_", "PassengerId"], axis = 1)

train_set.head(10)


# To start probing the performance expectations for this dataset, we'll use Tensorflow's pre-built estimator model of a DNN classifier on the training set and see how it does.  
# To train and evaluate the model,we can split it into train-dev-test sets, where the data will all be from the same distribution (except that the labels will be missing for testing data).  In this case, the training and dev set will come from the training data with recorded survival values, and the test set will be used to make final predictions once the model achieves good performance on the training and dev sets.

# In[14]:


import tensorflow as tf

full_size = train_set.shape[0]
train_size = 700
dev_size = full_size - train_size

train_in_fn = tf.estimator.inputs.pandas_input_fn(train_set[0:train_size], 
                                                  y_train[0:train_size],  
                                                  num_epochs = 10,
                                                  batch_size = 100,
                                                  shuffle = True)

feature_names = train_set.columns
feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]

classifier = tf.estimator.DNNClassifier(feature_columns = feature_columns, 
                                        hidden_units = [30,20,40,30,10])

classifier.train(input_fn = train_in_fn, steps = 10000)

classifier.evaluate(input_fn = train_in_fn, steps = 10000, name = "Train")


# Playing with the specifics of the model, (number of epochs, hidden layers, training iterations), this model achieves a modestly above-chance performance of 70%, made slightly less impressive by the fact that the distribution of survival tends towards zero. Next I'll check the how the trained model performs on the dev dataset

# In[15]:


dev_in_fn = tf.estimator.inputs.pandas_input_fn(train_set[train_size:(train_size + dev_size)], 
                                                y_train[train_size:(train_size + dev_size)], 
                                                num_epochs = 10,
                                                batch_size = 100,
                                                shuffle = True)

classifier.evaluate(input_fn = dev_in_fn, steps = 10000, name = "Dev")


# The dev set perform slightly better which makes sense assuming that the trained model is good at generalizing, since the dev set is smaller and presents fewer opportunities for errors of equal weight.  75% dev-set accuracy in not good enough yet, so we will try some different approaches after creating an initial prediction.

# In[ ]:


test_in_fn = tf.estimator.inputs.pandas_input_fn(test_set, 
                                                None,
                                                batch_size = 100,
                                                shuffle = True)
predictions = classifier.predict(input_fn = test_in_fn)
classes = []
for ind, prediction in enumerate(predictions):
    classes.append(int(prediction['class_ids'][0]))

submit_ray = pd.DataFrame(data = {"PassengerId" : p_test.values, "Survived": classes})
x = submit_ray.to_csv(index = False)

#Estimate 1
#print(x)


# The performance of the basic densely-connected Tensorflow model on the test set wasn't great, only slightly better than chance.  Since the estimates for the TensorFlow model weren't great we'll see how a keras sequential model performs on the same dataset.  Although Keras is even more general than TensorFlow (and usually runs on top of a TensorFlow environment), it does offer a little more control than the pre-baked densely connected network we trained before.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical

y_train = to_categorical(y_train)

model = Sequential()
model.add(Dense(36, activation = 'relu', input_shape = (11,)))
model.add(Dropout(0.25))
model.add(Dense(45, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(36, activation = 'relu'))
model.add(Dense(2, activation = 'sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adagrad',
              metrics=['accuracy'])

model.fit(train_set[0:train_size], y_train[0:train_size],
          epochs=1500,
          batch_size=20)


# In[ ]:


score = model.evaluate(train_set[train_size:full_size], y_train[train_size:full_size], batch_size=20)
score


# In[ ]:


prediction = model.predict(test_set, batch_size = 20)
pred_binary = []
for i in prediction:
    if i[0] > i[1]:
        pred_binary.append(0)
    else:
        pred_binary.append(1)

submit_ray = pd.DataFrame(data = {"PassengerId" : p_test.values, "Survived": pred_binary})
x = submit_ray.to_csv(index = False)

#Estimate 2
#print(x)


# Now that we've used two different deep learning approaches, it might be time to back off a little and test the performance against a more general purpose ML method.  To be honest, with so little data for training deep neural networks probably aren't the right approach to this problem, but when you like using dense network hammers, every dataset looks like a nail.
# Let's see how random forests compare to the DNN approach.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(verbose = 1) 
RFC.fit(train_set[0:train_size], y_train[0:train_size])
RFC.score(train_set[0:train_size], y_train[0:train_size])
RFC.score(train_set[train_size:], y_train[train_size:])


# In[ ]:


RFCprediction = RFC.predict(test_set)
pred = []
for i in RFCprediction:
    if i[0] > i[1]:
        pred.append(0)
    else:
        pred.append(1)
len(pred)

RFC_submit_ray = pd.DataFrame(data = {"PassengerId" : p_test.values, "Survived": pred})
x = RFC_submit_ray.to_csv(index = False)

#Estimate 3
#print(x)

