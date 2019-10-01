#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import tensorflow as tf
import sklearn as skl
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import os


# In[ ]:


path = os.listdir("../input")
data_train = pd.read_csv("../input/train.csv", sep=",")
data_test = pd.read_csv("../input/test.csv", sep=",")
data_test_y = pd.read_csv("../input/gender_submission.csv", sep=",")


# In[ ]:


title_dict = {
    "Miss": 0,
    'Ms': 0,
    'Lady': 0,
    'Mme': 1,
    "Mrs": 1,
    "Mr": 2,
    'Don': 2,
    'Rev': 2,
    'Dr': 3,
    "Master": 3,
    'Major': 3,
    'Sir': 3,
    'Col': 3,
    'Countess': 3,
    'Jonkheer': 3,
    'Dona': 3,
    "Mlle": 3,
    "Capt": 3
}
Embarked_dict = {
    "S": 0,
    "C": 1,
    "Q": 2
}

desk_dict = {
    "T": 1,
    "G": 2,
    "F": 3,
    "E": 4,
    "D": 5,
    "C": 6,
    "B": 7,
    "A": 8
}

data_train["Title"] = data_train["Name"].str.extract("([A-Za-z]+)\.",expand=True)
data_train["Title"] = data_train["Title"].apply(lambda x: title_dict[x])
data_train["Had Cabin"] = data_train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
data_train["Embarked Num"] = data_train["Embarked"].apply(lambda x: 4 if type(x) == float else Embarked_dict[x])
data_train["Family Size"] = data_train["SibSp"] + data_train["Parch"] + 1
data_train["Alone"] = [1 if x == 1 else 0 for x in data_train["Family Size"].values]
data_train['Fare Per Person'] = data_train['Fare']/(data_train['Family Size'])
data_train["Desk"] = data_train["Cabin"].str.extract("([A-Za-z]+)",expand=True)
data_train["Desk"] = data_train["Desk"].apply(lambda x: 0 if type(x) == float else desk_dict[x])
data_train["Sex"] = data_train["Sex"].replace("male", 1)
data_train["Sex"] = data_train["Sex"].replace("female", 0)

data_test["Title"] = data_test["Name"].str.extract("([A-Za-z]+)\.",expand=True)
data_test["Title"] = data_test["Title"].apply(lambda x: title_dict[x])
data_test["Had Cabin"] = data_test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
data_test["Embarked Num"] = data_test["Embarked"].apply(lambda x: 4 if type(x) == float else Embarked_dict[x])
data_test["Family Size"] = data_test["SibSp"] + data_test["Parch"] + 1
data_test["Alone"] = [1 if x == 1 else 0 for x in data_test["Family Size"].values]
data_test['Fare Per Person'] = data_test['Fare']/(data_test['Family Size'])
data_test["Desk"] = data_test["Cabin"].str.extract("([A-Za-z]+)",expand=True)
data_test["Desk"] = data_test["Desk"].apply(lambda x: 0 if type(x) == float else desk_dict[x])
data_test["Sex"] = data_test["Sex"].replace("male", 1)
data_test["Sex"] = data_test["Sex"].replace("female", 0)

data = pd.concat([data_train, data_test], sort=True)

data.head(10)


# In[ ]:


desk_dict = {
    "T": 1,
    "G": 2,
    "F": 3,
    "E": 4,
    "D": 5,
    "C": 6,
    "B": 7,
    "A": 8
}

data["Desk"].unique()


# In[ ]:


cond_class_1 = data["Pclass"] == 1 
cond_class_2 = data["Pclass"] == 2
cond_class_3 = data["Pclass"] == 3

cond_f = data["Sex"] == 0
cond_m = data["Sex"] == 1

cond_d = data["Survived"] == 0
cond_a = data["Survived"] == 1

N = 3
ind = np.arange(N)   
width = 0.5 

dead_f = (
    len(data[cond_class_1 & cond_f & cond_d]),
    len(data[cond_class_2 & cond_f & cond_d]),
    len(data[cond_class_3 & cond_f & cond_d])
)

alive_f = (
    len(data[cond_class_1 & cond_f & cond_a]),
    len(data[cond_class_2 & cond_f & cond_a]),
    len(data[cond_class_3 & cond_f & cond_a])
)

dead_m = (
    len(data[cond_class_1 & cond_m & cond_d]),
    len(data[cond_class_2 & cond_m & cond_d]),
    len(data[cond_class_3 & cond_m & cond_d])
)

alive_m = (
    len(data[cond_class_1 & cond_m & cond_a]),
    len(data[cond_class_2 & cond_m & cond_a]),
    len(data[cond_class_3 & cond_m & cond_a])
)

fig, axes = plt.subplots(1, 2,figsize=(20,7))

p1_f = axes[0].bar(ind, dead_f, width, color='#d62728')
p2_f = axes[0].bar(ind, alive_f, width, bottom=dead_f)
axes[0].legend((p1_f[0], p2_f[0]), ('Dead', 'Alive'))
axes[0].set_ylabel('Number of people')
axes[0].set_title('Dead/alive stats for female groups')
axes[0].set_yticks(range(0, 500, 50))
axes[0].set_xticks(ind)
axes[0].set_xticklabels(("First class", "Second class", "Third class"))


p1_m = axes[1].bar(ind, dead_m, width, color='#d62728')
p2_m = axes[1].bar(ind, alive_m, width, bottom=dead_m)
axes[1].legend((p1_f[0], p2_f[0]), ('Dead', 'Alive'))
axes[1].set_ylabel('Number of people')
axes[1].set_title('Dead/alive stats for male groups')
axes[1].set_yticks(range(0, 500, 50))
axes[1].set_xticks(ind)
axes[1].set_xticklabels(('First class', 'Second class', 'Third class'))
plt.show()


# In[ ]:


plt.figure(figsize=(20,10))
plt.xticks(range(0, 90, 3))
plt.yticks(range(0, 150, 5))
cond_notnull = data["Age"].notnull()

ages_all = [
    data[cond_notnull & cond_f]["Age"],
    data[cond_notnull & cond_m]["Age"]
]
                    
plt.hist(ages_all, bins=range(0, 90, 3))
plt.legend(("Female", "Male"))

plt.show()


# In[ ]:


corrs = [
    data["Survived"].corr(data["Pclass"]),
    data["Survived"].corr(data["Sex"]),
    data["Survived"].corr(data["Age"]),
    data["Survived"].corr(data["SibSp"]),
    data["Survived"].corr(data["Parch"]),
    data["Survived"].corr(data["Alone"]),
    data["Survived"].corr(data["Family Size"]),
    data["Survived"].corr(data["Title"]),
    data["Survived"].corr(data["Fare"]),
    data["Survived"].corr(data["Had Cabin"]),
    data["Survived"].corr(data["Embarked Num"]),
    data["Survived"].corr(data["Fare Per Person"]),
    data["Survived"].corr(data["Desk"])


]

corrs_l = [
    "Class",
    "Sex",
    "Age",
    'Siblings',
    "Parch",
    "Alone",
    "Family Size",
    "Title",
    "Fare",
    "Had Cabin",
    "Embarked",
    "Fare Per Person",
    "Desk"
]

plt.figure(figsize=(20, 10))
plt.plot(corrs_l, corrs)
plt.plot(corrs_l, np.zeros([len(corrs_l)]))
plt.show()


# In[ ]:


x = tf.placeholder(dtype=tf.float32, shape=[None, 12])
y = tf.placeholder(dtype=tf.float32, shape=[None, 2])

w_1 = tf.Variable(tf.truncated_normal((12, 64)), dtype=tf.float32)

w_2 = tf.Variable(tf.truncated_normal((64, 2)), dtype=tf.float32)

l_1 = tf.matmul(tf.nn.sigmoid(x), w_1)

l_1_r = tf.nn.relu(l_1)

l_2 = tf.matmul(l_1_r, w_2)

loss = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=l_2, labels=y)))

opt = tf.train.AdamOptimizer(0.02).minimize(loss)


# In[ ]:


epoch = 10000

sess = tf.Session()
sess.run(tf.global_variables_initializer())

features = ["Pclass", "Sex", "SibSp", "Parch", "Alone", "Family Size", "Title", "Fare", "Had Cabin", "Embarked Num", "Fare Per Person", "Desk"]

for i in range(epoch):
    train_feed = {
        x: data_train[features].values,
        y: np.array([[1,0] if i == 0 else [0,1] for i in data_train["Survived"].values])
    }
    
    _ = sess.run([opt], feed_dict=train_feed)
    
    
    if(i % 250 == 0):
        pred = np.argmax(sess.run([l_2], feed_dict=train_feed)[0], axis=1)
        print("Acc: ", np.sum(np.equal(data_train["Survived"].values , pred)) / len(pred))
    
    
    


# In[ ]:


pred_train = np.argmax(sess.run([l_2], feed_dict={x: data_train[features]})[0], axis=1)
print("Acc for train data: ", np.sum(np.equal(data_train["Survived"].values , pred_train)) / len(pred_train))

pred_test = np.argmax(sess.run([l_2], feed_dict={x: data_test[features]})[0], axis=1)
print("Acc for test data: ", np.sum(np.equal(data_test_y["Survived"].values , pred_test)) / len(pred_test))


# In[ ]:


submission = pd.DataFrame({
    "PassengerId" : data_test["PassengerId"].values,
    "Survived": np.argmax(sess.run([l_2], feed_dict={x: data_test[features]})[0], axis=1),
})

submission.to_csv("titanic.csv", index=False)


# In[ ]:




