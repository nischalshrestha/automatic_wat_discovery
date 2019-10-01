#!/usr/bin/env python
# coding: utf-8

# In[74]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Loading data
# 

# In[75]:


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


# # View data

# In[76]:


train_data.head(10)


# In[77]:


train_data.info()


# In[78]:


train_data.describe()


# In[79]:


list(test_data.columns.values)


# In[80]:


# Describe column with string value
train_data.describe(include=['O'])


# In[81]:


print('total data length', len(train_data))
print('survived', len(train_data[train_data["Survived"] == 1]))
print('survived female', len(train_data[(train_data["Survived"] == 1) & (train_data["Sex"] == "female")]))

print("%.2f" % (100 * np.mean(train_data["Survived"][train_data["Sex"] == "female"])), "% female survied")
print("%.2f" % (100 * np.mean(train_data["Survived"][train_data["Sex"] == "male"])), "% male survied")

print("%.2f" % (100 * np.mean(train_data["Survived"][train_data["Age"] >= 18])), "% adult survied")
print("%.2f" % (100 * np.mean(train_data["Survived"][train_data["Age"] < 18])), "% teenage survied")

print("%.2f" % (100 * np.mean(train_data["Survived"][train_data["Pclass"] == 3])), "% poor people survied")
print("%.2f" % (100 * np.mean(train_data["Survived"][train_data["Pclass"] == 2])), "% middle-class people survied")
print("%.2f" % (100 * np.mean(train_data["Survived"][train_data["Pclass"] == 1])), "% rich people survied")

print("%.2f" % (100 * np.mean(train_data["Survived"][(train_data["Pclass"] == 1) & (train_data["Sex"] == "female")])), "% rich women survied")


# In[82]:


train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending = False)


# # Data pre-processing

# In[83]:


combain_data = [train_data, test_data]
for data in combain_data: 
    data["Sex"] = data["Sex"].apply(lambda x: 1 if x == "male" else 0)
    data["Age"] = data["Age"].fillna(np.mean(data["Age"]))
    data["Fare"] = data["Fare"].fillna(np.mean(data["Fare"]))


# In[84]:


test_data_passengerId = test_data["PassengerId"]
test_data = test_data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
train_data = train_data[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]


# In[85]:


X = train_data.drop("Survived", axis = 1)
y = train_data["Survived"]


# # Predict

# In[86]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 21)
classifier = DecisionTreeClassifier(max_depth = 3)
classifier.fit(X_train, y_train)


# # Evaluate

# In[87]:


from sklearn.metrics import accuracy_score
print("Train accuracy", accuracy_score(y_train, classifier.predict(X_train)))
print("Validation accuracy", accuracy_score(y_test, classifier.predict(X_test)))


# ### Draw a map for reviewing

# In[88]:


from sklearn import tree
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

with open("tree1.dot", "w") as f:
    f = tree.export_graphviz(classifier, out_file = f, feature_names = X_test.columns, class_names = ["No", "Yes"], rounded = True, filled = True)
    
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
img.save('sample-out.png')
PImage("sample-out.png")


# In[ ]:


submission = pd.DataFrame({
    "PassengerId": test_data_passengerId,
    "Survived": classifier.predict(test_data)
})
submission
submission.to_csv("submission.csv", index=False)


# In[ ]:




