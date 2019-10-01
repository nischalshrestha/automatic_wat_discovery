#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont


# In[ ]:


get_ipython().magic(u'matplotlib inline')
sns.set()


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


df_train.info()


# In[ ]:


df_train['Survived'].value_counts()


# In[ ]:


1-(342/891)**2-(549/891)**2


# In[ ]:


0.382*(314/891) + 0.306*(577/891)


# In[ ]:


# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])


# In[ ]:


# fill in missing numerical values with medians
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())


# In[ ]:


# convert sex categories to numerical indicators
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
data.head()


# In[ ]:


# select columns and view head
data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]
data.head()


# In[ ]:


# split back to train-test
data_train = data.iloc[:891]
data_test = data.iloc[891:]


# In[ ]:


x = data_train.values
test = data_test.values
y = survived_train.values


# In[ ]:


# instantiate model and fit to data
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(x, y)


# In[ ]:


# export our trained model as a .dot file
with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(clf,
                              out_file=f,
                              max_depth = 3,
                              impurity = True,
                              feature_names = list(data_train),
                              class_names = ['Died', 'Survived'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
draw.text((10, 0), # Drawing offset (position)
          'Decision Tree', # Text to draw
          (0,0,255), # RGB desired color
          font=font) # ImageFont object with desired font
img.save('sample-out.png')
PImage("sample-out.png")


# In[ ]:


# make predictions and store in 'Survived' column of df_test
y_pred = clf.predict(test)
df_test['Survived'] = y_pred


# In[ ]:


# generate output file
df_test[['PassengerId', 'Survived']].to_csv('predictions.csv', index=False)

