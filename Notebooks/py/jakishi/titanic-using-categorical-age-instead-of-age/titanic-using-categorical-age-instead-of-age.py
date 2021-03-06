#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


def kesson_table(df): 
        null_val = df.isnull().sum()
        percent = 100 * df.isnull().sum()/len(df)
        kesson_table = pd.concat([null_val, percent], axis=1)
        kesson_table_ren_columns = kesson_table.rename(
        columns = {0 : '欠損数', 1 : '%'})
        return kesson_table_ren_columns


# In[ ]:


kesson_table(train)


# In[ ]:


kesson_table(test)


# In[ ]:


train["Age"] = train["Age"].fillna(train["Age"].median())
# Mapping Age
train["Age"][train["Age"] <= 16 ] = 0
train["Age"][(train["Age"] > 16) & (train["Age"] <= 32) ] = 1
train["Age"][(train["Age"] > 32) & (train["Age"] <= 48) ] = 2
train["Age"][(train["Age"] > 48) & (train["Age"] <= 64) ] = 3
train["Age"][ train["Age"] > 64 ] = 4
train["Age"] = train['Age'].astype(int)
    
train["Embarked"] = train["Embarked"].fillna("S")


# In[ ]:


kesson_table(train)


# In[ ]:


train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"][train["Embarked"] == "S" ] = 0
train["Embarked"][train["Embarked"] == "C" ] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2


# In[ ]:


train.head(10)


# In[ ]:


test["Age"] = test["Age"].fillna(test["Age"].median())
# Mapping Age
test["Age"][test["Age"] <= 16 ] = 0
test["Age"][(test["Age"] > 16) & (test["Age"] <= 32) ] = 1
test["Age"][(test["Age"] > 32) & (test["Age"] <= 48) ] = 2
test["Age"][(test["Age"] > 48) & (test["Age"] <= 64) ] = 3
test["Age"][ test["Age"] > 64 ] = 4
test["Age"] = test["Age"].astype(int)

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test.Fare[152] = test.Fare.median()
 
test.head(10)


# In[ ]:


from sklearn import tree


# In[ ]:


# 「train」の目的変数と説明変数の値を取得
target = train["Survived"].values
feature_names_one = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]
features_one = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].values
 
# 決定木の作成とアーギュメントの設定
max_depth = 3
min_samples_split = 2
my_tree_one = tree.DecisionTreeClassifier(criterion="entropy", max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
my_tree_one = my_tree_one.fit(features_one, target)

# 「test」の説明変数の値を取得
test_features = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].values

# 「test」の説明変数を使って「my_tree_one」のモデルで予測
my_prediction = my_tree_one.predict(test_features)


# In[ ]:


my_prediction.shape


# In[ ]:


print(my_prediction)


# In[ ]:


# PassengerIdを取得
PassengerId = np.array(test["PassengerId"]).astype(int)
 
# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
 
# my_tree_one.csvとして書き出し
my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])


# In[ ]:


#ROCとAUC


# In[ ]:


# 決定木構造の可視化
import graphviz
from IPython.display import Image
from graphviz import Digraph
from sklearn.externals.six import StringIO

from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

# Export our trained model as a .dot file
with open("tree1.dot", 'w') as f:
  f = tree.export_graphviz(my_tree_one, out_file=f,feature_names=feature_names_one, max_depth=10)

#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree_categorical_age.png'])

# Annotating chart with PIL
img = Image.open("tree_categorical_age.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
img.save('sample-out.png')
PImage("sample-out.png")

