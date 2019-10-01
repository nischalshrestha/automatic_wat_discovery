#!/usr/bin/env python
# coding: utf-8

# <h3>Importing all packages needed:</h3>

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
plt.style.use('fivethirtyeight')
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, GridSearchCV


# <h3>Setting paths for test and train files:</h3>

# In[ ]:


TEST_PATH = "../input/test.csv"
TRAIN_PATH = "../input/train.csv"


# In[ ]:


test = pd.read_csv(TEST_PATH)
train = pd.read_csv(TRAIN_PATH)


# <h3>Let's have a look on our train and test data:</h3>

# In[ ]:


print(train.info())
print(test.info())


# <h3>First of all I'd like to drop "Survived" column and analyse the features of the merged table:</h3>

# In[ ]:


train_target = train.loc[:,"Survived"]
train = train.drop("Survived", axis=1)
model_data = train.append(test)


# In[ ]:


###DELETED GRAPHS###


# <h3>Let's have a look on "Name" feature, it has a Title in it, so we can get it from the Name</h3>
# And after we got a column "Title" with all titles, let's group titles to *`['Mr', 'Miss', 'Master', 'Pro..', 'Mrs', 'Royal']`*

# In[ ]:


model_data["Title"] = model_data['Name'].map(lambda name:name.split(",")[1].split(".")[0].strip())

titles = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Rev": "Professional",
    "Dr": "Professional",
    "Col": "Professional",
    "Major": "Professional",
    "Ms": "Mrs",
    "Mlle": "Miss",
    "Dona": "Royal",
    "Mme": "Mrs",
    "Capt": "Professional",
    "Sir": "Royal",
    "the Countess": "Royal",
    "Lady": "Miss",
    "Don": "Royal",
    "Jonkheer": "Royal"
}

model_data["Title"] = model_data["Title"].map(titles)


# <h3>Finally we have a feature to fill an "Age" feature</h3>
# We will do it later :)

# <h3>Let's look on "Cabin" feature:</h3>
# It has the most NAs but it can have a great value to our prediction model

# In[ ]:


print("Cabin feature has {} out of {} values".format(model_data.loc[model_data.Cabin.notnull(),'Cabin'].count(),model_data.PassengerId.count()))


# <h3>First, pull out first letter of the "Cabin number"<br></h3>
# Some cabin numbers have "F" letter before another letter + number, but I didn't get what does it mean so I droped those F letters :)

# In[ ]:


import re
model_data.loc[model_data.Cabin.notnull(),"Cabin_Letter"] = model_data.loc[model_data.Cabin.notnull(),"Cabin"].map(lambda letter:letter[0] if not re.match("F \w+",letter) else letter[2])
model_data.loc[model_data.Cabin.notnull(),('Cabin','Cabin_Letter')]


# <h3>Looks nice, isn't it?</h3>

# In[ ]:


model_data.loc[:,"Family_count"] = model_data["SibSp"] + model_data["Parch"]
model_data.loc[model_data.Fare.isnull(),'Fare'] = 11.10
model_data.loc[model_data.Embarked.isnull(),'Embarked'] = 'S'


# In[ ]:


f_class_letters = model_data.loc[model_data.Pclass == 1,('Cabin_Letter')].value_counts()
s_class_letters = model_data.loc[model_data.Pclass == 2,('Cabin_Letter')].value_counts()
t_class_letters = model_data.loc[model_data.Pclass == 3,('Cabin_Letter')].value_counts()

df = pd.concat([f_class_letters,s_class_letters,t_class_letters], axis=1)
df

#pd.DataFrame(["f_class_letters","s_class_letters","t_class_letters"]).plot(kind="bar")


# In[ ]:


#model_data.loc[(model_data.Pclass == 1)&(model_data.Fare == )&model_data.Age.isnull(),('Age')] = 7


# In[ ]:


model_data['Cabin_Letter_num'] = model_data['Cabin_Letter'].map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':4, 'G':5, 'T':6})

model_data.loc[model_data.Ticket.notnull(),'Ticket_clf'] = model_data.loc[model_data.Ticket.notnull(),'Ticket'].map(lambda ticket: ticket[0:2] if re.match("PC",ticket) else ticket[0])
model_data.Ticket_clf.drop_duplicates()
model_data['Ticket_clf'] = model_data['Ticket_clf'].map({'A':0, 'PC':10, 'S':11, 'P':12, 'C':13, 'W':14, 'F':15, 'L':16, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9})


# In[ ]:


label = LabelEncoder()
label.fit(model_data.Title)
model_data.Title = label.transform(model_data.Title)

label.fit(model_data.loc[:,"Sex"])
model_data.loc[:,"Sex"] = label.transform(model_data.loc[:,"Sex"])

label.fit(model_data.Embarked)
model_data.loc[:,"Embarked"] = label.transform(model_data.loc[:,"Embarked"])


# In[ ]:


model_data.loc[(model_data.Pclass == 1)&(model_data.Title == 0)&model_data.Age.isnull(),('Age')] = 7
model_data.loc[(model_data.Pclass == 1)&(model_data.Title == 1)&model_data.Age.isnull(),('Age')] = 30
model_data.loc[(model_data.Pclass == 1)&(model_data.Title == 2)&model_data.Age.isnull(),('Age')] = 41
model_data.loc[(model_data.Pclass == 1)&(model_data.Title == 3)&model_data.Age.isnull(),('Age')] = 43
model_data.loc[(model_data.Pclass == 1)&(model_data.Title == 4)&model_data.Age.isnull(),('Age')] = 51
model_data.loc[(model_data.Pclass == 1)&(model_data.Title == 5)&model_data.Age.isnull(),('Age')] = 40

model_data.loc[(model_data.Pclass == 2)&(model_data.Title == 0)&model_data.Age.isnull(),('Age')] = 3
model_data.loc[(model_data.Pclass == 2)&(model_data.Title == 1)&model_data.Age.isnull(),('Age')] = 21
model_data.loc[(model_data.Pclass == 2)&(model_data.Title == 2)&model_data.Age.isnull(),('Age')] = 32
model_data.loc[(model_data.Pclass == 2)&(model_data.Title == 3)&model_data.Age.isnull(),('Age')] = 33
model_data.loc[(model_data.Pclass == 2)&(model_data.Title == 4)&model_data.Age.isnull(),('Age')] = 41

model_data.loc[(model_data.Pclass == 3)&(model_data.Title == 0)&model_data.Age.isnull(),('Age')] = 6
model_data.loc[(model_data.Pclass == 3)&(model_data.Title == 1)&model_data.Age.isnull(),('Age')] = 17
model_data.loc[(model_data.Pclass == 3)&(model_data.Title == 2)&model_data.Age.isnull(),('Age')] = 28
model_data.loc[(model_data.Pclass == 3)&(model_data.Title == 3)&model_data.Age.isnull(),('Age')] = 32


# In[ ]:


model_data.head()


# In[ ]:


train_data = model_data[0:891]
test_data = model_data[891:]
train_data = train_data.drop(['Cabin','Cabin_Letter','Cabin_Letter_num', 'Name', 'Ticket'], axis=1)

test_data = test_data.drop(['Cabin','Cabin_Letter','Cabin_Letter_num', 'Name', 'Ticket'], axis=1)
test_data.head()


# In[ ]:


clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(train_data, train_target)


# In[ ]:


features = pd.DataFrame()
features['feature'] = train_data.columns
features['importance'] = clf.feature_importances_
features


# In[ ]:


forest = RandomForestClassifier(max_features='sqrt')

parameter_grid = {
                 'max_depth' : [4,5,6,7,8],
                 'n_estimators': [200,210,240,250],
                 'criterion': ['gini','entropy']
                 }
cross_validation = StratifiedKFold(5)

grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(train_data, train_target)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))


# In[ ]:


output = grid_search.predict(test_data).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = test_data['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)

