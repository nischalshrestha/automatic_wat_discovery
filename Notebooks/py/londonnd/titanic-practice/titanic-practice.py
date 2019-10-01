#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import data
test_file_path = "../input/train.csv"
train_file_path = "../input/test.csv"
test_data = pd.read_csv(test_file_path)
train_data = pd.read_csv(train_file_path)

#analyze who is likely to survive
print(train_data.columns)
print(test_data.columns)
to_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
test_data.drop(to_drop, inplace = True, axis = 1)
test_data.head()
print("Null values: " + str(test_data.isnull().sum()))

#number of survived and died (0 or 1)
surv_data = test_data['Survived'].value_counts(ascending= False)
print("Frequency Data for Number of Passengers Who Survived: ")
print(surv_data)

#number of female to male passengers who survived
passengers = test_data.groupby(['Sex', 'Survived']).size()
print("Frequency Data for Number of Passengers Who Survived: ")
print(passengers)


#number of passengers who survived by class
passengers = test_data.groupby(['Pclass', 'Survived']).size()
print("Frequency Data for Number of Passengers Who Survived by Class: ")
print(passengers)

#graphs
sns.barplot(x="Sex", y="Survived", data= test_data)
sns.barplot(x="Pclass", y="Survived", data= test_data)
plt.figure(figsize=(15,8))
sns.kdeplot(test_data["Age"][test_data.Survived == 1], color="orange", shade=True)
sns.kdeplot(test_data["Age"][test_data.Survived == 0], color="lightblue", shade=True)


# In[ ]:




