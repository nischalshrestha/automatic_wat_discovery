#!/usr/bin/env python
# coding: utf-8

# ****    What if what happened in the movie is true ? ********

# ![https://upload.wikimedia.org/wikipedia/commons/7/75/Molly_brown_rescue_award_titanic.jpg](https://upload.wikimedia.org/wikipedia/commons/7/75/Molly_brown_rescue_award_titanic.jpg)

# **I'm sorry, but I can't stop thinking about the actors in the film more than tragedy when I analyze this problem, so I have an idea that's a little crazy : if rescuers choose the people to be rescued based on whether they are single girls, married women and men. !!!!!!!!!! (sorry)  **

# **I know it's crazy, but I'll still analyze this hypothesis**

# **First of all, I look in the names to categorize people according to whether they are "Mr", "Miss" or "Mrs".
# Then I will also take into consideration the person's Age and Class of Travel.**
# 

# In[ ]:


# loading the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import neighbors


# In[ ]:


#loading of the necessary data
train = pd.read_csv("/kaggle/input/train.csv")
test = pd.read_csv("/kaggle/input/test.csv")
ytest = pd.read_csv('/kaggle/input/gender_submission.csv')
train.head()


# **let's display some information that shows the importance of sex and class in the survival of passages**

# In[ ]:


class_sex= train.groupby(['Pclass','Sex']).mean()
class_sex['Survived'].plot.bar()


# In[ ]:


#some data cleaning
train['Age'].isna().sum()
train['Pclass'].isna().sum()
Age_mean = train['Age'].mean()
most_occuring_Pclass = train['Pclass'].value_counts().idxmax()
train['Age'] = train['Age'].fillna(Age_mean)
train['Pclass'] = train['Pclass'].fillna(most_occuring_Pclass)


# **we create the matrix that groups the  "Mr", "Miss" or "Mrs"**

# In[ ]:


NewX=np.ones(len(train))
for i in range(len(train)):
    if (train['Name'][i].find("Mr.") != -1) & (train['Name'][i].find("Miss.") == -1) & (train['Name'][i].find("Mrs.") == -1):
        NewX[i]=1
    else:
        if (train['Name'][i].find("Mr.") == -1) & (train['Name'][i].find("Miss.") == -1)  & (train['Name'][i].find("Mrs.") != -1):
            NewX[i]=2
        else:
                    if (train['Name'][i].find("Mr.") == -1) & (train['Name'][i].find("Miss.") != -1)  & (train['Name'][i].find("Mrs.") == -1):
                     NewX[i]=3
                    else:
                        if (train['Name'][i].find("Mr.") == -1) & (train['Name'][i].find("Miss.") == -1)  & (train['Name'][i].find("Mrs.") == -1):
                            NewX[i]=4


# **on that we create the training matrix **

# In[ ]:


X = np.matrix([np.ones(train.shape[0]),train['Pclass'].as_matrix(),train['Age'].as_matrix(),NewX]).T
y = np.matrix(train['Survived']).T


# **Now we create the test matrix **

# In[ ]:


test['Age'].isna().sum()
test['Pclass'].isna().sum()
Age_mean = test['Age'].mean()
most_occuring_Pclass = test['Pclass'].value_counts().idxmax()

test['Age'] = test['Age'].fillna(Age_mean)
test['Pclass'] = test['Pclass'].fillna(most_occuring_Pclass)


# In[ ]:


X_test=np.ones(len(test))
for i in range(len(test)):
    if (test['Name'][i].find("Mr.") != -1) & (test['Name'][i].find("Miss.") == -1) & (test['Name'][i].find("Mrs.") == -1):
        X_test[i]=1
    else:
        if (test['Name'][i].find("Mr.") == -1) & (test['Name'][i].find("Miss.") == -1)  & (test['Name'][i].find("Mrs.") != -1):
            X_test[i]=2
        else:
                    if (test['Name'][i].find("Mr.") == -1) & (test['Name'][i].find("Miss.") != -1)  & (test['Name'][i].find("Mrs.") == -1):
                     X_test[i]=3
                    else:
                        if (test['Name'][i].find("Mr.") == -1) & (test['Name'][i].find("Miss.") == -1)  & (test['Name'][i].find("Mrs.") == -1):
                            X_test[i]=4


# In[ ]:


X_d_test = np.matrix([np.ones(test.shape[0]),test['Pclass'].as_matrix(),test['Age'].as_matrix(),X_test]).T
y_test=ytest['Survived']


# **The model we choose to use and the KNN**([https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm))

# In[ ]:


knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)


# In[ ]:


print("Score of Model (%):", knn.score(X_d_test, y_test)*100)


# **83% waw it's a very good score for my first submission on kaggle and for a crazy hypothesis  :) **
# ![data:image/jpeg;base64,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](data:image/jpeg;base64,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)

# **Precision  ?????**

# In[ ]:


from sklearn.metrics import precision_score, recall_score, f1_score
predictions = knn.predict(X_d_test)
print("Precision:", precision_score(y_test, predictions))
print("Recall:",recall_score(y_test, predictions))
print("F1 Score:",f1_score(y_test, predictions))


# **quick, quick, ... send the results quickly ;)**

# In[ ]:


submission = pd.DataFrame({"PassengerId": test['PassengerId'],"Survived":predictions })
submission.to_csv("Submission.csv", index=False)


# **Merci .................. chokran**

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




