#!/usr/bin/env python
# coding: utf-8

# This is my beginner's approach after a 30-hour course in ML. The approaches are partly developed by myself and partly adopted from other kernels. I try to mark ideas taken over from others as far as possible.
# 
# First, we'll import some useful plugins. For didactic reasons, I load most plugins first in the section when they are also needed.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# I now load the test and training files and merge them into a data set. I do this so that I don't have to apply all steps twice and also have a larger database to calculate missing values. At the end the data will be separated again. 

# In[ ]:


df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

#Merging the Data Sets
data_df = df.append(df_test, sort=False) 
data_df

data_df.head()


# **1. Overview of the data**
# 
# **1.1. Check for missing values**
# 
# Already in the first line you can see missing values for 'Cabin', which is why we first check if there are more missing values. Most models cannot handle missing values and too many missing values can distort the measurement. So let's check this. 

# In[ ]:


print(data_df.columns[df.isnull().any()])

print(data_df[data_df["Age"].isnull() == True].count())
print(data_df[data_df["Cabin"].isnull() == True].count())
print(data_df[data_df["Embarked"].isnull() == True].count())


# We see that values are missing for three categories. Since there are a lot of missing values for the category 'Cabin', let's first look at the other two. 
# 
# Next, we fill the missing values of these columns. For simplicity's sake we use the median for the metric variables . For Embarked we use the most common value, which in this case is "S". My tests have shown that more complicated methods to fill in the age do not necessarily lead to a better score.

# In[ ]:


# Fill NaN
median_value=data_df['Age'].median()
data_df['Age']=data_df['Age'].fillna(median_value)
data_df['Age']=data_df["Age"].astype("int")

median_value_fare=data_df['Fare'].median()
data_df['Fare']=data_df['Fare'].fillna(median_value_fare)

data_df['Embarked']=data_df['Embarked'].fillna("S")


# **1.2 Let's take a look at some plots**
# 
# To get a rough overview of the data distribution we can first have a look at the 'Pairplot' for the single variables in relation to 'Survived'. But before we can look at the data we must replace the encoding with letters with a numeric encoding.

# In[ ]:


#Encoding Sex & Embarked
data_df["Sex"] = data_df["Sex"].replace("male", 0).replace("female", 1)
data_df["Embarked"]=data_df["Embarked"].replace("C", 0).replace("S", 1).replace("Q",2)

data_df.head()


# In[ ]:


#Pairplot
sns.pairplot(data_df, hue="Survived")


# Now we can take a closer look at different variables to find out whether they differ in their probability of survival.

# In[ ]:


data_df.groupby(['Pclass'])['Survived'].mean()


# In[ ]:


data_df.groupby(['Sex'])['Survived'].mean()


# In[ ]:


data_df.groupby(['Embarked'])['Survived'].mean()


# ***Explanation***
# 
# As you can see these three variables all have a distinct difference in relation to 'Survival'. The explanation for the differences by gender is simplest: "women and children first". 
# The differences between the classes can be explained by the location of each class on board, as they were at different distances from the lifeboats. 
# The influence of the port could be explained by the fact that an above-average number of people from the port coded "0" have booked a higher class onboard. If you like you can check this within the data.

# Let's look at the families. As you can see below, family size also has an impact on the likelihood of survival. We keep this in mind for further data preparation.

# In[ ]:


data_df.groupby(['SibSp'])['Survived'].mean()


# In[ ]:


data_df.groupby(['Parch'])['Survived'].mean()


# Finally, we should check 'Fare' and 'Age'. But since these are metrically scaled data, we should divide them into classes first in order to look at them better. Therefore we divide the values into equally large groups and encode the values. For 'Age' I use an additional class for children (up to 16) and one for seniors (from 65).

# In[ ]:


#Make Groups
print(pd.qcut(data_df["Fare"], 4).unique())
print(pd.qcut(data_df["Age"], 4).unique())


# In[ ]:


data_df.loc[data_df["Fare"] <= 7.896, 'Fare_Grouped'] = 0
data_df.loc[(data_df["Fare"] > 7.896) & (data_df["Fare"] <= 14.454),  'Fare_Grouped'] = 1
data_df.loc[(data_df["Fare"] > 14.454) & (data_df["Fare"] <= 31.275),  'Fare_Grouped'] = 2
data_df.loc[(data_df["Fare"] > 31.275),  'Fare_Grouped'] = 3
data_df['Fare_Grouped']=data_df["Fare_Grouped"].astype("int")

data_df.loc[data_df["Age"] <= 16, 'Age_Grouped'] = 0
data_df.loc[(data_df["Age"] > 16) & (data_df["Age"] <= 28),  'Age_Grouped'] = 1
data_df.loc[(data_df["Age"] > 28) & (data_df["Age"] <= 35),  'Age_Grouped'] = 2
data_df.loc[(data_df["Age"] > 35) & (data_df["Age"] <= 64),  'Age_Grouped'] = 3
data_df.loc[(data_df["Age"] > 64),  'Age_Grouped'] = 4
data_df["Age_Grouped"] = data_df.Age.astype(int)

data_df.head()


# In[ ]:


data_df.groupby(['Fare'])['Survived'].mean()


# In[ ]:


data_df.groupby(['Age'])['Survived'].mean()


# **2. Engineer New Variables**
# 
# **2.1 Family Size**
# 
# Let's have a look at the family variables from a few steps earlier. We have found that both the number of siblings/partners and the number of parents/children have an influence. So we generate a new variable from it that expresses the family size

# In[ ]:


#Family Variable
data_df["Family"] = data_df["SibSp"] + data_df["Parch"] +1 
df = data_df.drop(["SibSp", "Parch"], axis=1)
data_df.head()


# **2.2 Cabin**
# 
# One variable that we have not yet considered is the cabin. Here, besides the many missing values, it is noticeable that the name is composed of a letter indicating the deck and a number indicating the room. 
# 
# In theory, the deck can increase the probability of survival, as this includes the distance to the lifeboats. We've already seen this in the 'Class' attribute. The room cannot be assigned directly. In general it can be said that the variable is very experimental and needs further research. For example, it would be possible to predict the missing values using the ticket number, the fare and the surname. 
# For the beginning of the missing cabins we assign an own deck and the mean value (50).
# 
# For the existing values we split the column into the deck and the room number. Then we create dummy variables so that the models can work with them better. You can also just encode the deck as we do it with the other variables.

# In[ ]:


#Deck
data_df["Deck"] = data_df["Cabin"].str.slice(0,1)
data_df["Deck"] = data_df["Deck"].fillna("N")

#Room
data_df["Room"] = data_df["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
data_df["Room"] = data_df["Room"].fillna(data_df["Room"].mean())
data_df["Room"] = data_df.Room.astype(int)

#Create Dummies
data_df = pd.get_dummies(data_df, columns = ["Deck"])

data_df.head()


# **2.3 Tickets**
# 
# If the 'Ticket' column is examined randomly, some details will be noticed. Some tickets have a letter in front of them and others consist only of numbers. Furthermore, the tickets have a different length. Let's take a closer look at this.  (Based on https://www.kaggle.com/zlatankr/titanic-random-forest-82-78)
# 
# First we take the first character from the ticket number and in the next step we also determine the ticket length and look at the connection with the variable 'Survive'.
# 

# In[ ]:


#Ticket Length

data_df['Ticket_Len'] = data_df['Ticket'].apply(lambda x: len(x))
data_df.groupby(['Ticket_Len'])['Survived'].mean()


# In[ ]:


#Ticket First Letter

data_df['Ticket_Lett'] = data_df['Ticket'].apply(lambda x: str(x)[0])
data_df.groupby(['Ticket_Lett'])['Survived'].mean()


# So there is a different distribution between the different ticket letters as well as the length of the ticket number. Therefore, we use both variables in the model. 
# 
# I'm forming two groups to split the ticket letters. However, you can also test with a different classification. 
# 

# In[ ]:


#Ticket Letter Encoding
replacement = {
    'A': 0,
    'P': 1,
    'S': 0,
    '1': 1,
    '2': 0,
    'C': 0,
    '7': 0,
    'W': 0,
    '4': 0,
    'F': 1,
    'L': 0,
    '9': 1,
    '6': 0,
    '5': 0,
    '8': 0,   
    '3': 0,
}

data_df['Ticket_Lett'] = data_df['Ticket_Lett'].apply(lambda x: replacement.get(x))

data_df.head()


# **2.4 Name**
# 
# The only column we haven't looked at yet is 'Name'. So what can we do with it?
# 
# A first approach would be to determine the length here as well. Possibly important persons, who should be rescued at all costs, have a longer name with several first names or titles. In addition, there is not much that can be done with float values at first.

# In[ ]:


#Name Len
data_df["Name_Len"] = data_df['Name'].apply(lambda x: len(x))
data_df.groupby(['Name_Len'])['Survived'].mean()


# We see that longer names had significantly more chances of survival. Therefore we keep the variable in our model for the time being. 
# 
# Next, if we are dealing with important people, we could also examine the title of the person. We extract it into a separate column and encode it accordingly. When coding, you can test your own approaches. This classification is the best one for my model.

# In[ ]:


#Extract Title from Name
for name_string in data_df['Name']:
    data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.')


#Coding of Titles
replacement = {
    'Dr': 0,
    'Master': 1,
    'Miss': 2,
    'Ms': 2,
    'Mlle': 2,
    'Rev': 3,
    'Mme': 4,
    'Mrs': 4,
    'Lady': 4,
    'the Countess': 5,
    'Don': 5,
    'Dona': 5,
    'Jonkheer': 5,
    'Sir': 5,
    'Capt': 5,
    'Col': 5,
    'Major': 5,
    'Mr': 5,   
}

data_df['Title'] = data_df['Title'].apply(lambda x: replacement.get(x))

#Fill Missing Values
median_value=data_df['Title'].median()
data_df['Title']=data_df['Title'].fillna(median_value)

data_df["Title"] = data_df.Title.astype(int)

data_df.head()


# **Additional: 2.5 Family Survival (Based on S.Xu's kernel)**
# 
# This idea originally comes from "S.Xu" and was slightly changed by "konstantinmasich". I took the code from him.
# 
# In principle, the aim is to use the surname and the ticket price to group families and calculate the probability of survival for the family on the basis of this. 
# A similar approach could be used, for example, to predict the cabins or at least the deck using the ticket number and surname.
# 

# In[ ]:


#Family Group (from S.Xu's kernel)

data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ",")[0])

DEFAULT_SURVIVAL_VALUE = 0.5
data_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in data_df[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0

for _, grp_df in data_df.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    data_df.loc[data_df['PassengerId'] == passID, 'Family_Survival'] = 0
                        
print("Number of passenger with family/group survival information: " 
      +str(data_df[data_df['Family_Survival']!=0.5].shape[0]))


# **3. Train the Classifier**
# 
# First we will split the dataframe into the original training and test set again.
# 
# After that, we will train a RFC (Random Forest Classifier). For this we use 'GridSearchCV' to find the best parameters ('Hyperparameter Tuning'). 

# In[ ]:


data_df.head()


# In[ ]:


#Drop unused columns
data_df = data_df.drop(["Ticket", "Last_Name", "Name", "Cabin", "Age", "Fare"], axis=1)


# Split in TRAIN_DF and TEST_DF:
df = data_df[:891]
df_test= data_df[891:]


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

forrest_params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 40, 5)],
)

x = df.drop(["Survived", "PassengerId"], axis=1)
y = df["Survived"]

forrest = RandomForestClassifier()
forest_cv = GridSearchCV(estimator=forrest, param_grid=forrest_params, cv=5) 
forest_cv.fit(x, y)
print(forest_cv.best_score_)
print(forest_cv.best_estimator_)


# In[ ]:


testdat = df_test.drop(["Survived", "PassengerId"], axis=1)

predict = forest_cv.predict(testdat)

print(predict.astype(int))

submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": predict.astype(int)
})

submission.to_csv("submision.csv", index=False)

