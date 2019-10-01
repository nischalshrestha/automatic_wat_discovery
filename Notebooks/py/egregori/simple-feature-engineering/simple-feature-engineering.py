#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# This is to read the train and test file and store them in dataframe.

# In[ ]:


df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


df.columns


# Now we see, there are several columns in the data, some are numeric and some are categorical. You can read the detail in data section in the problem page.

# In[ ]:


df.columns[df.isna().any()].tolist()


# There are 3 columns that contain null value, Age, Cabin, and Embarked. Cabin and Embarked are categorical column and Age is numerical column. I will not replace the null value in  the categorical column but I will replace the null value in numerical column. There are many ways to replace the null value in numerical column, replace it with 0, its mean, its median, etc. Since the column I am trying to replace is age, it is weird to replace it 0 so median or mean is a good option, and I will go with median.

# In[ ]:


df['Age'] = df['Age'].fillna(df['Age'].median())


# In[ ]:


df['Survived'].describe()


# Now we see our output, if it is 1 then the passanger survived, 0 otherwise. We only have 891, not much, so choosing  the model with big capacity such as deep learning is a bad idea. The mean is 0.38, so it means that the data is quite balance. From this we know that we could choose method like decission tree, xgboost, svm, etc. For this example i will xgboost.

# In[ ]:


df.head(5)


# Above is the first 5 rows of the data. Well, PassangerId, Name, and Ticket might not represantive with the survival of a person so they will not be use.

# In[ ]:


def get_model(process_df, used_columns):
    train_df, validation_df = train_test_split(process_df, test_size=0.2)
    x_train = train_df[used_columns]
    y_train = train_df['Survived']
    x_validation = validation_df[used_columns]
    y_validation = validation_df['Survived']
    
    model = XGBClassifier()
    model.fit(x_train, y_train)
    print(accuracy_score(model.predict(x_validation),y_validation))
    
    return model


# For the rest of this kernel, the model that used will not be tuned. I will use the standard parameter by xgboost to prove that selecting feature is important.

# In[ ]:


used_columns = ['Pclass', 'Age', 'SibSp','Parch', 'Fare']
model = get_model(df, used_columns)


# In[ ]:


used_columns = ['Pclass', 'Age', 'SibSp','Parch']
model = get_model(df, used_columns)


# We tried using some random feature first and the accuracy is not so good.

# In[ ]:


def make_one_hot(df, columns):
    for column in columns:
        temp = pd.get_dummies(df[column], prefix=column)
        df = df.join(temp)
        
    return df


# In[ ]:


one_hoted_df = make_one_hot(df, ['Pclass'])
print(one_hoted_df.columns)
print(one_hoted_df[['Pclass', 'Pclass_1', 'Pclass_2','Pclass_3']].head(10)) 


# We change the categorical column to one hot column like above. The value of Pclass is 1, 2, and 3 so we make three new columns Pclass_1, Pclass_2, and Pclass_3. If Pclass is 1 then Pclass_1 is 1 and the rest is 0, if Pclass is 2 then Pclass_2 is 1 and the rest is 0, etc. The reason why we need to make the categorical column to one hot is because in categorical column larger value doesn't mean larger quality, 3 doesn't mean it is greater than 1 it just mean they are different.

# In[ ]:


one_hoted_df = make_one_hot(df, ['Pclass', 'Sex', 'Embarked'])
used_columns = ['Age', 'SibSp','Parch', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 
                'Embarked_C', 'Embarked_Q', 'Embarked_S']
model = get_model(one_hoted_df, used_columns)


# We make the categorical column into one hot encoding and get a very good result.. Hoooraayyy..

# In[ ]:


df['Age'].plot.hist(bins=20)


# We cannot make one hot encoding of age because it doesn't mean anything and it will be very sparse. So for age we will make it bucket.

# In[ ]:


age_group = [14, 20, 30, 35, 45, 55, 80]
previous_age_group = 0
for idx, age_group in enumerate(age_group):
    df['Age_' + str(idx)] = np.where((df['Age'] > previous_age_group) & (df['Age'] <= age_group), 1, 0)
    previous_age_group = age_group


# In[ ]:


df[['Age', 'Age_0', 'Age_1', 'Age_2', 'Age_3', 'Age_4', 'Age_5', 'Age_6']].head(10)


# * Age_0 : between 0 and 14
# * Age_1: between 14 and 20 
# * Age_2: between 20 and 30
# * Age_3: between 30 and 35 
# * Age_4: between 35 and 45 
# * Age_5: between 45 and 55
# * Age_6: between 55 and 80

# In[ ]:


one_hoted_df = make_one_hot(df, ['Pclass', 'Sex', 'Embarked'])
used_columns = ['Age_0', 'Age_1', 'Age_2', 'Age_3', 'Age_4', 'Age_5', 'Age_6', 'SibSp','Parch', 
                'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'Embarked_C', 
                'Embarked_Q', 'Embarked_S']
model = get_model(one_hoted_df, used_columns)


# Nooooo.. The accuracy get worse. Hmm maybe that because we choose the age_group too randomly, now we will try to count their corellation first.

# In[ ]:


for column in ['PassengerId', 'Pclass', 'Age', 'SibSp','Parch', 'Fare']:
    print(column, np.corrcoef(df[column], df['Survived'])[0, 1])


# The Pclass and fare have good correlation, so let's try to calculate the accuracy just based on them.

# In[ ]:


used_columns = ['Pclass', 'Fare']
model = get_model(df, used_columns)


# In[ ]:


one_hoted_df = make_one_hot(df, ['Pclass', 'Sex', 'Embarked'])
multipled_columns = ['Age_', 'Pclass_', 'Sex_', 'Embarked_']
for column in multipled_columns:
    columns = [col for col in list(one_hoted_df.columns) if col.startswith(column)]
    survived_columns = ['Survived' for _ in range(len(columns))]
    print(column, np.corrcoef(one_hoted_df[columns], one_hoted_df[survived_columns])[0, 1])


# The correlation between Pclass and survival is now increased, and we also have a strong correlation between Sex and Embarked but not for Age. So we will try to make new age bucket group that has larger correlation.

# In[ ]:


def create_age_bucket(df, new_age_bucket):
    for col in df.columns:
        if col.startswith('Age_'):
            del df[col]
    previous_age_group = 0
    for idx, age_group in enumerate(new_age_bucket):
        df['Age_' + str(idx)] = np.where((df['Age'] > previous_age_group) & (df['Age'] <= age_group), 1, 0)
        previous_age_group = age_group
    return df


# In[ ]:


def count_age_correlation(df, new_age_bucket):
    df = create_age_bucket(df, new_age_bucket)
    columns = [col for col in list(df.columns) if col.startswith('Age_')]
    survived_columns = ['Survived' for _ in range(len(columns))]
    print('Age', np.corrcoef(df[columns], df[survived_columns])[0, 1])


# In[ ]:


age_group = [15, 35, 50, 80]
count_age_correlation(df, age_group)


# We now have a better age group(supposely).

# In[ ]:


one_hoted_df = make_one_hot(df, ['Pclass', 'Sex', 'Embarked'])
used_columns = ['Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_male', 'Embarked_C', 
                'Embarked_Q', 'Embarked_S'] \
                + [col for col in list(df.columns) if col.startswith('Age_')]
model = get_model(one_hoted_df, used_columns)


# Yess we did it..

# So based on what we try, we didn't change the method or its parameter at all. By changing the input features we can have the accucary increase approximately 20%.

# In[ ]:


def run_predict(test_df, model, used_columns):
    x_test = test_df[used_columns]
    predict = model.predict(x_test)
    predict_df = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predict
    })
    
    return predict_df


# In[ ]:


new_test_df = create_age_bucket(test_df, age_group)
one_hoted_df = make_one_hot(new_test_df, ['Pclass', 'Sex', 'Embarked'])
predict_df = run_predict(one_hoted_df, model, used_columns)
predict_df.to_csv('prediction.csv', index=False)


# In[ ]:




