#!/usr/bin/env python
# coding: utf-8

# # Titanic with DNN in Tensorflow
# ### **André Koeppl**
# ### January 24th, 2018
# 
# I got inspiration for this kernel from the folowing kernels:
# * Yassine Ghouzam, PhD: "Titanic Top 4% with ensemble modeling
# * LD Freeman, from his kernel: "A Data Science Framework: To Achieve 99% Accuracy". 
# 
# I initiated in kaggle with my first kernel:ssss
# 
# Then I learnt a little bit about tensorflow and I'd like to try it with the Titanic competition.
# You can check the results for this case are no better than the best kernels/kagglers, but my objective was to implement this code using tensorflow and sharing this powerful tool with begginers as me.
# 
# ### Summary:
# * 1-Importing libraries
# * 2-Load and check data
# * 3- Filling missing values
# *     3.1- Checking the missing values
# *     3.2- Fare missing values
# *     3.3- Embarked missing values
# *     3.4- Cabin missing values
# *     3.5- Age missing values
# * 4- Feature engineering
# *     4.1- Engineering the variable 'Name'
# *     4.2- Engineering the variable 'Ticket'
# *     4.3- Engineering the variable 'Family size'
# * 5- Dataset preparation for Model
# *     5.1- One hot encoding the categorical variables
# *     5.2- Separation of the train test dataframes
# *     5.3- Creating feature list for tensorflow
# *     5.4- Scaling the numerical variables
# *     5.5- Bucketizing the variable 'Age'
# *     5.6- Bucketizing the variable 'Fare'
# *     5.7- Train_test_split
# * 6- Models
# *     6.1- Quick start with a linear model
# *     6.2- Dense Neural Networks (DNNs)
# * 7- Prediction
# *     7.1 Predict and Submit results
#     
# Good lecture!    
#     
#     
#     

# ## 1. Importing libraries
# 

# In[ ]:


#Importing initial libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', context='notebook', palette='deep')
get_ipython().magic(u'matplotlib inline')

import tensorflow as tf
print('-'*25)
print(tf.__version__)
sess=tf.InteractiveSession()

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#Since we will be running a lot of different models, it's important to rest  the tensorflow 
#graph between runs so that the various parts aren't erroneously linked together.
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()



# ## 2. Load and check data

# In[ ]:


# Load data
##### Load train and Test set
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
#Saving IDtest for the final result submission
IDtest = df_test["PassengerId"]
df_train.head()


# In[ ]:


print(df_train.shape)
print(df_test.shape)


# In[ ]:


## Join train and test datasets in order to obtain the same number of features during categorical conversion
train_len = len(df_train)
dataset =  pd.concat(objs=[df_train, df_test], axis=0).reset_index(drop=True)


# In[ ]:


dataset.head()


# ## 3. Filling Missing values

# ### 3.1 Checking the missing values

# In[ ]:


#Checking missing values
print(dataset.isnull().sum())


# ### 3.2 Fare missing values

# In[ ]:


Fare_median = df_train.Fare.median()
dataset.loc[dataset.Fare.isnull(),'Fare'] = Fare_median


# ### 3.3 Embarked missing values

# In[ ]:


g = sns.factorplot("Embarked",  data=dataset,size=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")


# Since S is the most frequent Embarkation port, let's assume the missing values to be S

# In[ ]:


dataset.loc[dataset.Embarked.isnull(),'Embarked'] = 'S'


# ### 3.4 Cabin missing values

# Let's take a look  at the NaNs of the categorical variable Cabin:

# In[ ]:


print("% of NaN values in Cabin Variable for dataset:")
print(dataset.Cabin.isnull().sum()/len(dataset))


# Very representative.

# In[ ]:


# Replace the Cabin number by the type of cabin 'X' if not
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])


g = sns.factorplot("Cabin",  data=dataset,size=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")


# ### 3.5 Age missing values

# In[ ]:


## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = df_train["Age"].median()
    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & 
                               (dataset['Parch'] == dataset.iloc[i]["Parch"]) &
                               (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        dataset['Age'].iloc[i] = age_pred
    else :
        dataset['Age'].iloc[i] = age_med

#Checking missing values
print(dataset.isnull().sum())


# # 4. Feature engineering

# ### 4.1 Engineering the variable 'Name'

# In[ ]:


title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(title)


j = sns.countplot(x="Title",data=dataset)
j = plt.setp(j.get_xticklabels(), rotation=45) 


# In[ ]:


# Exploring Survival probability
g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar", size = 10 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# Captain has not survived. The major and colonel are military titles and have ~50% chance of survival (with high dipsersion). It seems relevant to group possible dynamics of profession, crew membership, nobility title, and so on..

# In[ ]:


# Let's group the military and crew titles ensemble
dataset["Title"] = dataset["Title"].replace(['Capt', 'Col','Major'], 'Crew/military')
# Let's group the profession title ensemble
dataset["Title"] = dataset["Title"].replace(['Master', 'Dr'], 'Prof')
# Let's change the type of women title
dataset["Title"] = dataset["Title"].replace(['Miss', 'Mme','Mrs','Ms','Mlle'], 'Woman')
# Let's change the type of nobility title
dataset["Title"] = dataset["Title"].replace(['the Countess', 'Sir', 'Lady', 'Don','Jonkheer', 'Dona'], 'Noble')
# Let's change the type of religious title
dataset["Title"] = dataset["Title"].replace(['Rev'], 'Religious')

j = sns.countplot(x="Title",data=dataset)
j = plt.setp(j.get_xticklabels(), rotation=45) 


# In[ ]:


# Exploring Survival probability
g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# Clearly we understand that they prioitize Woman, followed by the Nobles and then Professional titles. We can't make a clear conclusion regarding nobility titles because of the high sigma, but I bet it's because of the percentage of men in this group. Obviously Religious title make the moral decision in this case. Regarding officers, the dispersion indicates part of the crew decided to remain onboard and others went on the survival boats. Ordinary men had the least chance of survival.

# ### 4.2 Engineering the variable 'Ticket'

# In[ ]:


dataset.Ticket.head(10)


# Let's extract the first letter of the ticket, which may be indicative of group reservations, and whenever it's not possible to find one letter, input 'X' value, as I've done with the 'Cabin' missing treatment.

# In[ ]:


## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 

Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
        
dataset["Ticket"] = Ticket
dataset["Ticket"].head(10)


# ### 4.3 Engineering the variable 'Family size'

# We can imagine that large families will have more difficulties to evacuate, looking for theirs sisters/brothers/parents during the evacuation. So, in the file I forked, the author choosed to create a "Fize" (family size) feature which is the sum of SibSp , Parch and 1 (including the passenger). It's a good idea, let's do it.

# In[ ]:


# Create a family size descriptor from SibSp and Parch
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1


# In[ ]:


g = sns.factorplot(x="Fsize",y="Survived",data = dataset)
g = g.set_ylabels("Survival Probability")


# Some models work better if we 'help' them segmenting the space of the possible levels for the variable. Let's create 4 types of Fsize:
# * Single individuals (1)
# * Couples or Parent-kid, Parent-rellative (2)
# * Medium families (up to 4 individuals)
# * Large Families (5+)
# 

# In[ ]:


# Create new feature of family size
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)

g = sns.factorplot(x="Single",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="SmallF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="MedF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="LargeF",y="Survived",data=dataset,kind="bar")
g = g.set_ylabels("Survival Probability")


# ## 5. Dataset preparation for Model

# ### 5.1 One hot encoding the categorical variables
# 
# 

# In[ ]:


#One-hot encoder for categorical variables
dataset = pd.get_dummies(dataset, columns = ["Title"])
dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
dataset = pd.get_dummies(dataset, columns = ["Cabin"], prefix="Cab")
dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T_")


# In[ ]:


# convert Sex into categorical value 0 for male and 1 for female
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})


# In[ ]:


# Create categorical values for Pclass
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")



# ### 5.2 Separation of the train test dataframes

# In[ ]:


# Drop useless variables 
dataset.drop(labels = ["PassengerId","Name"], axis = 1, inplace = True)

dataset.head()


# In[ ]:


print(dataset.info())


# In[ ]:


#separation of the train/test
df_train = dataset[:train_len]
df_test = dataset[train_len:]


# In[ ]:


df_test.drop(labels=["Survived"],axis = 1,inplace=True)
df_test.columns


# In[ ]:


#X vs Y split
df_train_X = df_train.drop(labels =['Survived'], axis =1)
df_train_Y = df_train['Survived']


# ### 5.3 Creating feature list for tensorflow

# In[ ]:


#Everything is numerical, so it's easier to create a list of variables
variable_names = df_train_X.columns

tf_variables =[]

for index in variable_names:
   tf_variables.append(tf.feature_column.numeric_column(index)) 

tf_variables


# ### 5.4 Scaling the numerical variables

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler()
scaled_X_train = Scaler.fit_transform(df_train_X)
scaled_X_test = Scaler.transform(df_test)


# In[ ]:


#Converting numerical variables into categorical variables
scaled_X_train  = pd.DataFrame(scaled_X_train,columns=df_train_X.columns)
scaled_X_train.head()


# ### 5.5 Bucketizing the variable 'Age'

# In[ ]:



scaled_X_train.Age.hist(bins=20)


# In[ ]:


Age_bucket = tf.feature_column.bucketized_column(tf_variables[0],boundaries=[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,
                                                                             0.50,0.55,0.60,0.65,0.70,0.80,1.02])


# ### 5.6 Bucketizing the variable 'Fare'

# In[ ]:



scaled_X_train.Fare.hist(bins=40)


# In[ ]:



Fare_bucket = tf.feature_column.bucketized_column(tf_variables[1],boundaries=[0.05,0.1,0.15,0.2,0.4,0.6,1.02])


# In[ ]:



tf_variables.append(Age_bucket)
tf_variables.append(Fare_bucket)


# ### 5.7 Train_test_split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:



X_train, X_test, Y_train, Y_test = train_test_split(scaled_X_train,df_train_Y, test_size =0.3, random_state =101)


# In[ ]:


print(X_train.shape)
print(scaled_X_train.shape)


# ## 6. Models

# ### 6.1 Quick start with a linear model

# In[ ]:


input_function = tf.estimator.inputs.pandas_input_fn(x=X_train, y= Y_train,
                                                     batch_size =10, num_epochs =1000, shuffle =True )


# In[ ]:


linear_model= tf.estimator.LinearClassifier(feature_columns=tf_variables, n_classes=2)


# In[ ]:


linear_model.train(input_fn=input_function,steps=1000)


# In[ ]:


#evaluate model
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=Y_test, batch_size=10, num_epochs=1, shuffle=False)


# In[ ]:


results =linear_model.evaluate(eval_input_func)


# In[ ]:


results


# Accuracy of 0.79 will be our start point or baseline.

# ### 6.2 Dense Neural Networks (DNNs)

# In[ ]:


#Resetting tensorflow graphs.
tf.Session.reset
sess = tf.InteractiveSession()

#redefining the tensors
tf_variables =[]

for index in variable_names:
   tf_variables.append(tf.feature_column.numeric_column(index)) 

Age_bucket = tf.feature_column.bucketized_column(tf_variables[0],boundaries=[0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,
                                                                             0.50,0.55,0.60,0.65,0.70,0.80,1.02])

Fare_bucket = tf.feature_column.bucketized_column(tf_variables[1],boundaries=[0.05,0.1,0.15,0.2,0.4,0.6,1.02])
tf_variables.append(Age_bucket)
tf_variables.append(Fare_bucket)


# In[ ]:


dnn_input_function = tf.estimator.inputs.pandas_input_fn(x=X_train, y= Y_train,batch_size =10,
                                                         num_epochs =1000, shuffle =True )


# In[ ]:


dnn_model=tf.estimator.DNNClassifier(hidden_units=[10,10,10,10],feature_columns=tf_variables, 
                                     n_classes =2, optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                    dropout=0.2)


# In[ ]:


dnn_model.train(input_fn = dnn_input_function, steps =1000)


# In[ ]:


#evaluate model
dnn_eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=Y_test, 
                                                          batch_size=10, num_epochs=1, shuffle=False)


# In[ ]:


dnn_results =dnn_model.evaluate(eval_input_func)


# In[ ]:


dnn_results


# Accuracy of 0.80 is better than a linear model

# ## 7 Prediction
# #### 7.1 Predict and Submit results

# In[ ]:


scaled_X_test = pd.DataFrame(scaled_X_test,columns=df_train_X.columns)
scaled_X_test.head()


# In[ ]:


#making predictions
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=scaled_X_test,batch_size=10,num_epochs=1, shuffle=False )


# In[ ]:


preds =list(dnn_model.predict(pred_input_func))
preds


# In[ ]:


predictions =[p['class_ids'][0] for p in preds]


# In[ ]:


predictions_df =pd.DataFrame(predictions,columns=['Survived'])
predictions_df.head()


# In[ ]:


df_test['Survived']=predictions

results =  pd.concat([IDtest,predictions_df], axis=1)

results.head()
results.to_csv("output_python_tf.csv",index=False)


# Thank you very much for reading it!!
