#!/usr/bin/env python
# coding: utf-8

# 
# 

# 

# ## 1 Importing the Libraries

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# ML Libraries: 
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


# In[ ]:


import plotly.figure_factory as ff
import plotly.offline as py 
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly import tools 
py.init_notebook_mode (connected = True)

import cufflinks as cf
cf.go_offline()


# ## 2 Importing the Dataset and General View

# In[ ]:


dftrain = pd.read_csv("../input/train.csv")
dftest = pd.read_csv("../input/test.csv")


# ###### Lets Concatenate both the data frames for Exploratory Data Analysis :

# In[ ]:


df = pd.concat([dftrain, dftest], axis = 0 )


# In[ ]:


# Head of Training set:
dftrain.head(5)


# In[ ]:





# In[ ]:


# Breif information about the Data Sets
dftrain.info()
print("________________________________")
dftest.info()


# In[ ]:


# General Description about the Training Data set:
dftrain.drop(["PassengerId"], axis =1).describe()


# ##  3. Data Visualisation of Entire Data Frame (Training and Test Set)

# In[ ]:


classd= {1:"First Class", 2: "Second Class", 3: "Third Class"}
df["Class"] = df["Pclass"].map(classd)

first = df[df["Class"]=="First Class"]
sec = df[df["Class"]=="Second Class"]
thrd= df[df["Class"]=="Third Class"]

male= df[df["Sex"]=="male"]
female= df[df["Sex"]=="female"]


# ### 3. a.  SEX 

# In[ ]:


#1. Pie Chart for Sex count
sex_count = df["Sex"].value_counts()
print(sex_count)

colors = ['aqua', 'pink']

trace= go.Pie(labels = sex_count.index,
              values = sex_count.values, marker=dict(colors=colors))

layout = go.Layout(title = "Sex Distribution")

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


# 1 Boxplot 
trace = go.Box(y = male["Fare"],fillcolor="aqua", name= "male" )
trace1 = go.Box(y = female["Fare"], fillcolor="pink", name= "female" )

layout = go.Layout(title="Fare distribution w.r.t Sex", yaxis=dict(title="Sex"), xaxis= dict(title="Fare"))

data=[trace, trace1]
fig = go.Figure(data = data, layout=layout)
py.iplot(fig)

# 2 Violin Plot 
trace1 = go.Violin( y = male["Fare"], fillcolor="aqua", name="Male")
trace2 = go.Violin( y = female["Fare"],fillcolor="pink", name="Female")

layout = go.Layout(title="Fare distribution w.r.t Sex", yaxis=dict(title="Fare"), xaxis= dict(title="Sex"))

data=[trace1, trace2]
fig = go.Figure(data = data, layout=layout)
py.iplot(fig)


# In[ ]:


#male= df[df["Sex"]=="male"]
#female= df[df["Sex"]=="female"]

# Box
trace = go.Box(y = male["Age"],fillcolor="aqua", name= "male" )
trace1 = go.Box(y = female["Age"], fillcolor="pink", name= "female" )
layout = go.Layout(title="Age w.r.t Sex", yaxis=dict(title="Age"), xaxis= dict(title="Sex"))

data=[trace, trace1]
fig = go.Figure(data = data, layout=layout)
py.iplot(fig)


# In[ ]:





# ### 2.b.  Class 

# In[ ]:


# Pie Chart
class_count = df["Class"].value_counts()
print(class_count)

colors = ["lightorange", "lightpurple", "lightgreen"]
trace = go.Pie( labels= class_count.index,values = class_count.values, marker = dict(colors = colors))

layout = go.Layout( title = "Total Class Distribution")

data= [trace]
fig = go.Figure(data= data, layout= layout)
py.iplot(fig)


# In[ ]:


# classd= {1:"First Class", 2: "Second Class", 3: "Third Class"}
#df["Class"] = df["Pclass"].map(classd)

# Bar Plot
trace1 = go.Bar(x=class_count.index, y= class_count.values, 
                marker=dict(
                color=dftrain["Age"],
                colorscale = 'Jet'))

layout1 = go.Layout(title = "Class Count" )

data= [trace1]
fig = go.Figure(data= data, layout= layout1)
py.iplot(fig)


#plt plot
plt.figure(figsize=(15,8))
sns.countplot(x= df["Class"], hue= df["Sex"], palette="seismic")
plt.title("Male/Sex per Class", fontsize = 18)
plt.xlabel("Class", fontsize = 18)
plt.ylabel("Count", fontsize = 18)
plt.show()




# In[ ]:


#first = dftrain[dftrain["Class"]=="First Class"]
#sec = dftrain[dftrain["Class"]=="Second Class"]
#thrd= dftrain[dftrain["Class"]=="Third Class"]

# Box plot
trace1 = go.Box( x = first["Fare"], fillcolor="yellow", name="First Class")
trace2 = go.Box( x = sec["Fare"],fillcolor="mediumpurple", name="Second Class")
trace3 = go.Box( x = thrd["Fare"], fillcolor="mistyrose", name="Third Class")

layout = go.Layout(title="Fare distribution w.r.t Class", yaxis=dict(title="Class"), xaxis= dict(title="Fare"))

data=[trace1, trace2, trace3]
fig = go.Figure(data = data, layout=layout)
py.iplot(fig)

# Violin Plot
trace1 = go.Violin( y = first["Fare"], fillcolor="yellow", name="First Class")
trace2 = go.Violin( y = sec["Fare"],fillcolor="mediumpurple", name="Second Class")
trace3 = go.Violin( y = thrd["Fare"], fillcolor="mistyrose", name="Third Class")

layout = go.Layout(title="Age distribution w.r.t Class", yaxis=dict(title="Fare"), xaxis= dict(title="Class"))

data=[trace1, trace2, trace3]
fig = go.Figure(data = data, layout=layout)
py.iplot(fig)

# violinplot
#plt.figure(figsize=(14,5))
#sns.violinplot(x = df["Class"], y=df["Age"], palette="magma")
#plt.xlabel("Class", fontsize =20)
#plt.ylabel("Age", fontsize =20)
#plt.title("Violin plot of Class w.r.t Age",fontsize =20)
#plt.show()


# In[ ]:


# Box Plot Age Vs Class
# Box plot
trace1 = go.Box( y = first["Age"], fillcolor="yellow", name="First Class")
trace2 = go.Box( y = sec["Age"],fillcolor="mediumpurple", name="Second Class")
trace3 = go.Box( y = thrd["Age"], fillcolor="lavender", name="Third Class")

layout = go.Layout(title="Age distribution w.r.t Class", yaxis=dict(title="Age"), xaxis= dict(title="Class"))

data=[trace1, trace2, trace3]
fig = go.Figure(data = data, layout=layout)
py.iplot(fig)


# ### 2.c. Age Distribution

# In[ ]:


# Age Bar plot 
age_count = df["Age"].dropna().value_counts()

trace = go.Bar(x = age_count.index,
              y = age_count.values, 
              marker = dict(color = df["Age"],
                           colorscale = "Jet", 
                           showscale = True))
layout = go.Layout(title = "Age Distribution", 
                  yaxis = dict(title = "Number of People"))
data = [trace]
fig = go.Figure(data = data, layout = layout)
py.iplot(fig)


# Age Vs Fare
trace=go.Scatter(x = df["Age"].dropna(), y=df["Fare"], mode = "markers", 
                marker = dict(size = 6, color = "lightgreen"))

layout = go.Layout(title ="Age to Fare Plot", xaxis= dict(title = "Age"), 
                   yaxis= dict(title = "Fare"))

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ###### 2.d. Joint plot: Distribution w.r.t to AGE & FARE

# In[ ]:



plt.figure(figsize=(14,5))
sns.jointplot(x=df["Age"], y=df["Fare"], kind="reg",
              color="purple", ratio=3, height=7,dropna= True)
plt.xlabel("Age", fontsize =16)
plt.ylabel("Fare", fontsize =16)
plt.title("Fare vs Age",fontsize =16)
plt.show()


# ### 2.e. Survived Plots

# In[ ]:


df["Survived"].value_counts()


# In[ ]:


# Survival count
sur_count = df["Survived"].value_counts()
trace1 = go.Bar(x=sur_count.index, y= sur_count.values, 
                marker=dict(
                color=df["Age"],
                colorscale = 'Portland'))

layout1 = go.Layout(title = "Class Count" )

data= [trace1]
fig = go.Figure(data= data, layout= layout1)
py.iplot(fig)


#  Looks like 1098 people did not survive, while around 684 people in both the dataset survived

# In[ ]:


# Survival with hue of Sex

fig, (ax1, ax2) = plt.subplots(2, figsize= (16,7))
sns.countplot(x="Survived", hue="Sex",  data=df, palette="magma", ax=ax1)
sns.countplot(x="Survived", hue="Class",  data=df, palette="magma", ax=ax2)

plt.show()
plt.tight_layout()


# We can see a trend here. It looks like people who couldn't survived were much more like to be male. While on the other hand people who survived are more likely to be female. 

# In[ ]:


plt.figure(figsize= (16,7))
sns.scatterplot(data=df, x="Age", y="Fare", hue="Sex")
plt.show()
plt.figure(figsize= (16,7))
sns.scatterplot(data=df, x="Age", y="Fare", hue="Survived")
plt.show()


# In[ ]:





# ### 2.f. Sibling and Spouse on board

# In[ ]:


sib_count = df["SibSp"].value_counts()
trace1 = go.Bar(x=sib_count.index, y= sib_count.values, 
                marker=dict(
                color=df["Age"],
                colorscale = 'Electric'))

layout1 = go.Layout(title = "Sibling and Spouse  Count" )

data= [trace1]
fig = go.Figure(data= data, layout= layout1)
py.iplot(fig)


# ###  2. g. Parent and child on board

# In[ ]:


# Count plot for number of sibling or Spouse aboard
par_count = df["SibSp"].value_counts()
trace1 = go.Bar(x=par_count.index, y= par_count.values, 
                marker=dict(
                color=df["Age"],
                colorscale = 'Cividis'))

layout1 = go.Layout(title = "Sibling and Spouse  Count" )

data= [trace1]
fig = go.Figure(data= data, layout= layout1)
py.iplot(fig)


# In[ ]:





# ###  3. g. Embarked Ship 

# In[ ]:


em_count = df["Embarked"].value_counts()
trace1 = go.Bar(x=em_count.index, y= em_count.values, 
                marker=dict(
                color=df["Age"],
                colorscale = 'Viridis'))

layout1 = go.Layout(title = "Sibling and Spouse  Count" )

data= [trace1]
fig = go.Figure(data= data, layout= layout1)
py.iplot(fig)


# In[ ]:





# ## 4 Data Visualisation for Data Preprocessing

# In[ ]:


fig, (ax1,ax2) = plt.subplots (2, figsize = (15,10))
sns.heatmap(dftest.isnull(), cmap="viridis", yticklabels=False
           ,cbar = False, ax= ax1) 
sns.heatmap(dftrain.isnull(), cmap="viridis", yticklabels=False
           ,cbar = False, ax= ax2)
plt.tight_layout()


# So we can check out that we're missing some age information and we are missing a lot of Cabin information. Roughly about 20 percent of that age data is missing and the proportion of age missing is likely small enough for a reasonable replacement of some form of imputation meaning I can actually use the knowledge of the other columns to fill in reasonable values for that age column. 
# Looking at the cabin column however it looks like we're just missing too much of that data to do something useful with it at a basic level. We're going to go ahead and probably drop this later or change it to send up some other feature like

# # 5 Data Preprocessing

# ### 5.a.  Dealing with MISSING VALUES:

# In[ ]:


# on the basis of the box plot of age vs class. 
# We can impute the average age on the basis of that plot
def imput_age(col):
    Age = col[0]
    Pclass = col[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29 
        else:
            return 24
    else:
        return Age

dftrain["Age"] = dftrain[["Age", "Pclass"]].apply (imput_age, axis=1)
dftest["Age"] = dftest[["Age", "Pclass"]].apply (imput_age, axis=1)


# I tried to fill the missing values based of my age columns for both the data set ( Trainind & Test Set) 

# In[ ]:


dftrain.head(1)


# In[ ]:


fig, (ax1,ax2) = plt.subplots (2, figsize = (15,10))
sns.heatmap(dftest.isnull(), cmap="viridis", yticklabels=False
           ,cbar = False, ax= ax1) 
sns.heatmap(dftrain.isnull(), cmap="viridis", yticklabels=False
           ,cbar = False, ax= ax2)
plt.tight_layout()


# In[ ]:


dftest.head(1)


# ###### 5.B. ENCODING: Categorical Data

# In[ ]:


sex = pd.get_dummies(dftrain["Sex"], drop_first=True)
embark = pd.get_dummies(dftrain["Embarked"], drop_first=True)
clss = pd.get_dummies(dftrain["Pclass"], drop_first=True)


sext = pd.get_dummies(dftest["Sex"], drop_first=True)
embarkt = pd.get_dummies(dftest["Embarked"], drop_first=True)
clsst = pd.get_dummies(dftest["Pclass"], drop_first= True)


# In[ ]:


# Concatinating the new features to the Clear Trainig set:
train = pd.concat([dftrain, sex, embark, clss], axis =1)
train.head(1)


# In[ ]:


# Concatinating the new features to the Test set:
test = pd.concat([dftest, sext, embarkt, clsst], axis =1)
test.head(1)


# In[ ]:


# Dropping columns which are not required from training set
train.drop(["PassengerId", "Pclass","Name", "Sex", "Ticket", "Cabin", "Embarked"], axis = 1,inplace = True)
train.head(1)


# In[ ]:


# Dropping columns which are not required from training set
test.drop(["PassengerId", "Pclass","Name", "Sex", "Ticket", "Cabin", "Embarked"], axis = 1,inplace = True)
test.head(1)


# In[ ]:





# In[ ]:


fig, (ax1, ax2) = plt.subplots(2, figsize = (15,10))
sns.heatmap(train.corr(), annot= True, cmap= "magma", ax=ax1)
sns.heatmap(test.corr(), annot= True, cmap= "magma", ax=ax2)

plt.show()
plt.tight_layout()


# In[ ]:


fig, (ax1, ax2) = plt.subplots(2, figsize = (15,10))
sns.heatmap(train.isnull(), ax=ax1, yticklabels=False)
sns.heatmap(test.isnull(), yticklabels=False)
plt.show()


# ### 5.C. Creating Train & Test Split

# In[ ]:





# In[ ]:


test_y = pd.read_csv("../input/gender_submission.csv")
X_train = train.drop("Survived", axis=1)
y_train = train["Survived"]
X_test = test.fillna(dftest["Fare"].mean())
y_test = test_y["Survived"]


# In[ ]:


sns.heatmap(X_test.isnull())


# ## 6. Modeling 

# ### 6. a. Logistic Regression

# In[ ]:


log = LogisticRegression()
log.fit(X_train, y_train)
pred = log.predict(X_test)


# In[ ]:





# ### 6.b. Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred1 = rf.predict(X_test)


# In[ ]:





# ## 7. Model Evaluation

# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[ ]:


# Report of Logistic Regression
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# In[ ]:





# In[ ]:


# Report of Random Forest
print(confusion_matrix(y_test, pred1))
print(classification_report(y_test, pred1))


# ##### Next step in order to increase the precision and get more accuracy. I will be doing more feature engineering such as trying to grab the title of the names, cabin letter and ticket information. 

# In[ ]:





# In[ ]:


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
subm = rf.predict(X_test)


# In[ ]:


sub_rep = pd.DataFrame({"PassengerId": dftest["PassengerId"], 
                       "Survived" : subm})


# In[ ]:


sub_rep.to_csv("Tit Sub.csv", index = False)
print (sub_rep.shape)

