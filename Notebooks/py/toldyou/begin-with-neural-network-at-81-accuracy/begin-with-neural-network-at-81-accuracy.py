#!/usr/bin/env python
# coding: utf-8

# 
# >This is my first experience in kaggle, and this kernal is all about self-recording. The final score of the kernal is 0.81 and  placed at top 7%.  Process included 1. problem statement 2. understand data structure 3. feature generating and formating  4. feature selecting 5. NN model building.  
# 
# *** Some other good kernals (Inspired By) ***
# 1.  idea of grouping rules which gives 84%, [The Titanic Mega Model scores 84%](#https://www.kaggle.com/cdeotte/titantic-mega-model-0-84210)
# 2. A complete data anlaysis procedure, [A Data Science Framework: To Achieve 99% Accuracy](#https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy)
# 3. --- [Exploring Survival on the Titanic](#https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic)
# 4. Plotly Implement [https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python](#https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)
# 
# # Begin with Neural Network at 81% Accuracy

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import re
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# Sklearn kit for xgboost
from sklearn import model_selection
# matplotlib setting
get_ipython().magic(u'matplotlib inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
# Loading Training Dat
train_dat = pd.read_csv("../input/train.csv")
test_dat  = pd.read_csv("../input/test.csv")
data1 = train_dat.copy(deep = True)
data_cleaner = [data1, test_dat]
# ============
reset = data_cleaner


# ## <strong> Content </strong> ##
# [Section 1. Identify the problem](#d1)<br>
# [Section 2. Understand Data](#d2)<br>
# [Section 3. Feature generating and formating](#d3)<br>
# [Section 4. Feature selecting](#d4)<br>
# [Section 5. NN modeling](#d5)<br>
#  

# ## Section 1. Identify the problem
# 
# > The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensationa .....
# 
# >  -->> Binary classification
# 
# According to the description, the target is to predict if the passenger survived or not (0 or 1). So its a binary classificaiton problem, the first thing came to my mind is a sigmoid layer, and then svm etc (which could support modeling phrase).

# ## Section 2. Understand Data
# |Variable Name|Description|Comments|
# |------------------|-------------|-------------|
# |Survival|0 = No, 1 = Yes| Target column|
# |Pclass| 1 = 1st, 2 = 2nd, 3 = 3rd| Q: 1st class has more chance to survive?|
# |Sex|-- | -- |
# |Age| -- | Q: Children ↑ ？|
# |Sibsp|--| Q: Related to parch ? |
# |Parch|--| Q: same above |
# |Ticket|--| Looks like it gave same kind of information as Pclass, maybe still valuable|
# |Fare| -- | Ticket ? Pclass ? |
# |Cabin| -- | Maybe related to the escape tunnel, the closer to the deck, the more chance survive| 
# |Embarked| -- | No idea|
# 
# ### 2.1 General information 

# In[ ]:


print(train_dat.info())
train_dat.sample(10)


# ### 2.2 NaN Problem 
# 1. **Cabin:** 8/10 are NaN, might not necessaryly help 
# 2. **Name:** comparing to the raw name, title might help  
# 3. **PassengerId:** not important
# <br>
# #### Null value detection

# In[ ]:


print(train_dat.isnull().sum())
print(train_dat["Age"].isnull().sum()/len(train_dat))
print(train_dat["Cabin"].isnull().sum()/len(train_dat))


# Base on the information above,  'Cabin' and 'Age' (20% and 80% data missing) need to be filled or simply drop, otherwise will hugely impact the performance. Age  can be filled with average/median,  Cabin is more difficult to guess ( drop).
# 
# The process will be done in the later process.
# 
# ## 2.3  Features
# To verify the intuition of these features, we could calcuate the correlation between these featuress and the target (survival rate).
# 
# ### Pclass & Survive Rate
# 

# In[ ]:


t =  train_dat["Pclass"].value_counts()
st = [len(train_dat.loc[(train_dat["Pclass"]==k) & (train_dat["Survived"]==1)])/len(train_dat.loc[(train_dat["Pclass"]==k)]) for k in (t.keys())]
annotations1 = [dict(
            x=xi,
            y=yi,
            text=str(yi),
            xanchor='auto',
            yanchor='bottom',
            showarrow=False,
) for xi, yi in zip([str(k)+" class" for k in (t.keys())], t.values)]

data = [go.Bar(x=[str(k)+" class" for k in (t.keys())],
               y=t.values,
               name="Number",
               marker=dict(
#                color=[str(k) for k in (t.keys())]
                color=["red","green","blue"]
               ),
               opacity=0.2
              ),
        go.Scatter(x=[str(k)+" class" for k in (t.keys())],
                   y=st,
                   name="Survive Rate",
                   mode='lines+text',
                   opacity=0.6,
                   yaxis="y2",
                   text=[str(round(k*100))+"%" for k in st],
                   textposition='top left')]
layout = go.Layout(title="Pclass vs Survival Rate",
                   yaxis=dict(title="Numbers"),
                   yaxis2=dict(
                        domain=[0.0,1.0],
                        title='Percentage',
                        overlaying='y',
                        side='right'
                       ),
                   annotations =annotations1
                  )
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)


# As the graph shows, the highest survival rate appeared at 1st class (63%) and 2nd class with (47%) survival rate, covered 44% data in total. Comparing this to the 3rd class, these two classes seems like higer classes have an advantage over lower classes, which provided that, wealth level is a factor needed to consider with. 

# ### Sex & Survive Rate

# In[ ]:


t =  train_dat["Sex"].value_counts()
st = [len(train_dat.loc[(train_dat["Sex"]==k) & (train_dat["Survived"]==1)])/len(train_dat.loc[(train_dat["Sex"]==k)]) for k in (t.keys())]
annotations1 = [dict(
            x=xi,
            y=yi,
            text=str(yi),
            xanchor='auto',
            yanchor='bottom',
            showarrow=False,
) for xi, yi in zip([str(k)+" class" for k in (t.keys())], t.values)]

data = [go.Bar(x=[str(k)+" class" for k in (t.keys())],
               y=t.values,
               name="Number",
               marker=dict(
               color=[ 'red','green']),
               opacity=0.2),
        go.Scatter(x=[str(k)+" class" for k in (t.keys())],
                   y=st,
                   name="Survive Rate",
                   mode='lines+text',
                   opacity=0.6,
                   yaxis="y2",
                   text=[str(round(k*100))+"%" for k in st],
                   textposition='top left')]
layout = go.Layout(title="Sex vs Survival Rate",
                   yaxis=dict(title="Numbers"),
                   yaxis2=dict(
                        domain=[0.0,1.0],
                        title='Percentage',
                        overlaying='y',
                        side='right'
                       ),
                   annotations =annotations1
                  )
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)


# It is clear to see that 74% female survived, which proved, gender imformation is an important factor as well. 

# ### Age & Survive Rate

# In[ ]:


dat = (sorted(train_dat.loc[~train_dat["Age"].isnull(),"Age"]))
data = [
        go.Scatter(x=list(range(len(dat))),
                   y=dat,
                   name="Survive Rate",
                   mode='lines+text',
                   opacity=0.6)]
layout = go.Layout(title="Age vs Survival Rate",
                   yaxis=dict(title="Numbers"),
                   )
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)


# In[ ]:


print("==== 4 split (quantile) ====")
print((train_dat.loc[~train_dat["Age"].isnull(),"Age"]).quantile([.1,.5,.75,.9]))


# ### Title & Survive Rate
# Extracting information in name. (Dr. Mr. etc.)

# In[ ]:


train_dat["Title"] = train_dat['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
# print(td.value_counts())
# group others <7
td = train_dat["Title"]
train_dat.loc[~train_dat["Title"].isin(["Mr","Mrs","Miss","Master","Dr"]),"Title"] = "Others"
t_list = train_dat["Title"].value_counts()
st = [len(train_dat.loc[(train_dat["Title"]==k) & (train_dat["Survived"]==1)])/len(train_dat.loc[(train_dat["Title"]==k)]) for k in t_list.keys()] 
z = t_list[5:].sum()
td = td.value_counts()[:5]
td["Others"] = z
print(td)


# In[ ]:


annotations1 = [dict(
            x=xi,
            y=yi,
            text=str(yi),
            xanchor='auto',
            yanchor='bottom',
            showarrow=False,
) for xi, yi in zip([k for k in (t_list.keys())], t_list.values)]

data = [go.Bar(x=[k for k in (t_list.keys())],
               y=t_list.values,
               name="Number",
               marker=dict(
               color=["red","green","blue"]
               ),
               opacity=0.2),
        go.Scatter(x=[k for k in (t_list.keys())],
                   y=st,
                   name="Survive Rate",
                   mode='lines+text',
                   opacity=0.6,
                   yaxis="y2",
                   text=[str(round(k*100))+"%" for k in st],
                   textposition='top left')]
layout = go.Layout(title="Title vs Survival Rate",
                   yaxis=dict(title="Numbers"),
                   yaxis2=dict(
                        domain=[0.0,1.0],
                        title='Percentage',
                        overlaying='y',
                        side='right'
                       ),
                   annotations =annotations1
                  )
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)


# By adding title information, we could see that the peak appeared at [ Miss / 70% ] and [ Mrs / 79% ], which is a relevent high rate. And it proved the importance of gender in another way.

# ### Fare & Survive Rate

# In[ ]:


print("==== 4 split (quantile) ====")
print((train_dat.loc[~train_dat["Fare"].isnull(),"Fare"]).quantile([.1,.5,.75,.9]))
print(3 in range(0,3))
print(train_dat.at[1,"Fare"])


# In[ ]:


for i in range(0,len(train_dat)):
    t = round(train_dat.get_value(i,"Fare"))
    if (t in range(0,8)):
        train_dat.at[i,"Fare_g"] = "[0,8)"
    elif (t in range(8,15)):
        train_dat.at[i,"Fare_g"] = "[8,15)"
    elif (t in range(15,31)):
        train_dat.at[i,"Fare_g"] = "[15,31)"
    elif (t in range(31,78)):
        train_dat.at[i,"Fare_g"] = "[31,78)"
    else:
        train_dat.at[i,"Fare_g"] = "[78,+)"

t_list = train_dat["Fare_g"].value_counts()
st = [len(train_dat.loc[(train_dat["Fare_g"]==k) & (train_dat["Survived"]==1)])/len(train_dat.loc[(train_dat["Fare_g"]==k)]) for k in t_list.keys()] 


# In[ ]:


annotations1 = [dict(
            x=xi,
            y=yi,
            text=str(yi),
            xanchor='auto',
            yanchor='bottom',
            showarrow=False,
) for xi, yi in zip([k for k in (t_list.keys())], t_list.values)]

data = [go.Bar(x=[k for k in (t_list.keys())],
               y=t_list.values,
               name="Number",
               marker=dict(
#                color=[k for k in (t_list.keys())]
               ),
               opacity=0.6),
        go.Scatter(x=[k for k in (t_list.keys())],
                   y=st,
                   name="Survive Rate",
                   mode='lines+text',
                   opacity=0.6,
                   yaxis="y2",
                   text=[str(round(k*100))+"%" for k in st],
                   textposition='top left')]
layout = go.Layout(title="Fare vs Survival Rate",
                   yaxis=dict(title="Numbers"),
                   yaxis2=dict(
                        domain=[0.0,1.0],
                        title='Percentage',
                        overlaying='y',
                        side='right'
                       ),
                   annotations =annotations1
                  )
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)


# ### Is Alone vs Survived Rate

# In[ ]:


for i in range(0,len(train_dat)):
    if (train_dat.at[i,"SibSp"]!=0 or train_dat.at[i,"Parch"]!=0 ):
        train_dat.at[i,"isAlone"] = 0
    else:
        train_dat.at[i,"isAlone"] = 1
t_list = train_dat["isAlone"].value_counts()
st = [len(train_dat.loc[(train_dat["isAlone"]==k) & (train_dat["Survived"]==1)])/len(train_dat.loc[(train_dat["isAlone"]==k)]) for k in t_list.keys()] 


# In[ ]:


annotations1 = [dict(
            x=xi,
            y=yi,
            text=str(yi),
            xanchor='auto',
            yanchor='bottom',
            showarrow=False,
) for xi, yi in zip([k for k in (t_list.keys())], t_list.values)]

data = [go.Bar(x=[k for k in (t_list.keys())],
               y=t_list.values,
               name="Number",
               marker=dict(
               color=["red","green"]
               ),
               opacity=0.2),
        go.Scatter(x=[k for k in (t_list.keys())],
                   y=st,
                   name="Survive Rate",
                   mode='lines+text',
                   opacity=0.6,
                   yaxis="y2",
                   text=[str(round(k*100))+"%" for k in st],
                   textposition='top left')]
layout = go.Layout(title="isAlone vs Survival Rate",
                   yaxis=dict(title="Numbers"),
                   yaxis2=dict(
                        domain=[0.0,1.0],
                        title='Percentage',
                        overlaying='y',
                        side='right'
                       ),
                   annotations =annotations1
                  )
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)


# There are two related columns, [ SibSp ]  and [ Parch ]. A simple approach is: if a person nither has paraent, nor has child, then we set this person["isAlone"] = True, otherwise "False". The result shows that the if a person are not alone, than he has a better chance to survive. Here is assumption, by considering moral problem, most of children and female were boraded, most of them are not alone. Further correlation could be found when female.age etc. info included.

# ### Ticket vs Survived Rate

# In[ ]:


for i in range(0,len(train_dat)):
    t = (train_dat.get_value(i,"Ticket"))
    if (re.sub(r"\d","",t)==""):
        train_dat.at[i,"Ticket_g"] = "Num"
    elif (re.search("A5",t)!=None):
        train_dat.at[i,"Ticket_g"] = "A5"
    elif (re.search("CA",t)!=None):
        train_dat.at[i,"Ticket_g"] = "CA"
    elif (re.search("PC",t)!=None):
        train_dat.at[i,"Ticket_g"] = "PC"
    elif (re.search("SOTON",t)!=None):
        train_dat.at[i,"Ticket_g"] = "SOTON"
    elif (re.search("STON",t)!=None):
        train_dat.at[i,"Ticket_g"] = "STON"
    else:
        train_dat.at[i,"Ticket_g"] = "Others"
train_dat
t_list = train_dat["Ticket_g"].value_counts()
st = [len(train_dat.loc[(train_dat["Ticket_g"]==k) & (train_dat["Survived"]==1)])/len(train_dat.loc[(train_dat["Ticket_g"]==k)]) for k in t_list.keys()] 


# In[ ]:


annotations1 = [dict(
            x=xi,
            y=yi,
            text=str(yi),
            xanchor='auto',
            yanchor='bottom',
            showarrow=False,
) for xi, yi in zip([k for k in (t_list.keys())], t_list.values)]

data = [go.Bar(x=[k for k in (t_list.keys())],
               y=t_list.values,
               name="Number",
               marker=dict(
#                color=[k for k in (t_list.keys())]
               ),
               opacity=0.6),
        go.Scatter(x=[k for k in (t_list.keys())],
                   y=st,
                   name="Survive Rate",
                   mode='lines+text',
                   opacity=0.6,
                   yaxis="y2",
                   text=[str(round(k*100))+"%" for k in st],
                   textposition='top right')]
layout = go.Layout(title="Ticket vs Survival Rate",
                   yaxis=dict(title="Numbers"),
                   yaxis2=dict(
                        domain=[0.0,1.0],
                        title='Percentage',
                        overlaying='y',
                        side='right'
                       ),
                   annotations =annotations1
                  )
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)


# Ticket's information is a little bit tricky. Although [ PC ] is a peak @ 65%, but due to the ticket type distribution, it might not the actual case. Majority people owned number tickets, but we could not distinct how these information influence the survived rate at current info level. So for now, this feature was excluded.

# ### Embarked vs Survive Rate

# In[ ]:


t_list = train_dat["Embarked"].value_counts()
st = [len(train_dat.loc[(train_dat["Embarked"]==k) & (train_dat["Survived"]==1)])/len(train_dat.loc[(train_dat["Embarked"]==k)]) for k in t_list.keys()] 
annotations1 = [dict(
            x=xi,
            y=yi,
            text=str(yi),
            xanchor='auto',
            yanchor='bottom',
            showarrow=False,
) for xi, yi in zip([k for k in (t_list.keys())], t_list.values)]

data = [go.Bar(x=[k for k in (t_list.keys())],
               y=t_list.values,
               name="Number",
               marker=dict(
               color=["red","green","blue"]
               ),
               opacity=0.2),
        go.Scatter(x=[k for k in (t_list.keys())],
                   y=st,
                   name="Survive Rate",
                   mode='lines+text',
                   opacity=0.6,
                   yaxis="y2",
                   text=[str(round(k*100))+"%" for k in st],
                   textposition='top right')]
layout = go.Layout(title="Embarked vs Survival Rate",
                   yaxis=dict(title="Numbers"),
                   yaxis2=dict(
                        domain=[0.0,1.0],
                        title='Percentage',
                        overlaying='y',
                        side='right'
                       ),
                   annotations =annotations1
                  )
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)


# Same situation, due to the unbalanced data, [ Embarded ] might not a good feature to predict survived rate.

# ### Family Size vs  Survive Rate 

# In[ ]:


train_dat["Fmsize"] = train_dat["SibSp"] + train_dat["Parch"] + 1
for i in range(0,len(train_dat)):
    t = (train_dat.at[i,"Fmsize"])
    if (t == 1):
        train_dat.at[i,"Fmsize_c"] = "Single"
    elif (t==2):
        train_dat.at[i,"Fmsize_c"] = "SmallF"
    elif (t>=3 and t <= 4):
        train_dat.at[i,"Fmsize_c"] = "MedF"
    else:
        train_dat.at[i,"Fmsize_c"] = "LargeF"
t_list = train_dat["Fmsize_c"].value_counts()
        
st = [len(train_dat.loc[(train_dat["Fmsize_c"]==k) & (train_dat["Survived"]==1)])/len(train_dat.loc[(train_dat["Fmsize_c"]==k)]) for k in t_list.keys()] 
annotations1 = [dict(
            x=xi,
            y=yi,
            text=str(yi),
            xanchor='auto',
            yanchor='bottom',
            showarrow=False,
) for xi, yi in zip([k for k in (t_list.keys())], t_list.values)]

data = [go.Bar(x=[k for k in (t_list.keys())],
               y=t_list.values,
               name="Number",
               marker=dict(
               color=["red","green","blue"]
               ),
               opacity=0.2),
        go.Scatter(x=[k for k in (t_list.keys())],
                   y=st,
                   name="Survive Rate",
                   mode='lines+text',
                   opacity=0.6,
                   yaxis="y2",
                   text=[str(round(k*100))+"%" for k in st],
                   textposition='top right')]
layout = go.Layout(title="FamilySize vs Survival Rate",
                   yaxis=dict(title="Numbers"),
                   yaxis2=dict(
                        domain=[0.0,1.0],
                        title='Percentage',
                        overlaying='y',
                        side='right'
                       ),
                   annotations =annotations1
                  )
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)


# ### correlations (Heatmap)

# In[ ]:


# train_dat = pd.read_csv("../input/train.csv")
tp=train_dat
tp[0:10]


# In[ ]:


tp = train_dat
tp.loc[tp["Embarked"]=="S","Embarked_c"] = 1.0
tp.loc[tp["Embarked"]=="C","Embarked_c"] = 2.0
tp.loc[tp["Embarked"]=="Q","Embarked_c"] = 3.0

tp.loc[tp["Sex"]=="male","Sex_c"] = 0
tp.loc[tp["Sex"]=="female","Sex_c"] = 1

tp.loc[tp["Title"]=="Mr","Title_c"] = 0
tp.loc[tp["Title"]=="Miss","Title_c"] = 1
tp.loc[tp["Title"]=="Mrs","Title_c"] = 2
tp.loc[tp["Title"]=="Master","Title_c"] = 3
tp.loc[tp["Title"]=="Dr","Title_c"] = 4
tp.loc[tp["Title"]=="Others","Title_c"] = 5

tp.loc[tp["Fare_g"]=="[0,8)","Fare_c"] = 0
tp.loc[tp["Fare_g"]=="[8,15)","Fare_c"] = 1
tp.loc[tp["Fare_g"]=="[15,31)","Fare_c"] = 2
tp.loc[tp["Fare_g"]=="[31,78)","Fare_c"] = 3 
tp.loc[tp["Fare_g"]=="[78,+)","Fare_c"] = 4

tp.loc[tp["Ticket_g"]=="Others","Ticket_c"] = 0
tp.loc[tp["Ticket_g"]=="PC","Ticket_c"] = 1
tp.loc[tp["Ticket_g"]=="STON","Ticket_c"] = 2
tp.loc[tp["Ticket_g"]=="Num","Ticket_c"] = 3
tp.loc[tp["Ticket_g"]=="CA","Ticket_c"] = 4
tp.loc[tp["Ticket_g"]=="SOTON","Ticket_c"] = 5

tp.drop(["PassengerId","Age","SibSp","Parch","Fare"], axis=1, inplace = True)


# In[ ]:


tp.corr()


# In[ ]:


sns.heatmap(tp.corr(),cmap='coolwarm')


# ## Section 5. Modeling
# Feature selection and modeling.
# 

# In[ ]:


drop_column = ['PassengerId','Name','Cabin','Title','Ticket',"SibSp","Parch"]
for dat in data_cleaner:
    # Extract title
    dat['Age'].fillna(  -1, inplace = True)
    dat['Cabin'].fillna(-1, inplace = True)
    # Extract title

    dat['Title'] = dat['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    dat['Fare']  = pd.qcut(dat["Fare"],5,labels=False)
    dat['Fare'].fillna( dat['Fare'].median(), inplace = True)
#     dat["Age_c"] = pd.cut(dat["Age"],5,labels=False)
#     dat["Age_c"].fillna(-1, inplace = True)
   
    dat["Fsize"] = dat["SibSp"] + dat["Parch"] + 1
    dat['Single'] = dat['Fsize'].map(lambda s: 1 if s == 1 else 0)
    dat['SmallF'] = dat['Fsize'].map(lambda s: 1 if  s == 2  else 0)
    dat['MedF'] = dat['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
    dat['LargeF'] = dat['Fsize'].map(lambda s: 1 if s >= 5 else 0)
#     dat["Age"] = dat["Age"].apply(np.int64)
#     dat["Fare"] = dat["Fare"].apply(np.int64)
    
    dat["Fare"] = dat["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
    
    
    # == Tickets == 
    dat["Ticket"] = dat["Ticket"].str.replace(".","").str.replace("/","").str.upper().str.split(" ",expand=True)[0]
#     print(dat["Ticket"])
    dat["tick_num"] = dat['Ticket'].map(lambda s: 1 if re.sub(r"\d","",s)=="" else 0)
    dat["tick_A5"] =  dat['Ticket'].map(lambda s: 1 if re.search("A5",s)!=None else 0)
    dat["tick_CA"] =  dat['Ticket'].map(lambda s: 1 if re.search("CA",s)!=None else 0)
    dat["tick_PC"] =  dat['Ticket'].map(lambda s: 1 if re.search("PC",s)!=None else 0)
    dat["tick_SOTON"] =  dat['Ticket'].map(lambda s: 1 if re.search("SOTON",s)!=None else 0)
    dat["tick_STON"]  =  dat['Ticket'].map(lambda s: 1 if re.search("STON",s)!=None else 0)
    dat["tick_other"] = dat['Ticket'].map(lambda s: 1 if re.sub(r"\d","",s)!="" and re.search("A5",s)==None and re.search("CA",s)==None and re.search("PC",s)==None and re.search("SOTON",s)==None and re.search("STON",s)==None else 0)
    
    
    dat.loc[(dat["SibSp"]>0) | (dat["Parch"] >0), "isAlone"] = 0
    dat.loc[(dat["SibSp"]==0) & (dat["Parch"] ==0), "isAlone"] = 1
    
    dat["Age"] = pd.cut(dat["Age"],[-2,0,6,18,25,50,100],labels=False)

    dat.loc[dat["Sex"]=="male","Sex"]= 0
    dat.loc[dat["Sex"]=="female","Sex"]= 1
    
    dat.loc[dat["Title"] == "Mr","isMr"] = 1
    dat.loc[dat["Title"] == "Mrs","isMrs"] = 1
    dat.loc[dat["Title"] == "Miss","isMiss"] = 1
    dat.loc[dat["Title"] == "Master","isMaster"] = 1
    dat.loc[dat["Title"] == "Dr","isDr"] = 1
    dat.loc[~dat["Title"].isin(["Mr","Mrs","Miss","Master","Dr"]), "rareTitle"] = 1
    dat['isMr'].fillna( 0, inplace = True)
    dat['isMrs'].fillna( 0, inplace = True)
    dat['isMiss'].fillna( 0, inplace = True)
    dat['isMaster'].fillna( 0, inplace = True)
    dat['isDr'].fillna( 0, inplace = True)
    dat['rareTitle'].fillna( 0, inplace = True)
        
    dat.loc[dat["Embarked"] == "Q","Embarked"] = 0
    dat.loc[dat["Embarked"] == "S","Embarked"] = 1
    dat.loc[dat["Embarked"] == "C","Embarked"] = 2
    dat.loc[~dat["Embarked"].isin([0,1,2]), "Embarked"] = -1

    dat.loc[dat["Cabin"].str.count("A") > 0,"Cabin"] = 0
    dat.loc[dat["Cabin"].str.count("B") > 0,"Cabin"] = 1
    dat.loc[dat["Cabin"].str.count("C") > 0,"Cabin"] = 2
    dat.loc[dat["Cabin"].str.count("D") > 0,"Cabin"] = 3
    dat.loc[dat["Cabin"].str.count("E") > 0,"Cabin"] = 4
    dat.loc[dat["Cabin"].str.count("F") > 0,"Cabin"] = 5
    dat.loc[dat["Cabin"].str.count("G") > 0,"Cabin"] = 6
    dat.loc[~dat["Cabin"].isin([-1,0,1,2,3,4,5,6]), "Cabin"] = 7
    
    dat.drop(drop_column, axis=1, inplace = True)


# ## Split data into training and testing

# In[ ]:


x = data_cleaner[0].iloc[:,1:]
y = data_cleaner[0].iloc[:,0:1]


# In[ ]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.0, random_state=42)


# In[ ]:


X_train.shape
X_train = X_train.astype(float)
y_train = (y_train.astype(float))


# In[ ]:


print(y_train.shape)
print(X_train.shape)


# # XGBoost - Feature importance

# In[ ]:


from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import accuracy_score

model = XGBClassifier()
model.fit(X_train,y_train.values.ravel())


# In[ ]:


plot_importance(model)


# As a guideline, top scored features were selected as input of DNN.

# In[ ]:


# y_pd = model.predict(data_cleaner[1])

# test_dat  = pd.read_csv("../input/test.csv")
# a = pd.Series(test_dat["PassengerId"], name='PassengerId')  
# b = pd.Series(y_pd.astype(int), name='Survived')  

# save = pd.DataFrame({'PassengerId':a,'Survived':b})  
# save.to_csv("../working/submission.csv", index=False)
# pd.DataFrame({'PassengerId':a,'Survived':b})


# ## DNN Model

# In[ ]:


import tensorflow as tf
import keras.layers as l
from keras.models import Model
import keras.optimizers as Opt
from keras import callbacks
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint


# In[ ]:


def DModel():

    x = l.Input(shape=(24,))
#     m = l.Dense(800,activation="relu")(x)
#     m = l.Dropout(0.5)(m)

    
#     m = l.Dropout(0.5)(m)
    m = l.Dense(50,activation="relu",kernel_initializer="glorot_uniform")(x)
    m = l.Dense(200,activation="relu",kernel_initializer="glorot_uniform")(m)
    m = l.Dense(200,activation="relu",kernel_initializer="glorot_uniform")(m)
    m = l.Dense(50,activation="relu",kernel_initializer="glorot_uniform")(m)
    m = l.Dense(50,activation="relu",kernel_initializer="glorot_uniform")(m)
    
    
#     m = l.Dropout(0.5)(m)
#     m = l.Dense(400,activation="relu",kernel_initializer="glorot_uniform")(m)
#     m = l.Dropout(0.5)(m)
#     m = l.Dense(800,activation="relu",kernel_initializer="glorot_uniform")(m)

    out = l.Dense(1,activation="sigmoid",kernel_initializer="glorot_uniform")(m)
    model = Model(inputs=x, outputs=out)
#     opt = Opt.Adam(lr=0.002, beta_1 =0.9, beta_2 = 0.999, decay=0.0001)
    opt = Opt.adadelta(lr=0.1, rho=0.95, epsilon=None, decay=0.0001)
    model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])
    return model


# In[ ]:


md = DModel()
earlyStopping=callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto')
# md.fit(X_train,y_train,epochs=30,callbacks=[earlyStopping],validation_split=0.1)
filepath="../working/weights-improvement-top.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = md.fit(X_train,y_train,epochs=100,shuffle=False, callbacks=[checkpoint],validation_split=0.1)


# In[ ]:


# ==============================================
idk = "val_acc"
va = max(history.history[idk]) if idk != "val_loss" else min(history.history[idk])
index = [i for i, j in enumerate(history.history[idk]) if j == va]
fig,ax = plt.subplots() 
plt.xlabel('Steps')  
plt.ylabel('Loss')
plt.grid(True) 
ax.set_ylim([0,1])
plt.plot(history.history["val_loss"], label="val_loss", linewidth=2.0)
plt.plot(history.history["loss"], label="loss", linewidth=2.0)
plt.plot(history.history["val_acc"], label="val_acc", linewidth=2.0)
plt.plot(history.history["acc"], label="acc", linewidth=2.0)
plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0.) 
plt.text(0, 0.5, "acc:       %.2f       loss:    %.2f\nval_acc: %.2f    val_loss: %.2f" % (history.history["acc"][-1],history.history["loss"][-1],history.history["val_acc"][-1],history.history["val_loss"][-1]), fontsize=20)
plt.text(0, 0.1, "at_%s: %.2f (epoch-%d)" % (idk,va,index[0]), fontsize=20)
plt.axvspan(index[0], index[0], color='red', alpha=1)
plt.savefig("training.png")


# In[ ]:


md.load_weights("../working/weights-improvement-top.hdf5")


# In[ ]:


rst = (md.predict(data_cleaner[1]))
rta = []
for t in rst:
    rta.append(int(round(t[0])))
test_dat  = pd.read_csv("../input/test.csv")
a = pd.Series(test_dat["PassengerId"], name='PassengerId')  
b = pd.Series(rta, name='Survived')  

save = pd.DataFrame({'PassengerId':a,'Survived':b})  
save.to_csv("../working/submission.csv", index=False)
pd.DataFrame({'PassengerId':a,'Survived':b})

