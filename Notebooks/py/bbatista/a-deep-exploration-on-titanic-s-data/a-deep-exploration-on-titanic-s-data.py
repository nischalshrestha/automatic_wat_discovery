#!/usr/bin/env python
# coding: utf-8

#  
# # EDA on Titanics dataset! 
#   
#   My 5th Kernel takes a dive deep on Titanic's Data, for discover some answers about what influenced more the chances of surviving the disaster. 
#    
#   In this simple kernel i also would like to share with all of you some new things i learned in my journey to become a data scientist. And as always i hope that you like it and in that case, please leave your comment or upvote.
#    
#    Thank you !
#    
#    
#  
# 

# In[ ]:


# import

import pandas as pd   # data manipulation and preparation
import numpy as np    # maths operation
import seaborn as sns # some really informative graphs

sns.set_style('whitegrid')

get_ipython().magic(u'matplotlib inline')

import matplotlib.pyplot as plt # gráficos


# In[ ]:


# importing the data

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


# first ten lines

df_train.head(10)


# In[ ]:


df_train.shape


# In[ ]:


df_train.describe()


# In[ ]:


# a closer look
df_train.info()


# In[ ]:


# null values

columns = df_train.columns

for col in columns:
    perc = df_train[col].isnull().sum()/len(df_train)
    
    print("The column '{}' has {} NaN values. Or {}% rows.\n".format(col, df_train[col].isnull().sum()
                                                                                                   ,perc * 100))
            


# In[ ]:



import missingno as msno 
msno.matrix(df_train); #'shows' the NaN values. white spaces represents NaN values


# ****## Dataset info:
# 
# *  It has 891 rows and 12 columns; 
# *  Columns Age, Cabin and Embarked need some cleaning; 
# *  transforming the columns name in lowercase 
# might be a nice idea...
# 
# ### Cleaning the dataset:
# 
# * age: We can fill this column with some measure of central tendency (average), considering that this is relevant information for our analysis.
# 
# * Cabin: This column contains the passenger cabin number. At first it does not seem to be relevant data for the analysis, so I believe the best approach is to exclude this column;
# 
# * Embarked: Represents the passenger's port of embarkation. It would be interesting to ask some questions about this data. Since we only have two missing values, it will be easy to fill.
# 
# ### Transformations to be made:
# 
# * Age: From float to int;
# 
# 
# * Name: Try to extract the title of the passengers (Mr., Miss., And etc);
# 
# * Ticket: Probably will be excluded, it does not appear to contain data relevant for exploratory analysis purposes (what the ticket number could say about these people?);
# 
# * Embarked: Change names to: C = Cherbourg, Q = Queenstown, S = Southampton.
# 
# ** I believe that these changes will leave our data set clean and much clearer for the analysis. **

# In[ ]:


cols = df_train.columns 
cols = [col.lower() for col in cols] # adjusting cols
df_train.columns = cols # changing the columns
df_train.columns # confirm


# In[ ]:


# filling 'age'

df_train.age.describe() 


# In[ ]:


# filling with the mean
age_mean = df_train.age.mean()
df_train.age.fillna(age_mean, inplace=True)

# from float to int

df_train.age = [int(x) for x in df_train.age]

df_train.age.describe()


# In[ ]:


# taking a look on the hist
plt.figure(figsize=(12,10))
plt.title("Age distribution")
sns.distplot(df_train.age);


# In[ ]:


# is there any NaN

print("'Age' col has {} NaN values.".format(df_train.age.isnull().sum()))


# In[ ]:



print("The dtype of 'age' is {}.".format(df_train.age.dtypes))


# In[ ]:


# droping 'cabin'
df_train.drop(axis=1, inplace=True, columns='cabin')


# In[ ]:


for i,j in enumerate(df_train.columns):
    print(i, j)


# In[ ]:


# filling 'embarked' with the most commom value
df_train.embarked.value_counts()


# In[ ]:


df_train.embarked.fillna('S', inplace=True)
df_train.embarked.isnull().sum()


# In[ ]:


# Embarked descriptions 

df_train.embarked = df_train.embarked.replace(['S', 'C', 'Q'],['Southampton', 'Cherbourg', 'Queenstown'])
df_train.embarked.value_counts()


# In[ ]:


# droping 'ticket'
df_train.drop(axis=1, columns= 'ticket', inplace=True)
for i,j in enumerate(df_train.columns):
    print(i, j)


# In[ ]:


# Lets take a new look
msno.matrix(df_train);


# In[ ]:


columns = df_train.columns

for col in columns:
    perc = df_train[col].isnull().sum()/len(df_train)
    
    print("The column '{}' has {} NaN values. Or {}% rows.\n".format(col, df_train[col].isnull().sum()
                                                                                                   ,perc * 100))


# Looks much better now...
# 
# Lets work on the 'titles'

# In[ ]:



df_train.name.head()


# In[ ]:


# Using lambda function to break the string between the '.' and a ',' and extract the title.
titulos = df_train.name.apply(lambda x: x.split('.')[0].split(',')[1])


# In[ ]:


titulos.value_counts()


# In[ ]:


# creating 'title' column 
df_train['title'] = titulos
print(df_train.columns)


# In[ ]:


df_train.title.head()


# In[ ]:


# Done, now we can create a new column, called
# 'age_range' that will represent the age range of each passenger

labels = ['young', 'adults', 'elder'] # ranges name

age_range = pd.cut(df_train.age, 3, labels=labels) # cuting the data
age_range.head()


# In[ ]:


# Incluindo a coluna no dataset
df_train['age_range'] = age_range


# In[ ]:


df_train.age_range.value_counts()


# In[ ]:


df_train.groupby('age_range')['age'].describe()


# By the mean age values by age group, the division into three categories makes a lot of sense.

# ### Sibsp e parch
# 
# According to the description of the [data dictionary] (https://www.kaggle.com/c/titanic/data) I get the impression that the main information that can be extracted from the sibsp and e parch columns is if the passenger was alone , accompanied (couple) or in family (more than 2 people). So it costs nothing to create another variable.
# 
# cia: indicates whether passengers are alone, in pairs or in groups (families)
# 
# 

# In[ ]:


cia = df_train.sibsp.replace([2,3,4,5], 'group')


# In[ ]:


cia = cia.replace([0, 1, 8], ['alone', 'double', 'group'])
cia.value_counts()


# In[ ]:


df_train['cia'] = cia


# In[ ]:


# making the names better to read
nome_sobre = df_train.name.apply(lambda x: x.split(',')[1]) + " " + df_train.name.apply(lambda x: x.split(',')[0]) 
df_train.name = nome_sobre



# ### It appears that we have finished cleaning and processing the data. Let's take one last check. ###
# 

# In[ ]:


df_train.info()


# ## Exploratory data analysis
# 
# Now we have data on almost 900 passengers of the Titanic. Let's try to find out some interesting information!
# 
# * How many people survived?
# * What is the survival rate by age group, sex, class, port of embarkation, company and title?
# * What is the average cost of passing class popup?
# * Where did most of the passengers come from?
# * Being accompanied helped in survival?
# 

# In[ ]:



df_train.describe(include='all')


# The crew was made up of people around the age of 30. Most were male. Most of them traveled alone and boarded Southampton. What draws attention is variation of the price of the ticket, the average price is 32 but can reach up to 512. Who will pay so dearly? Or is it a data collection error?
# 
# It's worth investigating.

# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(10,8))
sns.boxplot(df_train.fare);

plt.title('Ditribution of the Fare');


# In[ ]:


# There are many outliers
outliers = df_train[df_train['fare'] >= 55]
outliers


# Three very wealthy passengers paid 512 to board the Titanic. Curiously, the three survived. Does this suggest that the affluent had priority?
# 
# It is also curious that some people who traveled in the 3rd class paid such high prices.
# 
# Let's take a closer look.

# In[ ]:


outliers[outliers['pclass'] == 3]


# This is at least curious. The names of people who paid dearly for traveling in the 3rd class follow a pattern. They seem to be from a specific family, the Sage, and foreign (Chinese?)
# 
# Is it a collection error ?
# 
# Nice !
# 
# Let's look at the other aspects of the original dataset:

# * Survivors

# In[ ]:


sobreviventes = df_train[df_train['survived'] == 1]
print('The survivor rate was {}'.format(len(sobreviventes) / len(df_train)))
                                                            
plt.figure(figsize=(12,10))
sns.set_style('whitegrid')

sns.countplot(x = 'survived', data=df_train);
plt.title('Survivors');


# * Gender

# In[ ]:


hm = df_train[df_train['sex']== 'male']

ml = df_train[df_train['sex']== 'female']

hm_survived = df_train[(df_train['sex']== 'male') &
                         (df_train['survived'] == 1)]

ml_survived = df_train[(df_train['sex']== 'female') &
                         (df_train['survived'] == 1)]
                         
                          
print('Men: {}\nWomen: {}\n'.format(len(hm), len(ml)))
print('Men survivor rate: {}\nWomen survicor rate: {}'.format(len(hm_survived)/len(hm), len(ml_survived)
                                                                                                                  / len(ml)))
                                                            
plt.figure(figsize=(10,8))
sns.set_style('whitegrid')

sns.countplot(x = 'sex', data=df_train, hue='survived');
plt.title('Gender');


# * Age range

# In[ ]:


adults = df_train[df_train['age_range']== 'adults']

young = df_train[df_train['age_range']== 'young']

elder = df_train[df_train['age_range']== 'elder']

adult_survided = df_train[(df_train['age_range']== 'adults') &
                         (df_train['survived'] == 1)]

young_survived = df_train[(df_train['age_range']== 'young') &
                         (df_train['survived'] == 1)]

elder_survived = df_train[(df_train['age_range']== 'elder') &
                         (df_train['survived'] == 1)]
                         
                          
print('Youngs: {}\nAdults: {}\nAged: {}\n'.format(len(young), len(adults), len(elder)))
print('Youngs survivor rate: {}\nAdults survivor rate: {}\nAged survivor rate: {}'.format(len(young_survived)/len(young), 
                                                                                                                    len(adult_survided)/len(adults),
                                                                                                                    len(elder_survived)/ len(elder)))
                                                            
plt.figure(figsize=(10,8))
sns.set_style('whitegrid')

sns.countplot(x = 'age_range', data=df_train, hue='survived');
plt.title('Age range');


# * Class

# In[ ]:


primeira = df_train[df_train['pclass']== 1]

segunda = df_train[df_train['pclass']== 2]

terceira = df_train[df_train['pclass']== 3]

pr_survived = df_train[(df_train['pclass']== 1) &
                         (df_train['survived'] == 1)]

seg_survived = df_train[(df_train['pclass']== 2) &
                         (df_train['survived'] == 1)]

ter_survived = df_train[(df_train['pclass']== 3) &
                         (df_train['survived'] == 1)]
                         
                          
print('1st class: {}\n2nd class: {}\n3rd Class: {}\n'
      .format(len(primeira), len(segunda), len(terceira)))
print('1st class survivor rate: {}\n2nd survivor class rate: {}\n3rd survivor rate: {}'.format(len(pr_survived)/len(primeira), 
                                                                                                                    len(seg_survived)/len(segunda),
                                                                                                                    len(ter_survived)/ len(terceira)))
                                                            
plt.figure(figsize=(10,8))
sns.set_style('whitegrid')

sns.countplot(x = 'pclass', data=df_train, hue='survived');
plt.title('Class');


# * Embarked

# In[ ]:


south = df_train[df_train['embarked']== 'Southampton']

cherb = df_train[df_train['embarked']== 'Cherbourg']

queen = df_train[df_train['embarked']== 'Queenstown']

south_survived = df_train[(df_train['embarked']== 'Southampton') &
                         (df_train['survived'] == 1)]

cherb_survived = df_train[(df_train['embarked']== 'Cherbourg') &
                         (df_train['survived'] == 1)]

queen_survived = df_train[(df_train['embarked']== 'Queenstown') &
                         (df_train['survived'] == 1)]
                         
                          
print('Southampton: {}\nCherbourg: {}\nQueenstown: {}\n'.format(len(south), len(cherb), len(queen)))
print('Southampton survivor rate: {}\nCherbourg survivor rate: {}\nQueenstown survivor rate: {}'
      .format(len(south_survived)/len(south), len(cherb_survived)/len(cherb),len(queen_survived)/ len(queen)))
                                                            
plt.figure(figsize=(10,8))
sns.set_style('whitegrid')

sns.countplot(x = 'embarked', data=df_train, hue='survived');
plt.title('Embarked');


# * Cia

# In[ ]:


solo = df_train[df_train['cia']== 'alone']

dup = df_train[df_train['cia']== 'double']

grup = df_train[df_train['cia']== 'group']

solo_survived = df_train[(df_train['cia']== 'solo') &
                         (df_train['survived'] == 1)]

dup_survived = df_train[(df_train['cia']== 'double') &
                         (df_train['survived'] == 1)]

grup_survived = df_train[(df_train['cia']== 'group') &
                         (df_train['survived'] == 1)]
                         
                          
print('Lone travelers: {}\nDouble: {}\nGroup: {}\n'.format(len(solo), len(dup), len(grup)))
print('Lone traveler survivor rate: {}\nDouble survivor rate: {}\nGroup survivor rate: {}'
      .format(len(solo_survived)/len(solo), len(dup_survived)/len(dup),len(grup_survived)/ len(grup)))
                                                            
plt.figure(figsize=(10,8))
sns.set_style('whitegrid')

sns.countplot(x = 'cia', data=df_train, hue='survived');
plt.title('Cia');


# In[ ]:


plt.figure(figsize=(12,10))
sns.countplot(x = 'title', data=df_train, hue='survived');
plt.title('Título')
plt.xticks(rotation=45);


# From these points of view, we can see that most of the survivors were women, first-class passengers. Passengers from the port of Cherbourg had the highest survival rate among the ports. Have young people been more fortunate, or did their youthful vigor help them fight for their lives or in this group do we have many children who would have a preference in lifeboats?
# 
# Petitioners had higher survival rates.
# 
# Let's try to better understand these results. Starting at the port of cherborg, which had a higher survival rate.

# In[ ]:


print(df_train['embarked'].value_counts())

print('\n')

print(pd.crosstab(df_train['embarked'], df_train['pclass'], margins=True, margins_name='Total'))

print('\n')

print(pd.crosstab(df_train['embarked'], df_train['sex'], margins=True, margins_name='Total'))

print('\n')

print(pd.crosstab(df_train['embarked'], df_train['age_range'], margins=True, margins_name='Total'))

print('\n')

print(pd.crosstab(df_train['embarked'], df_train['cia'], margins=True, margins_name='Total'))

print('\n')

print(pd.crosstab(df_train['embarked'], df_train['title'], margins=True, margins_name='Total'))


# Based on these comparisons, we can infer that cherborg has the highest survival rate for the following reasons:
# 
# * It has the lowest proportion of passengers in the third class and as we saw in the graphs the third class have a survival rate quite inferior to the others;
# 
# * Half of the people who boarded Cherbourg are first class;
# 
# * It has the highest proportion of people traveling alone, once again the initial analysis had already shown us that people traveling alone has a higher survival rate;
# 
# * In this analysis we can also see why Southampton has the worst survival rate. 55% of the passengers were third-class passengers, 68% of men (who had a lower survival rate than women).
# 
# ### Now we can explore some of the value of the passages:
# 

# In[ ]:


plt.figure(figsize=(10,8))
sns.distplot(df_train.fare);
plt.title('Passagens distribution');


# In[ ]:


df_train.groupby('pclass')['fare'].mean().plot(kind='bar', figsize=(10,8), title = "Average Ticket Value per Class");


# In[ ]:


df_train.groupby('sex')['fare'].mean().plot(kind='bar', figsize=(10,8), title = "Average Passage Value By Gender");


# Interesting. Women paid an average ticket much higher than men. Is it because there are more women in the first class?
# 
# We will see:

# In[ ]:


pd.crosstab(df_train['pclass'], df_train['sex'], margins =  True, margins_name= 'Total')


# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(x = df_train['sex'], hue=df_train['pclass']);
plt.title('Proportion of men and women by class');


# The distribution by classes is very similar, so some other reason should explain the value so different from the average price of the passage between men and women.

# In[ ]:


df_train.groupby('age_range')['fare'].mean().plot(kind='bar', figsize=(10,8), title = "Average Ticket Value Age Range");


# Seniors pay more to travel, probably because they are more affluent and prefer more comfort.

# In[ ]:


pd.crosstab(df_train['age_range'], df_train['pclass'], margins =  True, margins_name= 'Total')


# Most of the seniors traveled first in the second class. This may explain the higher cost of passage.

# In[ ]:


# let's see if there is any correlation with passage value and age
plt.figure(figsize=(10,8))
sns.regplot(x = 'age', y = 'fare', data=df_train)
plt.title('Relação entre idade e valor da passagem');


# If there is any relationship it is clearly very weak.

# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot(x = 'age_range', y = 'fare', data=df_train, hue='survived')
plt.title('Age range and ticket value ratio');


# Once again it is clear that whoever paid more, had more chances of survival. For he paid dearly for a first class passage, whose facilities were to be nearer to the boats.

# In[ ]:


df_train.groupby('survived')['fare'].mean().plot(kind='bar', figsize=(10,8), title = "Average value of survival tickets");


# In[ ]:


df_train.groupby('embarked')['fare'].mean().plot(kind='bar', figsize=(10,8), title = "Average ticket price per port");


# Cherbourg appears with the highest average. It makes sense, the port with more single travelers, higher proportion of first class travelers, higher average ticket value = higher survival rate.
# 
# Southampton is in second place because although most of its passengers are second class, it is the port with the most embarkers and 45% of them are in second or first class.
# 
# Queenstown had only 77 people and most of its 3rd class.

# In[ ]:


embark = pd.crosstab(df_train['embarked'], df_train['pclass'], margins=True, margins_name='Total')


# In[ ]:


embark.plot(kind='bar', figsize=(10,8));


# In[ ]:


sex = pd.crosstab(df_train['embarked'], df_train['sex'], margins=True, margins_name='Total')
sex.plot(kind='bar', figsize=(10,8));


# ## What we have discovered so far
# 
# So far we have found that the groups with the greatest chance of survival are women, young people and members of the first class.
# 
# It is clear in the course of the analysis that the richest people had a greater chance of survival.
# 
# To better understand what influenced the survival of these groups, we will delve deeper into analysis. To do this, I will divide the dataset into 3 new subsets:
# 
# * Only young people: Has the maxim 'Women and children first' been respected? To find out I will divide this new set of data into new age groups;
# 
# * Richer Passengers: Remember the outliers we saw there? They paid dearly, did not they? If my assumption is correct (that the one who paid the most was more likely to survive), it is probably probable that the survival rate of this group is much higher than that of the whole as a whole
# 
# * Lonely travelers: We have seen that lonely passengers had the highest number of deaths. Let's focus specifically on them and try to find some clue as to what has influenced that behavior.

# # Modeling
# 
# This is my first time using machine learning with some real understanding about how the things must be done, and how the algorithms work ! So, please, if anyone can help with tips i will be very happy ! 
# 
#  I will not use the following columns because they were transformed up there:  age , sibsp and parch
# 
# 

# In[ ]:


df_modelo = df_train

colunas_to_drop = ['passengerid', 'name', 'age', 'sibsp'] # testing without the age column (using the age_range instead)
                                                         # testing without sibsp and parch (using cia column instead)
    
df_modelo.drop(colunas_to_drop, axis = 1, inplace = True)


# In[ ]:


df_modelo.head()


# In[ ]:


# using "get_dummies" 

df_modelo = pd.get_dummies(df_modelo)


# In[ ]:


df_modelo.head()


# ## Train and test split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df_modelo.drop('survived', axis=1) # Predictor variables
y = df_modelo['survived']              # Target variable


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[ ]:


# importing the Decision Tree algorithm
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(random_state=12, criterion="entropy") # Instanciando o algoritmo


# In[ ]:


dtree.fit(X_train, y_train) # Training the model


# In[ ]:


predictions = dtree.predict(X_test) # Making the predictions


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix # Importing Performance Evaluation Tools


# In[ ]:


target_names = ['morreu', 'viveu']

resultados_dtree = classification_report(y_test, predictions, target_names= target_names) # Saving the results

print(resultados_dtree)


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 100, random_state=12)


# In[ ]:


rfc.fit(X_train, y_train)


# In[ ]:


rfc_pred = rfc.predict(X_test)


# In[ ]:


resultado_rfc = classification_report(y_test, rfc_pred, target_names=target_names)

print('\n')

print(resultado_rfc)


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression(random_state=12)


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


pred_logreg = logmodel.predict(X_test)


# In[ ]:


resultado_logreg = classification_report(y_test, pred_logreg, target_names=target_names)

print(resultado_logreg)


# ### Well, i think its not so bad for a beginner ! I will keep improving ! Thanks !

# In[ ]:




