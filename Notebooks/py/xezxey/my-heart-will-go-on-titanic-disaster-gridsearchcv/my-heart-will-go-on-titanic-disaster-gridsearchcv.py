#!/usr/bin/env python
# coding: utf-8

# # Titanic Disaster
# ![RMS Titanic](http://irishamerica.com/wp-content/uploads/2012/03/Titanic-at-Southhampton-docks.jpg)
# 
# **History** : 
# RMS Titanic is a largest ship afloat at a time(est. 1911). Titanic was under command by Edward Smith who also went down with the ship. Titanic start the journey from Southampton -> Cherbough -> Queenstown -> America. RMS Titanic hit an iceberg at 11.50 pm and sunk in 2.30 hours at 2.20 am.
# Titanic contains 2,435 passengers and 892 crews. Total is 3,327 lives (or 3,547 according to other sources).
# ***
# *Titanic Route Map*
# ![RMS Titanic Route Map Picture](https://localtvkstu.files.wordpress.com/2012/04/04-13-titanic-4.jpg?quality=85&strip=all)
# ***
# [More Titanic Information By Wikipedia](https://en.wikipedia.org/wiki/RMS_Titanic)
# 
# 

# # Dive into the data
# ***
# **Features** 
# 1. Survival : 1 = Survive, 0 = Not Survive
# 2. Pclass : Ticket class of passengers (1, 2 and 3)
# 3. Sex : Male and Female
# 4. Age : Age in years
# 5. Sibsp : # of siblings / spouses aboard the Titanic
# 6. Parch : # of parents / children aboard the Titanic
# 7. Ticket : Ticket number
# 8. Fare : Passenger fare
# 9. Cabin : Cabin number
# 10. Embarked : C = Cherboug, Q = Queenstown, S = Southampton
# ***
# NaN is Not a Number.
# 

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import data sets
titanic_train_dataset = pd.read_csv("../input/train.csv")
titanic_test_dataset = pd.read_csv("../input/test.csv")
titanic_submission_form_dataset = pd.read_csv("../input/gender_submission.csv")

#Preview the data
titanic_train_dataset.head(n=5)
#By running .info method you will see some missing value in ['Age'] and ['Cabin'] Columns than we will clean this in the next part
titanic_train_dataset.info()


# # Part I : Data Cleaning
# **Goal : Cleaning some missing data**
# From previous part, you can see the missing value from 2 columns
# 1. Age : has 714 non-null float64 ------> from 814 records 
# 
#     *     First attempt : Find the regression model for predicting missing age. The regressors come from correlation between age and each feature. 
#     *     Second attempt : I will use mean and std to fill missing values by random value in range [mean-std, mean+std].
#     
# 2. Cabin : has 204 non-null object ------> from 814 records
# 
#     *       I will assume that the 3rd class passengers will stay in any cabin of floor F or G by randoming.
#       
#     *       ** Note : In fact, only 1st and 2nd class passengers will have a cabin room number. So the other records make me assume that they 're 3rd class passengers. So it will effect to the CabinFloorScore and probability to survive too because 3rd class passengers will stay in only F or G floors. So the point that i want to explain is Cabin columns doesnt contain the missing value but in Titanic 3rd class passengers have no room, no cabin to stay. **
#       

# In[13]:


#Filling some missing data
#Age
mean_age = titanic_train_dataset['Age'].mean()
std_age = titanic_train_dataset['Age'].std()
titanic_train_dataset['Age'] = titanic_train_dataset['Age'].fillna(np.random.randint(low = mean_age - std_age, high = mean_age + std_age))
titanic_train_dataset[['Age']].describe()    #Re-Check that we have fill all of missing data
#print(titanic_train_dataset['Name'].loc[titanic_train_dataset['Age'] < 1])
titanic_train_dataset['CategoricalAge'] = pd.qcut(titanic_train_dataset['Age'], q = 4)

#CabinFloorScore
#Use regular expression to match floor pattern
#Set default floor for 3rd class ticket(GuestRoom in floor F and G)
import re
cabin_pattern = re.compile("[a-zA-Z]")
GuestRoom_random_floor = ["F", "G"]

cabin_floor_list = []
for cabin in titanic_train_dataset['Cabin']:
    if pd.isnull(cabin):
        cabin_floor_list.append(GuestRoom_random_floor[np.random.randint(2)])
    else:
        cabin_floor = re.findall(cabin_pattern, cabin)
        cabin_floor_list.append(min(cabin_floor))

titanic_train_dataset['Cabin'] = cabin_floor_list

titanic_train_dataset.info()
titanic_train_dataset.head(5)


# > > > 

# # Part II : Feature Engineer
# **GOAL : Creating new features from original data. Following below**
# 1. FamilySize 
# 2. IsAlone
# 3. CabinFloorScore
# 4. FarePerPerson
# 5. TitleScore

# **FamilySize : ** 
# Family size of passengers can be calculate by using following equation.
# > FamilySize = Sibsp + Parch + 1 > 
# 
# **IsAlone : **
# Alone passenger status can be find by using FamilySize Column
# > If FamilySize == 1 : Passenger is alone else : No
# 
# **CabinFloorScore : **
# ![CabinFloor](http://www.eyeopening.info/wp-content/uploads/building-titanic.jpg)
# 
# 
# From an evacuate plan, lifeboats will release to the sea from a deck. So the people are nearest to the deck will have more chance to survive. and i will give score like following this 
# > 
# * CabinFloor T (Special Cabin for First Class Ticket)  : Score = 7                                                                                                          
# * CabinFloor A                                                                   : Score = 6                                                                                                         
# * CabinFloor B                                                                   : Score = 5                                                                                                                
# * CabinFloor C                                                                   : Score = 4                                                                                                         
# * CabinFloor D                                                                   : Score = 3                                                                                                               
# * CabinFloor E                                                                   : Score = 2                                                                                                              
# * CabinFloor F                                                                   : Score = 1                                                                                                                  
# * CabinFloor G                                                                  : Score = 0                               
# 
# 
# **FarePerPerson : **
# FarePerPerson can be calculate by using following equation.
# > FarePerPerson = Fare / FamilySize
# 
# **TitleScore : **
# From Name Column, we can know that there's a lot of name title given. So each name title has different priority and different priority make the chance of survive not the same.
# **Note : Name title observes from training data only ===> Title that not found in training set will be set as default and score will give to 0
# > I will give each title a score  following these conditions :
# 1. Influence : Don., Col., Major., Capt., Sir., Mme., Lady., Countess.    => Score = 3
# 2. Adult : Mr., Mrs., Miss., MS., Mlle., Dr., Rev.,                                               => Score = 2
# 3. Child : Jonkheer, Master                                                                                        => Socre = 1
# 
# ***
# 

# In[14]:


#FamilySize
titanic_train_dataset['FamilySize'] = titanic_train_dataset['SibSp'] + titanic_train_dataset['Parch'] + 1

#IsAlone
titanic_train_dataset.loc[titanic_train_dataset['FamilySize'] > 1, 'IsAlone'] = 0
titanic_train_dataset.loc[titanic_train_dataset['FamilySize'] == 1, 'IsAlone'] = 1

#FarePerPerson
titanic_train_dataset['FarePerPerson'] = titanic_train_dataset['Fare'] / titanic_train_dataset['FamilySize']
titanic_train_dataset['CategoricalFarePerPerson'] = pd.qcut(titanic_train_dataset['FarePerPerson'], q = 4)

#CabinFloorScore
#Use regular expression to match floor pattern
#Set default floor for 3rd class ticket(GuestRoom in floor F and G)
import re
cabin_pattern = re.compile("[a-zA-Z]")
GuestRoom_random_floor = ["F", "G"]

cabin_floor_list = []
for cabin in titanic_train_dataset['Cabin']:
    if pd.isnull(cabin):
        cabin_floor_list.append(GuestRoom_random_floor[np.random.randint(2)])
    else:
        cabin_floor = re.findall(cabin_pattern, cabin)
        cabin_floor_list.append(min(cabin_floor))

titanic_train_dataset['CabinFloor'] = cabin_floor_list

#Giving a score for each cabin floor.
titanic_train_dataset["CabinFloorScore"] = 0
titanic_train_dataset["CabinFloorScore"].loc[titanic_train_dataset['CabinFloor'] == "T"] = 7 
titanic_train_dataset["CabinFloorScore"].loc[titanic_train_dataset['CabinFloor'] == "A"] = 6
titanic_train_dataset["CabinFloorScore"].loc[titanic_train_dataset['CabinFloor'] == "B"] = 5
titanic_train_dataset["CabinFloorScore"].loc[titanic_train_dataset['CabinFloor'] == "C"] = 4
titanic_train_dataset["CabinFloorScore"].loc[titanic_train_dataset['CabinFloor'] == "D"] = 3
titanic_train_dataset["CabinFloorScore"].loc[titanic_train_dataset['CabinFloor'] == "E"] = 2
titanic_train_dataset["CabinFloorScore"].loc[titanic_train_dataset['CabinFloor'] == "F"] = 1
titanic_train_dataset["CabinFloorScore"].loc[titanic_train_dataset['CabinFloor'] == "G"] = 0

#TitleScore
name_title = []
title_pattern = re.compile("[a-zA-Z]{1,}\.")
for name in titanic_train_dataset['Name']:
    name_title.append(re.findall(title_pattern, name)[0])
    
titanic_train_dataset['Title'] = name_title
#print(set(titanic_train_dataset['Title']))
                                                                                            
#We will give each title a score 
#1.Have power : Don., Col., Major., Capt., Sir., Mme., Lady., Countess.    => Score = 3
#2.Adult : Mr., Mrs., Miss., MS., Mlle., Dr., Rev.,                        => Score = 2
#3.Child : Jonkheer, Master                                                => Socre = 1

prior1_title_influence = ['Don.', 'Col.', 'Major.', 'Capt.', 'Sir.', 'Mme.', 'Lady.', 'Countess.']
prior2_title_adult = ['Mr.', 'Miss.', 'Ms.', 'Mlle.', 'Mrs.', 'Dr.', 'Rev.']
prior3_title_kid = ['Jonkheer.', 'Master.']

titanic_train_dataset["TitleScore"] = 0
titleScore_list = []

def give_title_score(title):
    if title in prior1_title_influence:
        return 3
    elif title in prior2_title_adult: 
        return 2
    elif title in prior3_title_kid: 
        return 1
    else: 
        return 0

for i in range(len(titanic_train_dataset)):
    titleScore_list.append(give_title_score(titanic_train_dataset['Title'][i]))

titanic_train_dataset['TitleScore'] = titleScore_list

#Try to use the linear regression to predict missing age
plt.figure(9)
plt.scatter(titanic_train_dataset['Pclass'], titanic_train_dataset['Age'], c='red')
plt.xlabel('Pclass')
plt.ylabel('Age')

plt.figure(10)
plt.scatter(titanic_train_dataset['FarePerPerson'], titanic_train_dataset['Age'], c='blue')
plt.xlabel('FarePerPerson')
plt.ylabel('Age')

plt.figure(11)
plt.scatter(titanic_train_dataset['Sex'], titanic_train_dataset['Age'], c='green')
plt.xlabel('Sex')
plt.ylabel('Age')

plt.figure(12)
plt.scatter(titanic_train_dataset['IsAlone'], titanic_train_dataset['Age'], c='yellow')
plt.xlabel('IsAlone')
plt.ylabel('Age')

plt.figure(13)
plt.scatter(titanic_train_dataset['FamilySize'], titanic_train_dataset['Age'], c='black')
plt.xlabel('FamilySize')
plt.ylabel('Age')
#There're no correlation between [Age] feature and other features. So I will use second approach(Random between [mean-std, mean+std]).


# # Part III : Data Visualization
# ** Goal : Visualizing the data for presentation easier**
# 
# I will visualize the data to make you understand the data better and easier. Each features that I will visualize will be compare with 'Survival' columns.
# 
# ** Bar Plot : **
# Comapring each feature to survived rate show us how much impact it does.
#  1. Sex Vs. Survival
#  2. Age Vs. Survival : Since [Age] is interval value and can't use as input in plt.bar(). So i will change 
#  3. Pclass Vs. Survival
#  4. FamilySize Vs. Survival
#  5. IsAlone Vs. Survival
#  6. CabinFloorScore Vs. Survival
#  7. FarePerPerson Vs. Survival
#  8. TitleScore Vs. Survival
#  

# In[15]:


plt.rcParams["figure.figsize"] = (10, 7)
#1. Sex Vs. Survived
plt.figure(1)
sex_vs_survived = titanic_train_dataset[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
print(sex_vs_survived)
plt.bar(sex_vs_survived['Sex'], sex_vs_survived['Survived'], tick_label=sex_vs_survived['Sex'], width = 0.5)
plt.title('Sex Vs. Survived')
plt.xlabel('Sex')
plt.ylabel('Survived Rate')

#2. Age Vs. Survived
plt.figure(2)
age_vs_survived = titanic_train_dataset[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean()
age_vs_survived['AgeGroup'] = ['Child to Youth', 'Youth to Middle Aged', 'Middle Aged' ,'Middle Aged to Old']
print(age_vs_survived)
plt.bar(age_vs_survived['AgeGroup'], age_vs_survived['Survived'], width=0.5)
plt.title('Age Vs. Survived')
plt.ylabel('Survived Rate')
plt.xlabel('Age')

#3. Pclass Vs. Survived
plt.figure(3)
titanic_train_dataset[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
pclass_vs_survived = titanic_train_dataset[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
print(pclass_vs_survived)
plt.bar(pclass_vs_survived['Pclass'], pclass_vs_survived['Survived'], tick_label=pclass_vs_survived['Pclass'])
plt.title('Pclass Vs. Survived')
plt.ylabel('Survived Rate')
plt.xlabel('Pclass')

#4. FamilySize Vs. Survived
plt.figure(4)
famsize_vs_survived = titanic_train_dataset[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()
print(famsize_vs_survived)
plt.bar(famsize_vs_survived['FamilySize'], famsize_vs_survived['Survived'], tick_label=famsize_vs_survived['FamilySize'])
plt.title('FamilySize Vs. Survived')
plt.xlabel('FamilySize')
plt.ylabel('Survived Rate')
plt.plot(famsize_vs_survived['FamilySize'], famsize_vs_survived['Survived'], color='Red')

#5. IsAlone Vs. Survived
plt.figure(5)
isalone_vs_survived = titanic_train_dataset[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
print(isalone_vs_survived)
plt.bar(isalone_vs_survived['IsAlone'], isalone_vs_survived['Survived'], tick_label=isalone_vs_survived['IsAlone'])
plt.title('IsAlone Vs. Survived')
plt.xlabel('IsAlone')
plt.ylabel('Survived Rate')

#6. CabinFloorScore Vs. Survived
plt.figure(6)
cabinfloorscore_vs_survived = titanic_train_dataset[['CabinFloorScore', 'Survived']].groupby(['CabinFloorScore'], as_index=False).mean()
print(cabinfloorscore_vs_survived)
plt.bar(cabinfloorscore_vs_survived['CabinFloorScore'], cabinfloorscore_vs_survived['Survived'], tick_label=cabinfloorscore_vs_survived['CabinFloorScore'])
plt.title('CabinFloorScore Vs. Survived')
plt.xlabel('CabinFloorScore')
plt.ylabel('Survived Rate')

#7. FarePerPerson Vs. Survived
plt.figure(7)
fareperperson_vs_survived = titanic_train_dataset[['CategoricalFarePerPerson', 'Survived']].groupby(['CategoricalFarePerPerson'], as_index=False).mean()
fareperperson_vs_survived['TicketGrade'] = ['Very Cheap', 'Cheap', 'Moderate', 'Expensive']
print(fareperperson_vs_survived)
plt.bar(fareperperson_vs_survived['TicketGrade'], fareperperson_vs_survived['Survived'])
plt.title('FarePerPerson Vs. Survived')
plt.xlabel('FarePerPerson')
plt.ylabel('Survived Rate')

#8. TitleScore Vs. Survived
plt.figure(8)
titlescore_vs_survived = titanic_train_dataset[['TitleScore', 'Survived']].groupby(['TitleScore'], as_index=False).mean()
print(titlescore_vs_survived)
plt.bar(titlescore_vs_survived['TitleScore'], titlescore_vs_survived['Survived'], tick_label=['Influence', 'Adult', 'Kids'])
plt.title('TitleScore Vs. Survived')
plt.ylabel('Survived Rate')
plt.xlabel('Title')


# # ** Part III.I : Conclusion    ། ﹒︣ ‸ ﹒︣ །**
# 
# From each feature, you can see there're statistic signaficant impact to the survived rate and they follow to the evacuation plan(1st class, women and kids are first priority). 
# 1. Sex Vs. Survival
# > [Female] has more chance to survived than [Male] because of the evacuation plan.     
# 
# 2. Age Vs. Survival
# > [Middle Aged] has the most chance to survived at 40%. Following by [Child to Youth] with 38%. This feature still have impact from the evacuation plan that let kids to get help first.
# 
# 3. Pclass Vs. Survival
# > Absolutely that higher [Pclass] has more chance to survived because of 2 factor.
#     1. Cabin : Only 1st class and 2nd class will have own cabin number(private room). They stay in "floor T to floor E". The evacuate ship will deploy from the deck(top of the ship) so the higher floor you are the higher chance you survive.
#     2. Priority : High class people are always taken care by staffs. So this will increase the survive rate. 
# 
# 4. FamilySize Vs. Survival
# > [FamilySize] feature lead us to think in many different ways. From my hypothesis I will break down in many ways.
#     1. Too many cooks spoil the broth : You will see the most chance to survived is [FamilySize == 4] other will decrease either side(even many or less) like a "Normal Distribution Curve". So the number of people is a bit important.
#     ** In the other hand **
#     2. More people = More helpful & powerful : Just think you have someone to take care of you through the trouble. So this may increase the survived rate (Just insticnt!!!)
#     3. Old man & Kids : Big family maybe come from old people or many kids. So the survived rate will increase in this case
#     4. Big family = Probability to be a high class people and this will make some impact to survived rate from previous feature that we use.
#     
#     ** Note : There are so many way to think and every one impact will have some chaining to other impact too.** 
# 
# 5. IsAlone Vs. Survival
# > [IsAlone] is opposite to the [FamilySize]. Alone has more chance to survived than not alone.
# 
# 6. CabinFloorScore Vs. Survival
# > As I say from evacuation plan, the evacuate ship will deploy from the deck."Higher floor you are, Higher chance you survive.". So [CabinFloorScore] will impact to survived rate significally statistic.
# 
# 7. FarePerPerson Vs. Survival
# > More FarePerPerson = More you paid = More Class = More chance to survive => #MakeSense!!!
# 
# 8. TitleScore Vs. Survival
# > Title is one way to define your "Social Status". Some people have more influence than the others, and this will give your more chance to survive. Like a bias!!!. So you will see from correlation from the "TitleScore Vs. Survival" graph. The influence people and kids have around 50-60% to survived. By the way adults have only 30% to survive. This can show how much powerful that bias does.
# 
# # So we will try to use all of these feature to train several learning models and take the best models.

# # Part IV : Training and Testing
# ** Goal : Construct the models that can predict survival from unseen data **
# 
# Now we 're coming to the half of our journey. We have feature that can tell us what the survival should have. So I will use these set of features to train several models and comparing each other to find the best model, the best parameter with the best accuracy.
# 
# ** Training Model **
# > 
# 1. K-Nearest Neighbors
# 2. Decision Tree
# 3. Random Forest
# 4. Logistic Regression
# 5. Naive Bayes
# 
# ** Training Statregy **
# > 1. Grid Search : Algorithm to find the best hyperparameter for training model from list of parameter and applying k-fold cross validation for validate model performance.
# 
# ***
# ** Note : 1. K-fold cross validation is one way to validate model performance by splitting data set in to "K-fold" and "K-iteration". "You can setting size of k by changing {cv} parameter**
# ![K-fold cross validation](https://qph.fs.quoracdn.net/main-qimg-92a4fda85de5ac23353af74097eb6024-c)
# From the above figure, you can see in each iteration model is trained by different training set and tested by different test set. This can help us solving the "Variance accuracy". 
# 
# ** Variance accuracy mean that if you change test set then the accuracy will change too much. ** 
# ![Judge model on 1 test set is suck](https://image.ibb.co/mNkuKo/pablo.png)

# ** Preparing training set and test set **
# I will prepare the training set and test set that will be use in grid search for training models. I will separate the dataset into 2 parts.
# 1. Training Set : 90% of dataset
# 2. Test Set : 10% of dataset
# These will use to train, tuning and evaluate model by  gridsearch library.
# 
# > 
# ** Note : Why we need to create the function ?**
# 
# >  ** Ans : When we have new unseen dataset, we need to create features that have use in training step as a parameters. Unseen data has no parameters that we create from feature engineer step. This situation can occurs repeatedly. So We will refactoring code into function.**

# In[16]:


#Preparing dataset for training and testing 
#Creating the function names "create_feature" for create features from given data.
def create_feature(df, mode):
    """
    create_feature function 
    1. Input parameters : 
        1.1 df : Dataframe variable for input dataset that you want to create feature from it.
        1.2 mode : Mode selection (1 is for training dataframe, 0 for testing dataframe)
            Different between 2 mode is the selected columns
            - Training dataframe mode will return all of features include 'Survived' Columns
            - Testing dataframe mode will return only feature
    2. Returned value :
        2.1 df : return a dataframe after finish a creating feature step
    3. Function process : Take df and mode as input. Then create features and slice columns that we need to use for training and testing (depend on mode)
    """
    
    #Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['CategoryFamilySize'] = pd.cut(df['FamilySize'], bins=5)
    
    #IsAlone
    df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    
    #CabinFloor
    #Split cabin out of room number
    import re
    cabin_pattern = re.compile("[a-zA-Z]")
    GuestRoom_random_floor = ["F", "G"]
    
    cabin_floor_list = []
    #print(titanic_train_dataset["Name"].loc[titanic_train_dataset['Cabin'] > "O"])
    for cabin in df['Cabin']:
        if pd.isnull(cabin):
            cabin_floor_list.append(GuestRoom_random_floor[np.random.randint(2)])
        else:
            cabin_floor = re.findall(cabin_pattern, cabin)
            cabin_floor_list.append(min(cabin_floor))
    
    df['CabinFloor'] = cabin_floor_list
    
    #Score for each cabin floor
    df["CabinFloorScore"] = 0
    df["CabinFloorScore"].loc[df['CabinFloor'] == "T"] = 7
    df["CabinFloorScore"].loc[df['CabinFloor'] == "A"] = 6
    df["CabinFloorScore"].loc[df['CabinFloor'] == "B"] = 5
    df["CabinFloorScore"].loc[df['CabinFloor'] == "C"] = 4
    df["CabinFloorScore"].loc[df['CabinFloor'] == "D"] = 3
    df["CabinFloorScore"].loc[df['CabinFloor'] == "E"] = 2
    df["CabinFloorScore"].loc[df['CabinFloor'] == "F"] = 1
    df["CabinFloorScore"].loc[df['CabinFloor'] == "G"] = 0
    
    #Find how many passenger in each floor
    cabin_floor_list = ['T', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
    #for i in range(len(cabin_floor_list)):
    #    print("Cabin " + cabin_floor_list[i] + " : " + str(df["CabinFloorScore"].loc[df['CabinFloor'] == cabin_floor_list[i]].count()))
    
    #In test set : There's missing [fare] value as NaN. So I will fill this with mean
    df['Fare'] = df['Fare'].fillna(np.mean(df['Fare']))   
    
    #Fare per person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['CategoricalFarePerPerson'] = pd.qcut(df['FarePerPerson'], q = 4)
    

    
    
    #VIP
    #Impact
    name_title = []
    title_pattern = re.compile("[a-zA-Z]{1,}\.")
    for name in df['Name']:
        name_title.append(re.findall(title_pattern, name)[0])
        
    df['Title'] = name_title
                                                                                                
    #We will give each title a score 
    #1.Have power : Don., Col., Major., Capt., Sir., Mme., Lady., Countess.    => Score = 3
    #2.Adult : Mr., Mrs., Miss., MS., Mlle., Dr., Rev.,                        => Score = 2
    #3.Child : Jonkheer, Master                                                => Socre = 1
    
    prior1_title_powerful = ['Don.', 'Col.', 'Major.', 'Capt.', 'Sir.', 'Mme.', 'Lady.', 'Countess.']
    prior2_title_adult = ['Mr.', 'Miss.', 'Ms.', 'Mlle.', 'Mrs.', 'Dr.', 'Rev.']
    prior3_title_kid = ['Jonkheer.', 'Master.']
    
    df["TitleScore"] = 0
    TitleScore_list = []
    
    def give_title_score(title):
        if title in prior1_title_powerful:
            return 3
        elif title in prior2_title_adult:
            return 2
        elif title in prior3_title_kid : 
            return 1
        else: 
            return 0
    
    for i in range(len(df.index)):
        TitleScore_list.append(give_title_score(df['Title'][i]))
    
    df['TitleScore'] = TitleScore_list    
    
    #Age
    #No correlation between each features and age   
    #Age have some missing values ---> Use mean, median or std to fill the nan
    #Random between [mean-std, mean+std]
    mean_age = df['Age'].mean()
    std_age = df['Age'].std()
    df['Age'] = df['Age'].fillna(np.random.randint(low = mean_age - std_age, high = mean_age + std_age))
    df[['Age']].describe()    #Re-Check that we have fill all of missing data
    df['CategoricalAge'] = pd.qcut(titanic_train_dataset['Age'], q = 4)
    
    
    if mode:
        df = df.loc[:, ["Survived", "Pclass", "Sex", "Age", "FamilySize", 
                                                              "IsAlone", "CabinFloorScore", "FarePerPerson", 
                                                              "TitleScore"]]
    else:
        df = df.loc[:, ["Pclass", "Sex", "Age", "FamilySize", 
                                                              "IsAlone", "CabinFloorScore", "FarePerPerson", 
                                                              "TitleScore"]]
    
    #Encoding string into number : Male = 1, Female = 0
    from sklearn.preprocessing import LabelEncoder
    labelencoder_sex = LabelEncoder()
    df['Sex'] = labelencoder_sex.fit_transform(df['Sex'].values)
    return df


# In[17]:


#Calling function
titanic_train_dataset = pd.read_csv("../input/train.csv")    #Re-import a titanic data set
titanic_train_dataset_for_training_step = create_feature(df=titanic_train_dataset, mode=1)
titanic_test_dataset_for_testing_step = create_feature(df=titanic_test_dataset, mode=0)

#Showing some data
titanic_train_dataset_for_training_step.head(3)
titanic_test_dataset_for_testing_step.head(3)

#To ensure your data is ready to train and test 
titanic_train_dataset_for_training_step.info()
titanic_test_dataset_for_testing_step.info()

X_train = titanic_train_dataset_for_training_step.iloc[:, 1:]
y_train = titanic_train_dataset_for_training_step.iloc[:, 0]
#Now, It's ready for training


# In[18]:


#Training Phase
#1.Model Usage : D.Tree, Forest, ANNs, Logistic Regression, KNN, Naive Bayes
#2.Training Strategy : Grid Search
#3.Testing Strategy : K-fold Cross Validation

#Step 1. Creating model object from each class
#Importing Model libraries
from sklearn.neighbors import KNeighborsClassifier    #KNN
from sklearn.tree import DecisionTreeClassifier    #D.Tree
from sklearn.ensemble import RandomForestClassifier #Forest
from sklearn.naive_bayes import GaussianNB    #Naive Bayes
from sklearn.linear_model import LogisticRegression    #Logistic Regression

#Importing Model Validation
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.cross_validation import cross_val_score

#Initial Model Object from Class
clf_knns = KNeighborsClassifier()
clf_dtree = DecisionTreeClassifier()
clf_forest = RandomForestClassifier()
clf_logreg = LogisticRegression()
clf_naive = GaussianNB()

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    ]

classifiers = [clf_knns, clf_dtree, clf_forest, clf_logreg]
    
#Step 2. Delcare parameters for training and do a hyperparameter tuning by gridsearch
# You can change your parameters here!!!
params_knns = [{'n_neighbors' : range(1, 100)}, {'metric' : ['minkowski']}, {'p' : [2]}]
params_dtree = [{'criterion' : ['gini', 'entropy']}, {'splitter' : ['random', 'best']}]
params_forest = [{'n_estimators' : range(1, 100)}, {'criterion':['entropy', 'gini']}]
params_logreg = [{'penalty' : ['l1', 'l2']}, {'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}, {'max_iter' : range(100, 1000)}]

parameters = [
        params_knns,
        params_dtree,
        params_forest,
        params_logreg,
        ]

#Step 2. Training model using gridsearch
#Importing Grid Search
from sklearn.model_selection import GridSearchCV

clf_best_acc = []
clf_best_params = []
clf_best_estimator = []

        
grid_searchs = [] #grid_search_knns, grid_search_dtree, grid_search_forest, grid_search_svm, grid_search_logreg, grid_search_naive

#Training and append all of the best result from each model to a list
for i in range(len(classifiers)):
    grid_searchs.append(GridSearchCV(estimator=classifiers[i], param_grid=parameters[i], scoring='accuracy', cv=10, 
                                n_jobs=-1))   
    grid_searchs[i].fit(X_train, y_train)

    clf_best_acc.append(grid_searchs[i].best_score_)
    clf_best_params.append(grid_searchs[i].best_params_)
    clf_best_estimator.append(grid_searchs[i].best_estimator_)
 
print("Finishing Training")


# #  A training process will take times for 5-10 minutes(It depends on your parameters). You can minimize this training time by adjust parameters into a small size.  (ง'̀-'́)ง!!!
# 
# ** After fininshing training step, we will compare each model performance and choose the best one to be a predictor of titanic disaster problem. **
# 

# In[19]:


#best_classifier variable for storing best classifier from gridsearch as a dict : Key is the name of classifier, Value is list of [best accuracy, best parameters, best estimators]
best_classifier = {}

#Store each classifier in dictionary 
for i in range(len(classifiers)):
    best_classifier[classifiers[i].__class__.__name__] = [clf_best_acc[i], clf_best_params[i], clf_best_estimator[i]]

#Print out the result of each best classifier can do!!!
for key, value in best_classifier.items():
    print("Classifier name : " + str(key), end="\n")
    print("Accuracy : " + str(value[0]), end="\n")    #value[0] is best acccuracy
    print("Best parameters : " + str(value[1]), end="\n")    #value[1] is best parameters
    print("Best estimator : " + str(value[2]), end="\n")    #value[2] is best estimators
    print("************************************************************************************************", end="\n")
   


# In[20]:


#Comparing between each model performance
import os
import numpy as np
import matplotlib.pyplot as plt

x = best_classifier.keys()
y = list(value[0]*100 for key, value in best_classifier.items())

fig, ax = plt.subplots()   

width = 0.75 # the width of the bars 
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color=['blue', 'red', 'green', 'purple'])    # Make bar plot in horizontal line
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
for i, v in enumerate(y):
    ax.text(v+.7, i , str(v) + '%', color='red', fontweight='bold')
plt.title('Model Performance')
plt.xlabel('Classifier Performance(Percentage)')
plt.ylabel('Classifier name')   


# # Part V : Submission on kaggle
# The final step is submitting test set 's prediction to kaggle in the same format and form of "gender_submission.csv" file.
# 
# ** Note : A "gender_submission.csv" file is not the result of test set. It's just example to show how submission file look like and assuming all female survived. So do not test a test set with this result file. **
# 
# 

# In[21]:


#Testing on test set and create a submission file for submitting to kaggle.
y_pred_submission = pd.DataFrame(grid_searchs[2].predict(titanic_test_dataset_for_testing_step))
y_pred_submission['PassengerId'] = titanic_test_dataset['PassengerId']
y_pred_submission.columns = ['Survived', 'PassengerId']
y_pred_submission = y_pred_submission[['PassengerId', 'Survived']]

#Make sure we have the same format for submission to kaggle
y_pred_submission.head(3)
titanic_submission_form_dataset.head(3)

y_pred_submission.info()
titanic_submission_form_dataset.info()

#Let's goooooooooooooo
y_pred_submission.to_csv('forest_submission.csv', index=False)



# # Part VI : Conclusion of our journey
# Finally, the long journey has ended. We come a long way and i will conclude everything we have passed.
# 1. Part I - Data cleaning : We make a data clean by filling missing data
# 2. Part II - Feature Engineer : From the given feature is not enough to make a powerful model because it didn't tell about passenger enough. So we will create some feature from given feature that can explain and identify passengers.
# 3. Part III - Data visualisng : We visualise the data to make it more understandable.
# 4. Part IV - Training and Testing : We have prepared data set for train and test using "GridSearch"
# 5. Part V - Submission on Kaggle : We have created submission file following the example that we can use it for submit and get score from test set in kaggle.
# 6. Part VI - The end of journey
# 
# ***
# 

# # Message from author
# ** Author : This is my first kernel and I'm trying to learn a machine learning. If there's something wrong or you want me to edit and add more information. Please tell me, I'm happy to see every comment. And I will answer it as soon as possible. ** 
# # #Many thanks
# # # Enjoy and Have fun
# # #♥(ˆ⌣ˆԅ)

# ![](https://i.pinimg.com/736x/9b/75/9f/9b759f7145d4589eba689f667dda6802--titanic-poster-titanic-facts.jpg)
# # ... RIP
