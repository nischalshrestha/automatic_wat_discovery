#!/usr/bin/env python
# coding: utf-8

# In this notebook, I would like to try exploring the Titanic dataset primarily using Pandas Dataframe features. At the end of this notebook, we will export the file with the augmented features into a CSV file which can be worked upon to build predictive models. Here is a quick Table of Contents to help you navigate. 
# 
# <a href="#lessons">Lessons learnt along the way</a>
# 
# <a href="#import">Import Data</a>
# 
# <a href="#vis">Basic Data Visualization</a>
# 
# <a href="#fe">Feature Engineering</a>
# 
# <a href="#mva">Missing Value Imputation on Age</a>
# 
# <a href="#mve">Missing Value Imputation on Embarked</a>
# 
# <a href="#famsize">Analysis of family size</a> 
# 
# <a href="#pclass">One-hot Encoding of Passenger Class</a> 
# 
# <a href="#cabin">Analysis of Cabin information</a> 
# 
# <a href='#exp'>Export Results</a>
# 

# <a id="lessons">**LESSONS LEARNT ALONG THE WAY**</a>
# 
# It appears that Machine Learning is really an art (stating the obvious, huh !?). That being the case, one might make several mistakes along the way. So this section itself was added as an after thought in V.11. Upto V.11., the best accuracy on Test data I got was about 77%. I had built a[ DNN model using Keras ](https://www.kaggle.com/inturiyam/titanic-survivor-prediction-using-keras) to do the predictions. Since I could not improve the performance any further, I decided that it was best to come back to the data set and try improve the features and fix some mistakes. Here is a quick summary of mistakes I made:
# 
# *Mistakes made upto V.11*
# a. I did not one-hot-encode PClass. That was a blunder ! 
# 
# b. I had not done Feature Scaling of Fare here (instead I was doing it in the Keras notebook). And when I had done Feature Scaling, I made a mistake of dividing it by Mean ! 
# 
# c. Initially I tried creating a column called "High Probability Survivor Group" based on my inference from titles people had. I was trying to bias the model in a way. I fixed it subsequently by simply doing one-hot-encoding. 
# 
# d. Also I had not inferred much from the Cabin column. I did so subsequently. 
# 
# *An important note on Mean and Median*
# I found that several notebooks on this competition impute age and fare values using some method or the other. Here I am going to try mark the missing Age values with Median instead of mean and try running models to see if they are any better. The sample here is small and it is likely that there are some Age outliers or Fare outliers. It seems Median is better when datasets have outliers than using the Mean. [See this](https://creativemaths.net/blog/median/)
# 

# <a id='import'> **IMPORT DATA** </a>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))


# In[ ]:


#Read data from CSV input
df_train = pd.read_csv("../input/train.csv")
print(df_train.dtypes)
df_train.info()
df_train.describe()


# <a id='vis'>**BASIC DATA VISUALIZATION**</a>
# 
# We will start with some basic data visualization using the Pandas Histogram option.  At this stage we are trying to identify if there are some groups which had a higher chance of survival than others. As you can see from below histograms,
# * More first class passengers survived (about 2/3rd)
# * Third class passengers had very little chance of survival (about 7/9 died)
# * Second class passengers had a near 50% chance of survival 
# 
# Similarly, 
# * Women had significantly higher chance of survival compared to men
# * Passengers who boarded from Q and S had lower chance of survival compared to C (did Q and S stops have more men or more lower class passengers?)
# * Children seem to have higher probability of survival compared to adults in general

# In[ ]:


#Visualize the data
df_train.hist("Survived", by="Pclass", grid="False", layout=[1,3],figsize = [10,3])
df_train.hist("Survived", by="Sex",figsize = [10,3])
df_train.hist("Survived", by="Embarked", layout=[1,3],figsize = [10,3])
df_train.hist("Age", bins=10, by = ["Survived", "Sex"], layout=[1,4],figsize = [20,3])


# In[ ]:


#Missing value analysis
#This will help identify how many missing values are in each column and take some suitable corrective action
df_train.isnull().sum()


# <a id='mva'>**MISSING VALUE ANALYSIS and FEATURE SCALING ON AGE**</a>
# 
# As we can see from above, there are:
# * 177 missing values in age
# * 2 rows with missing values in embarked and 
# * Nearly all (687) cabin column values being null
# 
# With the above information, we can focus on imputing age values first and consider some form of inferencing for Embarked column subsequently. We will ignore cabin values for the time being as it is very difficult to infer at present. 
# 
# For the age column, for simplicity, we will replace NaN values with average of all age values. This is a simplistic approach. (Maybe we can infer something better from Titles and do a better replacement of missing age values?)

# In[ ]:


#In this step, we replace missing values of Age with their average values
#df_train["Age"].describe()
#avg = np.average(df_train["Age"].fillna(value=0))
#print(avg)
#df_train["Age"].fillna(value = avg, inplace = True)
#df_train["Age"].describe()


# In[ ]:


#A better way to infer age seems to be to do it 'Title' group wise rather than do a replacement on a wholescale basis. Let me try it out ! 
df_train["title"]=df_train["Name"].str.lower().str.extract('([a-z]*\.)', expand=True)
#Passengers in each title group whose age is missing
#df_train[df_train["Age"].isnull()].groupby(by = ["title"])["PassengerId"].count() 

#We will now compute the average age in each group and replace the missing values with the average age of that particular group 
#df_train[((df_train["title"]=="dr.") & (df_train["Age"].isnull()==False))]["Age"]#Some outliers here. Better to use Median
#df_train[((df_train["title"]=="master.") & (df_train["Age"].isnull()==False))]["Age"].hist() #Some outliers here. Better to use Median
#df_train[((df_train["title"]=="miss.") & (df_train["Age"].isnull()==False))]["Age"].hist() #Some outliers here - as old as 50s and 60s. 
#df_train[((df_train["title"]=="mr.") & (df_train["Age"].isnull()==False))]["Age"].hist() #Some outliers here. Better to use Median
#df_train[((df_train["title"]=="mrs.") & (df_train["Age"].isnull()==False))]["Age"].median() #Wow ! someone as young as 14!?

avg_dr = df_train[((df_train["title"]=="dr.") & (df_train["Age"].isnull()==False))]["Age"].median()  
avg_master = df_train[((df_train["title"]=="master.") & (df_train["Age"].isnull()==False))]["Age"].median()  
avg_miss = df_train[((df_train["title"]=="miss.") & (df_train["Age"].isnull()==False))]["Age"].median()  
avg_mr = df_train[((df_train["title"]=="mr.") & (df_train["Age"].isnull()==False))]["Age"].median()  
avg_mrs = df_train[((df_train["title"]=="mrs.") & (df_train["Age"].isnull()==False))]["Age"].median()  
#print(avg_dr,avg_master,avg_miss,avg_mr,avg_mrs)

#We will now replace the missing age values in each group with the corresponding average values 
# Refer - https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
df_train.loc[((df_train["title"]=="dr.") & (df_train["Age"].isnull()==True)).tolist(),'Age']=avg_dr
df_train.loc[((df_train["title"]=="master.") & (df_train["Age"].isnull()==True)).tolist(),'Age']=avg_master
df_train.loc[((df_train["title"]=="miss.") & (df_train["Age"].isnull()==True)).tolist(),'Age']=avg_miss
df_train.loc[((df_train["title"]=="mr.") & (df_train["Age"].isnull()==True)).tolist(),'Age']=avg_mr
df_train.loc[((df_train["title"]=="mrs.") & (df_train["Age"].isnull()==True)).tolist(),'Age']=avg_mrs

df_train["Age"].describe()
#df_train[df_train["Age"].isnull()].groupby(by = ["title"])["PassengerId"].count() 
#We will now scale the Age column and add it as a new column to the dataframe. For this we need to compute the mean and Std.Dev. - we will not do this Title group wise but on whole training set. 
age_mean = df_train["Age"].mean() 
age_std = df_train["Age"].std()
print(age_mean,age_std)
df_train["age_norm"]=((df_train["Age"]-age_mean)/age_std)
df_train["age_norm"].hist()
df_train.describe()


# <a id = 'fe'>**FEATURE ENGINEERING**</a>
# 
# Now we will embark on a little bit of feature engineering. 
# * To start with we will One-Hot encode Sex, Passenger Class and Point of Embarkment
# 
# I am also a bit curious to see what information we can derive from the Name column. We will try to see if there is something meaningful hidden in the names that we can use and suitably engineer the features out of it. 
# 

# In[ ]:


df_train["is_male"] = pd.get_dummies(df_train["Sex"], drop_first=True) #we use drop_first to avoid creating another correlated column is_female


# In[ ]:


#We will bin the passengers into few age groups just to see if children and older passengers had any higher survival probability
bins = [0,15,25,50,100]
df_train["age_group"]=pd.cut(df_train["Age"],bins)
#print(pd.get_dummies(df_train["age_group"]))
df_train[["age15","age25","age50","age100"]]=pd.get_dummies(df_train["age_group"], dtype="uint8")
#print(df_train["age15"])


# In[ ]:


df_train.info()
df_train.hist("Survived", by=["age_group","Sex"], layout=[2,4], figsize = [10,10])


# **Analysis of Name Length**
# 
# We will now try to assess if some meaningful pattern can be observed between Names and survivorship. To start with we will look at Name Lengths. 

# In[ ]:


#We will first add a new feature "Name Length" - just in case there is some impact of it on survival
namelen = []
for i in range(len(df_train["Name"])):
    namelen.append(len(df_train["Name"][i]))
df_train["len_name"]=namelen
#df_train.hist("len_name", by=["Survived", "Pclass"] , bins=10,layout=[4,3], figsize = [15,15])
df_train["len_name"].describe()
#df_train[df_train["len_name"] >= 30]

#We will normalize the name lengths as well here. 
len_name_avg = df_train["len_name"].mean()
len_name_std = df_train["len_name"].std()
print(len_name_avg,len_name_std)
df_train["norm_len_name"]=(df_train["len_name"]-len_name_avg)/len_name_std
df_train["norm_len_name"].hist()
df_train.hist("norm_len_name", by = "Survived") #Honestly I cannot infer much from this. But will let the model figure out if its worthwhile.

#We will now try to see if there was something useful in the Titles that people carried in their names. 
df_train["title"].head()
df_train.hist("title", by=["Survived"], figsize = [15,15], layout = [2,1]) 
#It looks like some titles (e.g. miss) had higher probabilities of survival compared to others (e.g. mr.). 


# <a id='tsr'>**MORE ON TITLES AND SURVIVAL RATE**</a>
# 
# Just to be more sure about our histogram data, we can compute the survival rates by title. 
# 
# Here I tried 2 approaches: 
# a. In order to have a manageable number of features, we will create a boolean column called "high_prob_group" which will have persons with titles showing higher survival rate (>50%) in the training data. For now, I am not considering the sample size in doing this categorization. We will add the following titles to this group - countess, lady, master, miss, mlle, mme,mrs,ms, sir.
# 
# b. Simply one hot encode a few groups of titles and mark the rest as rare titles 

# In[ ]:


#To start with we will calculate survival rate by each title group
#df_train.pivot(index="PassengerId",columns = "title", values = "Survived")
df_train.groupby(["title"])["PassengerId"].count()
df_train.groupby(["title"])["Survived"].sum()/df_train.groupby(["title"])["PassengerId"].count()


# In[ ]:


#Approach 1
lookfor = np.array(['mrs.','sir.','countess.', 'lady.', 'master.', 'miss.', 'mlle.', 'mme.','mrs.','ms.', 'sir.'])
#s = pd.Series(lookfor)
df_train["high_prob_group"]=df_train["title"].isin(lookfor).astype('uint8')
df_train["high_prob_group"].sum()


# In[ ]:


#Approach 2
#use of x.astype('uint8') helps convert the Boolean output of isin() to an integer (0,1) representation 
df_train["title_ms"] = df_train["title"].isin(["miss.","ms."]).astype('uint8')
df_train["title_mrs"] = df_train["title"].isin(["mrs.","mme.","mlle."]).astype('uint8')
df_train["title_mr"] = df_train["title"].isin(["mr."]).astype('uint8')
df_train["title_others"]=df_train["title"].isin(['countess.', 'lady.', 'master.', 'dr.', 'don.','jonkheer.','rev.','major.','sir.','col.','capt.']).astype('uint8')


# <a id='mve'>**MISSING VALUES OF  EMBARKED COLUMN**</a>
# 
# We will now try to fill in the missing values for Embarked column. To do so, we need to figure out what might be the approximate fares for a [PClass, Embarked] combination. We also need to then identify the rows where we need to infer and fill in the missing values manually.  So we start with plotting a histogram on the Fare column. 
# 
# Ah !, it turns out that both the passengers without Embarkation information travelled in the same cabin with same ticket numbers. Based on the below histograms, I am proceeding with the assumption that both of them boarded from "C". 

# In[ ]:


#We start with a boxplot to figure out the range of Fare values
df_train.boxplot("Fare", by=["Embarked","Pclass"], figsize = [8,8])


# In[ ]:


df_train.hist("Fare", by=["Embarked", "Pclass"],layout=[4,3], figsize = [15,15], bins=10)
df_train[df_train["Embarked"].isna()] 
df_train["Embarked"].fillna(value = "C", inplace = True)
df_train.hist("Fare", by=["Embarked", "Pclass"],layout=[4,3], figsize = [15,15], bins=10)
#df_train[df_train["Embarked"]=="C"]
df_train[["embC","embQ","embS"]]=pd.get_dummies(df_train["Embarked"], dtype="uint8")
#df_train.hist("Embarked",by=["Survived","Pclass"],layout=[2,3], figsize = [10,8]) #Just ran this to see if there is any significant pattern in data


# In[ ]:


#At this point, let's quickly normalize the fares as well for future use
df_train["Fare"].describe()
fare_mean = 32.204208
fare_std = 49.693429
df_train["norm_fare"]= (df_train["Fare"]-fare_mean)/fare_std
df_train["norm_fare"].describe()


# <a id='famsize'>**ANALYSIS OF FAMILY SIZE**</a>
# 
# To start with we will assess if having someone accompany a person had any impact on their survivorship. 
# * Broadly, it appears that having 1 Parent/Child accompanying a person has a small chance of higher survivorship. Otherwise, the survivorship does not seem to have got affected by how many children are traveling with a parent with large families being at a disadvantage.
# * Except couples, large families seem to have lower chance of survival
# 
# There are again 2 possible approaches here : 
# Approach 1: For now, we will not alter this information and directly let the models learn from this data. 
# Approach 2: Compute a Total Family Size = Sibsp + Parch and see if there is any pattern

# In[ ]:


df_train.hist("Survived", by=["Parch"],layout=[2,4], figsize = [15,10])
df_train.hist("Survived", by=["SibSp"],layout=[2,4], figsize = [15,10])


# In[ ]:


df_train["tot_family_size"] = df_train["Parch"]+df_train["SibSp"]
print(df_train["tot_family_size"].mean(),df_train["tot_family_size"].std())
#df_train.hist("Survived",by ="tot_family_size", layout = [3,3],figsize = [15,10] )
#df_train["tot_family_size"].hist()
#I will just do a feature normalization and leave it there instead of putting them into different family size compartments since the sample sizes for larger families is too small to figure out if their survivorship was really influenced by family size. 
df_train["norm_family_size"] = (df_train["tot_family_size"]-df_train["tot_family_size"].mean())/(df_train["tot_family_size"].std())
df_train["norm_family_size"].hist()


# <a id='pclass'>**ONE-HOT ENCODE PASSENGER CLASS**</a>
# 
# We will quickly do a one-hot encoding of passenger class. 

# In[ ]:


df_train["Pclass"].plot.kde()
df_train[["P2","P3"]]=pd.get_dummies(df_train["Pclass"],drop_first=True)


# <a id="cabin">**ANALYSE CABIN INFORMATION**</a>
# 
# We will now look at the last unprocessed column, Cabin, to see if any meaningful inference can be drawn from it. We saw earlier that it had lot of missing information.

# In[ ]:


df_train["Cabin"].isnull().sum()
df_train["cab"] = df_train["Cabin"].str.lower().str.get(0)
df_train["cab"].fillna(value="z",inplace=True)
df_train["cab"].isna().sum()
df_train.hist("Survived", by = "cab",figsize = [10,10])
df_train[["cab_b","cab_c","cab_d","cab_e","cab_f","cab_g","cab_t","cab_z"]] =pd.get_dummies(df_train["cab"],drop_first=True)


# <a id='exp'>**EXPORT RESULTS**</a>
# 
# Export processing results to a new CSV file for further work - that will be different notebook. Thank you for reading through till here. Hope it helped you get started. 

# In[ ]:


#It appears that the file gets stored to a folder called working. See below. 
df_train.info()
df_train.to_csv(path_or_buf="train_processed.csv")
print(os.listdir("../"))
print(os.listdir("../working"))


# In[ ]:


import seaborn as sns
corr = df_train[["P2","P3","norm_len_name","title_ms","title_mrs","title_mr","title_others","is_male","age_norm","norm_family_size","norm_fare",
                    "cab_b","cab_c","cab_d","cab_e","cab_f","cab_g","cab_z","embQ","embS"]].corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(0, 50, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:




