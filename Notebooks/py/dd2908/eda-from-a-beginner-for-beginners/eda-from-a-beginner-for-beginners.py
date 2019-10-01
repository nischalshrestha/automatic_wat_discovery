#!/usr/bin/env python
# coding: utf-8

# This is an exploratory analsyis performed on the titanic dataset. I am a beginner in the field of data science, and hence this work might not be the best in terms of choice of visualizations, analysis or inference. However, I am still publishing this so as to help other beginners like myself get started with their own analysis of this data. Any constructive criticisms are welcome!

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Load the data set
df = pd.read_csv("../input/train.csv")


# In[ ]:


(df["Age"].describe())


# Once the dataset is loaded, we can  take a look at variables in it

# In[ ]:


df.head()


# We see that there are 12 fields/columns in the data set. A description of the each of these column is as follows
# 
# 1) **PassengerID**: Serves as an index in the dataset
# 
# 2) __Survived__: 1 indicates that the passenger survived the disaster, 0 indicates otherwise
# 
# 3) **PClass**: The class in which the passenger was traveling. The values are 1, 2 and 3 which stand for first, second and third class resepctively
# 
# 4) **Name**: The name of the passenger
# 
# 5) **Sex**: The gender of the passenger
# 
# 6) **Age**: The age of the passenger
# 
# 7) **SibSp**: The number of people abroad, who were either the spouse or siblings of the passenger
# 
# 8) **Parch**: The number of people abroad, who were either parents/childern of the passenger
# 
# 9) **Ticket**: The ticket number of the passenger
# 
# 10) **Fare**: The ticket fare the passenger paid
# 
# 11) **Cabin**: The cabin allocated to the passenger
# 
# 12) **Embarked**: The port from which the passenger embarked on to the ship. C stands for Cherbourg; Q for Queenstown and S for Southampton

# Let us create a new column indicating the total number of relatives of the passenger on board. We will be deriving subsets of this original data frame df in the following lines, and hence it is best to create any new columns that may be required in the dataset at this point itself. The reason for creating this particular column 'tot_rel' shall be explained later

# In[ ]:


df["tot_rel"] = df["SibSp"]+df["Parch"]


# Having had a first look at the different variables, it is a good idea to look at a short summary of dataset, which is shown below.

# In[ ]:


pd.DataFrame(df.describe())


# The table above shows a brief summary of the numeric fields in the dataset. It is not meanigful to look at the statistical paraemeters such as mean, quartiles, min and max in case of the passengerID, survived, and PClass variables since passengerID is simply a serial number, whereas Pclass and Survived are basically categorical variables expressed as numbers. What is immediately noticable from the above summary is the difference between the number of values in the Age field, as opposed to the other variables in the dataset (714 vs 891). This, along with the presence of NaN for the quartile values of the Age column, indicates that some of the datapoints are missing for the age variable. The missing values can be dealt in either of two ways - 1) Remove the rows for which missing values are present completely, or 2) Fill the missing values appropriately. Option 1 would be appropriate only if there are only very few missing values compared to the size of the dataset, otherwise, it would lead to loss of valuable imformation from the dataset. Often option 2 is preferred, and there are number of ways to do this. But before going into that, let us have a look at the check for all rows in the data frame with at least one value missing.
# 
# 

# In[ ]:


df_missing = df[df.isnull().any(axis=1)]
df_missing


# The above command checks if any of the value in the a given row of the dataframe is NA (axis=1 parameter ensures this), and returns the same to df_missing. Here df_missing is a new data frame including all rows of the original dataframe with at least one value missing. Although there were only 177 values missing from the age column, we see that 708 rows of the 891 in the original data set has at least one value missing. It could be that of these 708, some the missing values are from the the Name or Ticket columns, which might not be that important in analyzing the data. Let's check that 

# In[ ]:


print(df["Name"].isnull().sum())
print(df["Ticket"].isnull().sum())


# We see that that the missing values are indeed from the other fields which are important in understanding this data. 708 rows out of 891 is about 80% of the data, hence option 1 mentioned earlier, which involved removing rows with missing values from the data set is out of question for this dataset. We need to find other ways of filling in the missing values in this data set. Let's have a look at the number of missing data values in each of the relevant fields

# In[ ]:


fields = ["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Cabin","Embarked"]
na_count = {}
for field in fields:
    na_count[field]=df[field].isnull().sum()
    
display(na_count)    


# All the fields execpt Age, Cabin and Embarked are not missing any values. 177 values are missing from Age column, 687 values are missing from Cabin column, while 2 columns are missing from Embarked column. Since most values are missing from the Cabin column (~ 77% values missing, it is perhaps best to exclude this variable from further analysis. 
# 
# __NOTE__: It should be remembered that cabin number, is indicative of the location of the cabin inside the ship. The position of the cabin, and its proximity to the life boats would be an important predictor of survival of the passenger inside the cabin. It can be assumed that the Passenger Class will already include information about the Cabin implicitly (Cabins of the same class may be close together). If that is indeed the case, then perhaps the exclusion of Cabin names from this data set will not affect our analysis much. However, this need not be true always, (same class of passengers might have had cabins at different decks within the ship). We will need more domain knowledge about this case to clarify this information. 

# __NOTE__:The number of missing values can also be found using the command df.info(), as shown below

# In[ ]:


df.info()


# Let us explore the "Age" variable, and see if we can figure out a way to fill in the missing values. First, let us look at the distribution of ages in the dataset.
# 

# In[ ]:


get_ipython().magic(u'matplotlib inline')
sns.boxplot(y=df["Age"])
plt.title("All Passengers")


#    **Fig 1. Distribution of Age of All Passengers**

# We see that half of the passengers were aged between 20 and 40. Let us also look at the distribution of ages according to their Genders
# 
# 

# In[ ]:


df_male = df.groupby("Sex").get_group("male")
df_female = df.groupby("Sex").get_group("female")
fig = plt.figure()
plt.subplot(121)
sns.boxplot(y=df_male["Age"])
plt.title("Male Passengers")
plt.ylim(0,80)
plt.subplot(122)
sns.boxplot(y=df_female["Age"])
plt.title("Female Passengers")
plt.ylim(0,80)
fig.tight_layout()


# **Fig 2. Age of Passengers according to Gender **

# Let us now look at the distribution of age according to Gender & Passenger class. The reason for looking at this combination of the data is that one approach to filling in the missing data for the Age column would be to look at the gender and passenger class for that row, and fill in the median value of age for that combination in the dataset. For instance, if an age value is missing for a male passenger travelling by first class, we would impute it by the median age of first class male passengers. 
# Note that this is just one idea to impute age. There could be alternate approaches for the same

# In[ ]:


fig = plt.figure()
fig.set_figheight(13)
fig.set_figwidth(13)
pclasses = [1,2,3]
plot_num_1 = 0
male_median_age_by_class = {}
female_median_age_by_class = {}
for pclass in pclasses:
 
    plot_num_1 = plot_num_1+1
    plt.subplot(2,3,plot_num_1)
    df_male_class = df_male.groupby("Pclass").get_group(pclass)
    sns.boxplot(y=df_male_class["Age"])
    title_string = "Class "+str(pclass)+" Male Passengers"
    plt.title(title_string)
    plt.ylim(0,80)
    male_median_age_by_class[pclass]=df_male_class["Age"].median()
    
    plot_num_2 = plot_num_1+3
    plt.subplot(2,3,plot_num_2)
    df_female_class = df_female.groupby("Pclass").get_group(pclass)
    sns.boxplot(y=df_female_class["Age"])
    title_string = "Class "+str(pclass)+" Female Passengers"
    plt.title(title_string)
    plt.ylim(0,80)
    female_median_age_by_class[pclass]=df_female_class["Age"].median()



fig.tight_layout()


# **Fig 3. Age of Passengers according to Gender and Class **

# Among the male passengers, those travelling by classes 1, 2 and 3 have median ages approximately 40,30 and 25 approximately. In case of female passengers, the class 1, 2 and 3 passengers have median ages approximately 35,28 and 22 respectively.  These values are stored in the dictionary "male_median_age_by_class" and "female_median_age_by_class". 

# In[ ]:


display(male_median_age_by_class)
display(female_median_age_by_class)


# Now that we have found the median ages of passengers according to their gender and travelling class, we will use them to impute the missing values for ages in the dataset, depending on which group the passenger falls into. Let us do it now

# In[ ]:


for gender in ['male','female']:
    for pclass in [1,2,3]:
        if (gender=='male'):
            age_dict = male_median_age_by_class
        else:
            age_dict = female_median_age_by_class
        df.loc[(df['Sex']==gender)&(df['Pclass']==pclass)&(df['Age'].isnull()),'Age']=age_dict[pclass]
    


# In[ ]:


df["Age"].isnull().sum()


# Having imputed the missing values for ages, we now look at the "Embarked" field, which is the only other column with missing values, and have just 2 missing values. Let us look at the percentage of passengers boarded from the three ports of embarkation in the data set

# In[ ]:


df["Embarked"].value_counts()/df.shape[0]


# We see that over 72% of the passengers boarded from the first port of embarkation, Southampton. Since that forms a significant majority, let us just impute the missing values with S

# In[ ]:


df["Embarked"].fillna("S",inplace=True)


# In[ ]:


df["Embarked"].isnull().sum()


# Let's take a look at the passengers grouped by their gender

# In[ ]:


get_ipython().magic(u'matplotlib inline')
df["Sex"].value_counts().plot(kind="bar")


# **Fig 4. Number of male and female passengers **

# It would be more informative if we could plot the above chart as percentages of the total number of members in the dataset. This is done below

# In[ ]:


((df["Sex"].value_counts())*100/df.shape[0]).plot(kind="bar",title="All Passengers")


# **Fig 5. Percentage of male and female passengers **

# Here, we see that about 65% of the total passengers were male, and the rest were female. Also look at the distributoin of passengers according to the pasesnger class in the figure 6 below. About 55% of the passengers travelled by class 3, about 25% by class 1 and 20% by class 2. 
# 

# In[ ]:


(df["Pclass"].value_counts()/df.shape[0]).plot(kind="Bar",title="Passenger Class (All Passengers)")


# **Fig 6. Percentage of passengers in Classes 1, 2 and 3 **

# Let us also look at the distribution of passenger class for males and females separately

# In[ ]:


fig = plt.figure()
plt.subplot(121)
(df_female["Pclass"].value_counts()/df_female.shape[0]).plot(kind="bar",title="Passenger Class (Females)")
plt.subplot(122)
(df_male["Pclass"].value_counts()/df_male.shape[0]).plot(kind="bar",title="Passenger Class (Males)")


# **Fig 7. Percentage of male and female passengers in Classes 1, 2 and 3 **

# About 45% of the female passengers travelled by class 3, 30% by class 1 and 25% by class 2, whereas 60% of the male passengers travelled by class 3, about 20% each by classes 1 and 2

# Having looked at some aspects of the dataset as a whole until now, let us now analyse the composition of the survivors from the dataset. For instance, let us now take a look at the gender distribution among survivors (figure 8). Approximately 70% of the survivors were females and rest males.
# 

# In[ ]:


df_group_gender = df.groupby("Sex")
df_female  = df_group_gender.get_group("female")
df_male  = df_group_gender.get_group("male")


# In[ ]:


df_survived = df.groupby("Survived").get_group(1)
df_not_survived  = df.groupby("Survived").get_group(0)

((df_survived["Sex"].value_counts())/df_survived.shape[0]).plot(kind="bar",title="Survivors")
#ax.set_xticklabels(['Survived','Not Survived'])


# **Fig 8. Composition of Survivors (percentage) according to Gender  **

# It is a well known fact about the titanic disaster that women and children were prioritized during evacuation of the sinking ship, something that is evident from the figure above. Figure 9 below actually shows the survival chances of males and females abroad the titanic. It can be infered from the figure that a female had over 70% chance of surviving the disaster, whereas the male had only 20% chance of surviving it

# In[ ]:


df_survived_male = df_survived.groupby("Sex").get_group("male")
df_survived_female = df_survived.groupby("Sex").get_group("female")
                                                    
order = [1,0] #To ensure the order of plotting as survived, not survived in the bar chart
fig = plt.figure()
plt1 = plt.subplot(121)
plt1 = ((df_female["Survived"].value_counts())/df_female.shape[0]).ix[order].plot(kind="bar",title="Female Passengers")
plt1.set_xticklabels(['Survived','Not Survived'])
plt2 = plt.subplot(122)
plt2 = ((df_male["Survived"].value_counts())/df_male.shape[0]).ix[order].plot(kind="bar",title="Male Passengers")
plt2.set_xticklabels(['Survived','Not Survived'])


# **Fig 9. Percentage of survivors among Male and Female Passengers  **

# We see a significantly higher chance of survival for a female passenger, which makes gender the single most factor that predicts the survival chances. Hence, while looking for the influence of other variables in the dataset , we will do that separately for males and females. Figure 10 below shows the distribution of age groups among all of the passengers as well as the survivors, according to their gender.

# In[ ]:


fig1,ax1 = plt.subplots(1,2) 
df.hist("Age",by="Sex",ax=ax1,color='orange',bins=(0,10,20,30,40,50,60,70,80,90))
df_survived.hist("Age",by="Sex",ax=ax1,color='green',bins=(0,10,20,30,40,50,60,70,80,90))
#df.plot("Age",kind='hist',ax=ax1)
#df_survived.hist("Age",by="Sex",ax=ax1)
plt.text(100,100,"All Passengers",color='Black',fontsize=10,backgroundcolor="Orange")
plt.text(100,90,"Survivors         ",color='Black',fontsize=10,backgroundcolor="green")


# **Fig 10. Age-wise distribution of Passengers and Survivors according to Gender**

# As mentioned already, the survival rates among women is much higher than men. Specifically, we see that among women aged 50 and above, there is a survival rate close to 100%, whereas among the males, the survival rate is highest for the 0-10 age group. This should be the case, since children were also prioritized along with women in the evacuation
# 
# Another interesting plot to look at would be the survival rates among passengers travelling different passenger classes. Let us first define a function to plot stacked bar charts, since we will be making a number of bar plots below

# In[ ]:


#Since we will be plotting a number of stacked bar charts from now, let us define a function for the same.
def plot_stacked_bar_chart(data1,data2,order,chart1_color,chart2_color):
    data1.ix[order].plot(kind="bar",color = chart1_color)
    data2.ix[order].plot(kind="bar",color = chart2_color,stacked="True")


# In[ ]:


order = [1,2,3]
fig = plt.figure()

plt1 = plt.subplot(121)
plot_stacked_bar_chart(df_male["Pclass"].value_counts(),df_survived_male["Pclass"].value_counts(),order,"Orange","green")
plt.title("Male Passengers by Class")

plt2 = plt.subplot(122)
plot_stacked_bar_chart(df_female["Pclass"].value_counts(),df_survived_female["Pclass"].value_counts(),order,"Orange","green")
plt.title("Female Passengers by Class")


plt.text(2.75,150,"All Passengers",color='Black',fontsize=10,backgroundcolor="Orange")
plt.text(2.75,140,"Survivors         ",color='Black',fontsize=10,backgroundcolor="green")


# **Fig 11. Number of Passengers and Survivors in each Class according to Gender**

# Female passengers in classes 1 and 2 had over 90% survival chances, whereas those travelling by third class only had approximately 50% surival chance. This should be noted, since being a female passenger significantly improved the odds of survival only if the passenger was in either of the two top classes, despite the fact a significant number of women (about 45%, as can be seen from figure 7) were travelling by class 3. Among the male passengers,survival chances were still the best among first class passengers, although it was less than 40% 
# 
# Let us now take a look at the influence of number of family members on board, on the survival chances of the passenger

# In[ ]:


fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(8)

title_dict = {}
fields = ["tot_rel","SibSp","Parch"]
title_base_string_1 = "Survival (Males) according \nto number of \n"
title_base_string_2 = "Survival (Females) according \nto number of \n"
title_dict = {"tot_rel":"Relatives","SibSp":"Siblings","Parch":"Parents & Children"} # To set plot titles
plot_num = 0
order1 = [0,1,2,3,4,5,6] #passed into function plotting bar chart, to set the order of x-values of the chart
for field in fields:
    plot_num = plot_num+1
    plt.subplot(3,2,plot_num)
    plot_stacked_bar_chart(df_male[field].value_counts(),df_survived_male[field].value_counts(),order1,"Orange","green")
    title_string = title_base_string_1+title_dict[field]
    plt.title(title_string)
    
    plot_num = plot_num+1
    plt.subplot(3,2,plot_num)
    plot_stacked_bar_chart(df_female[field].value_counts(),df_survived_female[field].value_counts(),order1,"Orange","green")
    title_string = title_base_string_2+title_dict[field]
    plt.title(title_string)
    
plt.tight_layout()
#plt.text(10,0.8,"All Passengers",color='Black',fontsize=14,backgroundcolor="blue")
#plt.text(10,0.7,"Survivors    ",color='Black',fontsize=14,backgroundcolor="red")

fig = plt.figure()
fields = ["tot_rel","SibSp","Parch"]
title_base_string = "Survival Rate according \nto number of \n"
title_dict = {"tot_rel":"Relatives","SibSp":"Siblings","Parch":"Parents & Children"} # To set plot titles
plots = [1,2,3]
order1 = [0,1,2,3,4,5,6] #passed into function plotting bar chart, to set the order of x-values of the chart

for (i,field) in zip(plots,fields):
    plt.subplot(1,3,i)
    data1 = df_survived_female[field].value_counts()/df_female[field].value_counts()
    data2 = df_survived_male[field].value_counts()/df_male[field].value_counts()
    plot_stacked_bar_chart(data1,data2,order1,"blue","red")
    title_string = title_base_string+title_dict[field]
    plt.title(title_string)
plt.tight_layout()

plt.text(10,0.8,"Female",color='Black',fontsize=14,backgroundcolor="blue")
plt.text(10,0.7,"Male    ",color='Black',fontsize=14,backgroundcolor="red")


# **Fig 12. Number of Passengers and Survivors according to Relatives,Parents or Siblings  grouped by Gender**

# The survival rate among females is seen to decrease when the number of members onboard that are either their parents or children is greater than 3, or 2 in case they are sibilings. However, it should be noticed that total number of females who had 3 or more relatives on board is much low, and hence it might not be very appropriate to look at the survival rate. So is the case among males, in case of which we see an increase in survival rate with increase in number of relatives,up to 3, however, the total number of males with more  3 or more relatives is too low, to conclusively speak of survival rates. Perhaps we can just conclude that in case of females,having more than 3 relatives reduces their chance of survival 

# Now, let us look at the survival rate depending on the port of embarkation

# In[ ]:


fig = plt.figure()
data_1 = (df_survived_female["Embarked"].value_counts())/(df_female["Embarked"].value_counts())
data_2 = (df_survived_male["Embarked"].value_counts())/(df_male["Embarked"].value_counts())
order = ['S','C','Q']
plot_stacked_bar_chart(data_1,data_2,order,'Blue','Red')
plt.title("Survival Rate by Port of Embarkation")

plt.text(3,0.8,"Female",color='Black',fontsize=14,backgroundcolor="blue")
plt.text(3,0.7,"Male    ",color='Black',fontsize=14,backgroundcolor="red")



# **Fig 13. Survival Rate according to Port of Embarkation**

# The first port of embarkation for the Titanic was Southampton (marked as "S"), followed by Cherbourg (marked as "C") and the last being Queenstown (marked as "Q"). Among the female passengers, those who boarded from Cherbourg had the highest survivors among them, followed by those who boarded from Queenstown, and lastly Southampton. Among the male passengers, again, those who boarded from Cherbourg had the most survivors, however, followed by Southampton and then lastly by Queenstown. It should however be noted that the survival rates between passengers boarded from different ports is not different by a large margin. It would then be of interest to check if the higher survival rates, say for example, in case of females boarded from Cherbourg, is large enough to be statistically significant. In other words, let us run a statitical test for binomial distribution to establish (or disprove) the statistcial significance of the higher survival rate of the group of females who boarded from Cherbourg compared to the population of females in the data set. For this purpose, let us print out the raw nubmers in addition to the figures 

# In[ ]:


display('Survival rate of female passengers according to port of embarkation:(S=Southampton, C=Cherbourg, Q=Queenstown) : ')
display(df_survived_female["Embarked"].value_counts()/df_female["Embarked"].value_counts()) #survival rate among females according to Port of Embarkation
display("Survival rate of female passengers in the population: ")
display(df_survived_female.shape[0]/df_female.shape[0])


# In[ ]:


display("Female Survivors")
display(df_survived_female["Embarked"].value_counts()) 
display("Female Passengers")
display(df_female["Embarked"].value_counts())


# The exact numbers are displayed above, which shows that females who boarded from Southampton, Cherbourg and Queenstown had 69%, 87% and 75% chance of survival respectively. We can see that the survival rate among female passengers who boarded from Southampton and Queenstown is about the same as the general survival rate among the entire female passengers of titanic, so that there is nothing special about these group. However, the 87% chance we see for female passengers boarded from Cherbourg can be tested for statistical significance.

# The distributions we are dealing with here are binomial, where each trial would be a particular case of passenger, and the success would be the survival of the passenger. For our purposes of testing the survival rate of female passengers from Cherbourg, the population would be the subset comprising of all female passengers from the original data set, and the sample would be the subset of female passengers who boarded from Cherbourg.
# Hence, the binomial parameters would as follows
# 
# $$p = probability\ of\ success = proportion\ of\ success\ in\ population\ (female\ survival\ rate\ in\ data\ set)\ =\ 0.742 $$
# 
# $$q = 1 - p$$
# 
# $$n = sample\ size = 73$$
# 
# $$x = no.\ of\ successes\ in\ sample =  64$$
# 
# $$mean\ of\ a\ binomial distribution = np$$
# 
# $$stdev\ of\ binomial\ distribution\ \sigma = \sqrt{np(1-p)}$$
# 
# Having listed the binomial statistics, let us list the null and alternate hypotheses
# 
# $$ H_0 : \mu = 73 * 0.742  $$
# $$ H_A : \mu > 73 * 0.742  $$

# Let us now calcualte the z statistic
# 
# $$ z = \frac{(X - \mu)}{\sigma} = \frac{(X - np)}{\sqrt{(npq)}} $$
# where X indicates the number of cases of survival in the sample, whereas $\mu$ (= np) indicates the expected number of survival in the sample with a probabilty of success being p and ${\sigma}$ is the standard deviation of the binomial distribution of samples. By central limit theorem, z ~ N(0,1). Let us calculate the z score

# In[ ]:


p  = 0.7420382165605095
q  = 1 - p
n  = 73
x = 64

z = (x - n*p)/pow((n*p*q),0.5)
display(z)


# A z value of 2.63 as above means that the probability of us seeing a survival rate of 87% or more as we see in the group of females who boarded from Cherbourg, given the general survival rate of females in the entire data set as  0.742, is 1 - 0.9957 = 0.0043 or less than 0.5% (the value 0.9957 is obtained from Z-tables). Hence, this increased survival rate among female passengers from Cherbourg is sigificant even at an ${\alpha}$ level of 0.005, and so we reject the null hypothesis
# 
# We can run a similar test for male passengers from Cherbourg.
# 

# In[ ]:


display('Survival rate of Male passengers according to port of embarkation:(S=Southampton, C=Cherbourg, Q=Queenstown) : ')
display(df_survived_male["Embarked"].value_counts()/df_male["Embarked"].value_counts()) #survival rate among females according to Port of Embarkation
display("Survival rate of Male passengers in the population: ")
display(df_survived_male.shape[0]/df_male.shape[0])



# In[ ]:


display("Male Survivors")
display(df_survived_male["Embarked"].value_counts()) 
display("Male Passengers")
display(df_male["Embarked"].value_counts())


# $$p = probability\ of\ success = proportion\ of\ success\ in\ population\ (male\ survival\ rate\ in\ data\ set)\ =\ 0.189 $$
# 
# $$q = 1 - p$$
# 
# $$n = sample\ size = 95$$
# 
# $$x = no.\ of\ successes\ in\ sample =  29$$
# 
# $$mean\ of\ a\ binomial distribution = np$$
# 
# $$stdev\ of\ binomial\ distribution\ \sigma = \sqrt{np(1-p)}$$
# 
# Having listed the binomial statistics, let us list the null and alternate hypotheses
# 
# $$ H_0 : \mu = 29 * 0.189  $$
# $$ H_A : \mu > 29 * 0.189  $$

# Let us now calcualte the z statistic
# 

# In[ ]:


p  = 0.18890814558058924
q  = 1 - p
n  = 95
x = 29

z = (x - n*p)/pow((n*p*q),0.5)
display(z)


# A z value of 2.897 as above means that the probability of us seeing a survival rate of 30% or more as we see in the group of Males who boarded from Cherbourg, given the general survival rate of males in the entire data set as  0.189, is 1 - 0.9981 = 0.0019 or less than 0.2% (the value 0.9981 is obtained from Z-tables). Hence, this increased survival rate among male passengers from Cherbourg is sigificant even at an ${\alpha}$ level of 0.002

# 
# The reasons for the above trends can be partially explained from figure 14 shown below. The higher survival rate among passengers boarded from Cherbourg is explained from the larger proportion of class 1 passengers among them, which can be seen from the figure 14. As we saw earlier, the survival chances are highest for passengers travelling by class 1, be it male or female. 
# 
# An additional interesting observation that can be made is that almost 90% of the female passengers from Queenstown travelled by class 3, yet, their survival rate remained marginally higher than the average survival rate of female passengers. This probably indicates that of those passengers who traveled by class 3, the ones who boared from Queenstown had higher chances of survival. 
# 

# In[ ]:


fig = plt.figure()
fig.set_figheight(14)
fig.set_figwidth(12)
order = [1,2,3]
ports = ["S","C","Q"]
port_names = {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}
title_base = "Passenger class for Port of Embarkation: "
plot_num = 0
for port in ports:
    plot_num = plot_num+1
    title_string = title_base+port_names[port]+"(Female)"
    plt.subplot(3,2,plot_num)
    data_1 = df_female.groupby("Embarked").get_group(port)["Pclass"].value_counts()
    data_1.ix[order].plot(kind="Bar")
    plt.title(title_string)
    
    plot_num = plot_num+1
    title_string = title_base+port_names[port]+"(Male)"
    plt.subplot(3,2,plot_num)
    data_2 = df_male.groupby("Embarked").get_group(port)["Pclass"].value_counts()
    data_2.ix[order].plot(kind="Bar")
    plt.title(title_string)


# **Fig 14. Number of passengers in each class for each port of embarkation**

# **Conclusion**

# This work involved some preliminary investigation of the titanic dataset to identify which factors contributed to improved chance of passenger survival. Specifically, the effect of 5 factors viz., Gender, Age, Passenger class, Number of Relatives and Port of Embarkation  were looked at separately to check for possible influence in survival chances. Analysis of the dataset indicates an significantly improved survival rate among women as opposed to men. Among women, those aged 50 and above had the highest chance of survival, whereas for males, children (aged < 10) had the best odds of survival. Among both men and women, those who travelled by first class had the highest survival chances, whereas those who traveled by third class had the least odds of survival. As for the number of relatives on board, having 3 or more relatives clearly reduced the survival chances among females, althoug the same cannot be conclusively said about males owing to lack of enough samples. Among both males and females, those who embarked from Cherbourg had the highest odds of survival, which was proved statistically. However, this was probably due to the fact that hte majority of passengers who boarded from Cherbourg, travelled by first class.
