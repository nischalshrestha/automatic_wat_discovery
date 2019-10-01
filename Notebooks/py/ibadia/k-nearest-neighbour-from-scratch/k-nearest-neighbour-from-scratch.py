#!/usr/bin/env python
# coding: utf-8

# ## In this notebook we are going to build a simple K Nearest Classifier from Scratch
# 
# 
# ### and do a very bad prediction on titanic dataset. :D
# NOTE : THIS DOES NOT DO A GOOD PREDICTION ON DATASET, The purpose of this notebook is just to understand K Nearest Neighbor Classifier

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input/"))
from collections import Counter


# So K nearest Neighbor is a simple algorithm which can be used to do classification on different data. To understand K Nearest Neighbour we just need to understand few simple things: 
# 1. Euclidean Distance (distance between two points in Graph)
# 2. How voting works (literally)
# 3. How to plot point (1,3) on a x,y plane graph
# 4. How to plot (1,2,4) on a simple x,y,z graph
# 
# 
# Now without any issue lets come to the data, We will be using simple K nearest neighbor to find out that whether it is possible to find out that given a person Age and Fare can we predict that he will survive in the titanic or not.
# 

# So First we will read the data,  The training data and lets look at the top5 rows of the data.
# Training Data: Training data is the data we give to our Algorithm to learn with, Then we give our Algorithm the testing data. 
# Lets sayif i gave you some questions and answers like:
# 
# X=[
# 1+2=3
# 
# 2+3=5
# 
# 1+1=2]
# 
# 
# Then I ask you a question What is 33+3=?
# 
# Here 33+3 is our testing data and X is our training data.
# We train our model and test our model to see how many of the questions in test our model gave right answers to.
# 
# 
# Just like that in our dataset we will give our model a person Age and Fare along with the result that he/she survived or not.
# 
# Then we will give our algorithm the testing data in which there will be Age and Fare and our model will predict whether the person survives or not.
# 
# Have a look at a head which means top5 rows of our trainng data.
# 

# In[ ]:


data=pd.read_csv("../input/train.csv")[["Survived","Age","Fare"]]
data=data.fillna(data.mean())
data.head()


# As you can see in training data we gave the person age and fare and the result that whether the person survived or not. 
# 
# 
# For the test data we will just give the Age and Fare and let our algorithm decide that whether the person will survice or not.
# Here have a look at top5 rows of our testing data to get an idea of how our data is.

# In[ ]:


passenger_id=pd.read_csv("../input/test.csv")["PassengerId"]


# In[ ]:


test_data=pd.read_csv("../input/test.csv")[["Age","Fare"]]
test_data=test_data.fillna(test_data.mean())
test_data.head()
test_data=test_data.values


# Now that we have the training data in training_data, lets try to give it to our Algorithm to learn from that data
# 
# Now we will start to understand the K Nearest Neighbor, 
# First of all we will plot all points on a graph, How?
# 
# On X Axis we will put Age and on y axis we will put Fare
# 
# Example, you look at the first sample and on that Age:22 and Fare:7.25, So we find 7.25 on x axis and 22 on y axis and put a dot (point) on the intersection
# 
# Example

# In[ ]:


import matplotlib.pyplot as plt
plt.scatter( x=[7.25],y=[22])
plt.show()


# Now how about we have 2 points Age:22 , Fare 7.25 and Age:30 and Fare: 19??
# 

# In[ ]:


import matplotlib.pyplot as plt
plt.scatter( x=[7.25, 19],y=[22,30])
plt.show()


# Just like that we will plot for all the data we have in data variable and the plot will look something like this.

# In[ ]:


import matplotlib.pyplot as plt
col=data["Survived"]
plt.scatter( x=data["Fare"],y=data["Age"], c=col)
plt.show()


# These points that we see on the graph are just GRAPHICAL representation of AGE AND FARE.Every single point over here corresponds to a row in our data, It corresponds to a sinlge person Age and Fare.
# Here Purple point implies that the person didnt survived and was dead in titanic and yellow point specifies that he is dead.
# ### Fair enough, Now if I gave you a new Person AGE=30 and FARE=500, can you tell me whether he will survive by looking at the graph.
# ### As the Name suggest K nearest neighbor just focus on the word Neighbor: 
# #### What we will do is this that will will take a new person AGE and FARE, try to find the point on graph where x=500 and Y=30 and simply put the point(dot) there.
# 
# #### We still have not predicted whether the person will survive or not, Try it your self look at the above graph and find a point where x=500 and y=35
# 
# ### If you have found the specified point, look for the point closest to that point?
# 
# #### Now look at the color of the point closest to the point (x,y)=(500,30)?
# ### What is the color?
# 
# #### YELLOW, Since it is close (neighbor ) to the point which survived hence we can classify that the new person will survive the titanic.
# 
# 
# 
# 
# 
# 

# # Congratulations, We just learned how to classify using K Nearest Neighbor with k=1.
# Now what is k?
# Recall that the prerequisite for this kernel is this that you know how voting works. 
# Assume that there are 3 people and two of them vote for Pizza and one of them votes for Burger. So by voting Pizza wins. 
# 
# 
# Similarly in K nearest neighbor we will do voting. 
# How?
# Lets consider a new point 250,55.
# Now we will try to find out with k=5. 
# 
# We will follow the same protocol we have followed before and find a point on graph where x=250 and y=55. 
# 
# NOW we find 5 colored points on graph closest to our new point!!.
# Fair enough, 
# After that we will look at the colors of the 5 nearest points to our new point.
# Lets say we find 5 nearest points and 3 of them have a purple color and 2 of them have a yellow color. 
# 
# Which means three of them died and 2 of them survived
# 
# Sadly our new point will be colored PURPLE and our algorithm says that new person will die. :(.
# 
# 
# Just like that if k=3, then we find two nearest points to our new point.
# 
# 
# We dont choose k to be even number, guess why?
# Because lets say we choose k=4 and the among the closest 4 points two of them are purple and two of them are yellow, What we will predict??.
# 
# 
# 
# Now lets come to implementation, Now by looking at the graph we visually analyzed that so and so point are closer to the new point but programatically how we will do it
# 
# We will use simple Euclidean Distance, Now what is this?
# ED is just a way of finding distance between two points in a graph
# E.G  P1=(1,25) and P2=(2,35)
# 
# Then Euclidean distance =   sqrt (    (1-2)^2   +   (25-35)^2  )
# 
# =sqrt(    1   +   100  )
# 
#   =sqrt(    101)
#   
#   
#   
#   Now lets try to code it, Here is the thing, We have an X_train which have co-ordinates of all the current points whole labels we have and an array of new_points which have some new points.
#  
#  What we will do first is that we will find distance (EUCLIDEAN DISTANCE) using the formula specified  of the new point to all the current points.
#  
# 
# 
# 

# In[ ]:


def compute_distances_one_loop(new_points, X_train):
    #X_Train have all of our training Samples, 
    # new_points is our new points which we want to predict
    num_test = len(new_points)
    num_train =X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
        
        difference = new_points[i] - X_train
        difference = np.square(difference)
        sum1 = np.sum(difference, axis=1)
        dists[i] = np.sqrt(sum1)
    return dists


# ## Let us try to plot the graph for our toy data set having only 4 people data of age and fare.

# In[ ]:


#Example:
X_train=np.array([[1,2],[2,3],[4,5], [6,7]])
Y_train=np.array([1,0,0,1])
col=Y_train
plt.scatter( x=X_train[:,0],y=X_train[:,1], c=col)
plt.show()


# In[ ]:


#OUR TESTING DATA
new_point=[[3,4]]
dists=compute_distances_one_loop(new_point, X_train)
print (dists)
print ("ARRAY DIMENSIONS IS :")
print (dists.shape)


# #### So we gave  testing data 3,4
# #### So our functions returns an array having one row and 4 columns. Now column 1 shows the distance with 1st row in X_train and column2 in result shows the distance with 2nd row in X_train  and so on.
# 
# Visually we can see that our point is closest to second and third rows in X_train data. AND SECOND AND THIRD ROW IN X_TRAIN DATA DIDNT SURVIVED. 
# Hence we can predict that [3,4] in our test data will not survive too.
# 
# 
# 

# In[ ]:


# Now that we have distance from each point, we will simple find out the distance which is
#least
def predict(dists, training_labels, k=3):
    closest_y = []
    rank = list(np.argsort(dists))
    for x in range(0, k):
        closest_y.append(training_labels[rank[x]])
    closest_y = np.asarray(closest_y)
    c=Counter(closest_y)
    return (c.most_common()[0][0])


# In[ ]:


result=predict(dists[0], Y_train)
print (result)


# #### Programatically we made the functions predict which will do the same task, we did visually 
# ### Now we will apply the same logic with our 891 training dataset rows

# In[ ]:


compute_distances_one_loop([[1,2]],data[["Fare","Age"]]).shape


# ## Since we gave one new point and 891 current points, So it gave us a new array with one row containing 891 columns specifying distance with each point.
# 
# 
# 
# ## FINALLY lets try it with our testing data and send it to kaggle for review

# In[ ]:


dists=compute_distances_one_loop(test_data,data[["Fare","Age"]])


# In[ ]:


Results=[]
for x in dists:
    Results.append(predict(x, data["Survived"]))


# In[ ]:


f=open("result.csv","w")
f.write("PassengerId,Survived")
for i in range(0, len(Results)):
    f.write("\n")
    f.write(str(passenger_id[i])+","+ str(Results[i]))
    
f.close()


# ### For a sanity check we will now just check whether our predictions are similar to scikit learn (inbuilt ) library or not.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data[["Fare","Age"]], data["Survived"])
scikit_result=neigh.predict(test_data)


count=0
notr=[]
for i in range(0, len(scikit_result)):
    if scikit_result[i]==Results[i]:
        count+=1
    else:
        notr.append(i)
print ("TOTAL INSTANCES")
print (len(scikit_result))
print ("SIMILAR RESULT B/W KNN AND SCIKIT LEARN INBUILT KNN")
print (count)
print ("")

print ("INDEXES OF WRONG RESULT")
print (notr)


# In[ ]:


Results[55]


# In[ ]:


scikit_result[55]


# In[ ]:


data.iloc[55]


# **Our self implemented knn was correct, Although there were two places where it gave different result, that might be due to distance measuring technique. ** 
# ## of all the 418 samples the result of self implemented knn and inbuilt knn on scikit learn, only 2 instances produced different results, else all results were same.
