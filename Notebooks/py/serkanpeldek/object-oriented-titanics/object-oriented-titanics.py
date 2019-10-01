#!/usr/bin/env python
# coding: utf-8

# * **The aim of this kernel is to provide a solution for data science projects by using an Object Oriented Programming(OOP) approach. Object oriented programming is largely based on personal experience and is open to development. For this reason, code design can be improved according to the comments to be made for the kernel. Comments and criticisms will provide a better code design.**
# 
# 
# **Many thanks for upvotes and feedbacks ^ . ^ ! **

# <a class="anchor" id="0."></a>
# **Contents**
# 
# 1. [OBJECT ORIENTED PROGRAMMING](#1.)
# * * [Design Goals](#1.1.)
# * *  [Design Prenciples](#1.2.)
# * * [Convention for Coding Style](#1.3.)
# * * [Starting Point of Object Oriented Programming: Class](#1.4.)
# * * [First Example: KaggleMember class](#1.5.)
# * * [Inheritance](#1.6.)
# 1. [WORKING ON TITANIC DATASET](#2.)
# * * ['Information' class](#2.1.)
# * *  ['Preprocess' class](#2.2.)
# * * ['PreprocessStrategy' class](#2.3.)
# * * ['GridSearchHelper' class](#2.4.)
# * * [Visualizer class](#2.e1.)
# * * ['ObjectOrientedTitanic' class](#2.5.)
# * * [Testing](#2.6.)
# * * * [Create ObjectOrientedTitanic object](#2.6.1.)
# * * * [Display R Type Information](#2.6.2.)
# * * * [Define preprocess strategy](#2.6.3.)
# * * * [Visualize](#2.6.4.)
# * * * [Get GridSearchCV Results](#2.6.5.)

# <a class="anchor" id="1."></a>**OBJECT ORIENTED PROGRAMMING** <=====>[Go to Contents Menu](#0.)
# 
# The key point in the object oriented paradigma is **object**.  Each **object** is an instance of **class**. Each class consists of a description that contains the properties and functions of the objects.
# 
# Typically a class includes:
# 
# **Instance variables**, also known as **data members**
# 
# **Methods**, also known as **member functions**

# <a class="anchor" id="1.1."></a>**Design Goals** <=====>[Go to Contents Menu](#0.)
# 
# The key point in the object oriented paradigma is
# 
# Design goals of object oriented paradigms are **robustness**, **adaptability** and **reusability**
# 
# **Robustness** is capablity of handling unexpexted input.
# 
# **Adaptability**(or evolvability or portability) is ability to run on different hadware and operating systems.
# 
# **Reusability** is opportunity of using same code as a component of different systems. 

# 
# <a class="anchor" id="1.2."></a>**Design Prencible** <=====>[Go to Contents Menu](#0.)
# 
# Design princeble of object oriented programming is **modularity, abstraction, encapsulation.**
# 
# **Modularity** is organizing principle of code into various functional units
# 
# **Abstraction** is to simplisfication  a complex inner machanism to fundemantel parts
# 
# **Encapsulation** mean that class inner details have not to be known by outer users

# <a class="anchor" id="1.3."></a>**Convention for Coding Style ** <=====>[Go to Contents Menu](#0.)
# 
# [The official Style Guide for Python Code](http://www.python.org/dev/peps/pep-0008/)
# 
# Python code blocks are typically indented by 4 spaces. It is strongly recommended that tabs be avoided
# 
# Use meaningful names for variables, funcitons, classes etc.
# 
# **Classes**: should be singular and capitalized (e.g., 'Book' rather than 'book' or 'Books'
# 
# **Functions**: sould be a lowercase verb and separeted by underscore for multple words (e.g., preprocess_data).
# 
# **Variables**: sould be lowercase noun and separeted by underscore for multiple words (e.g., product_id)
# 
# **Constant**: sould be uppercase noun and separeted by underscore for multiple words (e.g., MAX_SIZE)

# <a class="anchor" id="1.4."></a>**Starting Point of Object Oriented Programming: Class** <=====>[Go to Contents Menu](#0.)
# 
# A class play as fundemental role for abstraction in OOP. 
# 
# **Mebmer functions** (also known as methods) perform set of actions ,with implementations that are common to all instances of that class. 
# 
# **Attributes** (also known as fields, instance variables, or data members) determine the information for each instance.
# 
# **Instance**:  An individual object of a certain class. An object obj that belongs to a class Circle, for example, is an instance of the class Circle.
# 
# **Class variable**: A variable that is shared by all instances of a class. Class variables are attributes which are owned by the class itself. They will be shared by all the instances of the class. Therefore they have the same value for every instance. 
# 
# **Instance variable: **A variable that is defined inside a method and belongs only to the current instance of a class. Instance variables are owned by the specific instances of a class. This means for two different instances the instance variables are usually
# different.
# 
# **Constructor: **A special funciton which is called when a new instance of class created. 
# 
# 
# 

# <a class="achor" id="1.5."></a>**First Example: KaggleMember class** <=====>[Go to Contents Menu](#0.)
# 
# Let's have some code to understand above definitions

# In[ ]:


class KaggleMember(): #line1
    #class member
    kaggle_member_count=0  #line2
        
    def __init__(self, name, surname, level=None): #line3
        
        #object members
        self.name=name.capitalize() #line4
        self.surname=surname.upper() #line5
        self._set_level(level) #6
        
        KaggleMember.kaggle_member_count+=1 #line7
        self.kaggle_id=KaggleMember.kaggle_member_count #line8
        
        
    
    def display_number_of_member(self): #line11
        print("There are {} members in Kaggle community".format(KaggleMember.kaggle_member_count)) #line12
    
    def display_member_info(self): #line13
        print("Kaggle Member Full Name:{} {}".format(self.name, self.surname)) #line14
        print("Level:{:15}".format(self.level)) #line15
        print("Kaggle ID:",self.kaggle_id)
    
    def _set_level(self, level):
        if level is None: #line6
            self.level='Novice' #line7
        else: #line8
            self.level=level.title() #line9
        
    
    myclass_name='Kaggle Member' #line16
    


# Let's explain the code above. Numbers are given at the end of each line to make the descriptions easier (eg, # line1)
# 
# First, we create a class named KaggleMember using the 'class' key word. The bracket symbols '()' are added at the end of the class name. Python allows class definition without adding parentheses (eg, KaggleMember). However, since most programming languages require the addition of parentheses symbols after the class name, the use of parentheses  has been preferred in this class definition
# 
# In the second line, the class variable named kaggle_member_count is defined. Class variables are defined outside of class functions. However, it is a good programming practice to write class variables just below the class name. At the bottom of the KaggleMember class, a class variable defined for learning purposes
# 
# Constructor function is defined in line 3. In Python, the constructor function has a special name: __init __ (). Constructor functions are invoked on every instance of the class, and are generally used to assign initial values. In the above example, the constructor function has four different parameters: self, name, surname, level.
# * The first argument of every class method, including init, is always a reference to the current instance of the class. The word 'self' should be written as the first parameter of each function created in the Python class definition. It is important to note that; The word 'self' can be replaced with another word based on user preference. By convention, this argument is always named self.
# * There are three parameters that come after 'self'. The second and third parameters have no default value. For this reason, these parameters must be sent when the function is called. The fourth parameter has a default value. For this reason, if the fourth parameter is not sent when the function is called, it will get its default value. For the example above, it will take the value None.
# 
# In lines 4, 5, 6 and 8, the initial assignments of instance variables were made with values from the constructor function. Note that when values are assigned to variables, different inputs are transformed into the desired form. This input control is a small example for ** Robustness ** mentioned above.
# 
# In the sixth line, another instance variable is assigned a value. A function has been used for value assignment. The function assigns the value of the 'self.level' variable according to the value of the 'level' parameter.
# 
# The remaining lines contain the functions used.
# 
# Let's create object using KaggleMember class
# 

# In[ ]:


kaggler1=KaggleMember('SERkan','peldek','contributor')
kaggler1.display_member_info()
kaggler1.display_number_of_member()
print()
kaggler2=KaggleMember('kaan','can', 'grand master')
kaggler2.display_member_info()
kaggler1.display_number_of_member()
print()
kaggler3=KaggleMember('Kral','Adam')
kaggler3.display_member_info()
kaggler1.display_number_of_member()

print()
print('Following codes check other class variable value ')
print(kaggler1.myclass_name)
print(kaggler2.myclass_name)
print(kaggler3.myclass_name)


# Let's examine the codes above.
# Three different KaggleMember objects were created. The same process is performed for each object:
# 
# When creatin Kaggler1 object, three parameters were sent to constructor function; 'SeRKaN', 'peldek', 'contributor'. Then the display_member_info () function of the kaggler1 object is called. The function prints the information of the object kaggler1. The function kaggler1.display_number_of_member () prints the number of members.
# 
# When creating kaggler2 object, three parameters were sent to constructor function; 'kaan', 'can', 'grand master'.
# 
# When creating kaggler3 object, three parameters were sent to constructor function; 'King man'. Note that, after each object is created, the number of objects created is correctly given even though the function **kaggler1.display_number_of_member () **is called.

# <a  class="anchor" id="1.6."></a>**Inheritance** <=====>[Go to Contents Menu](#0.)
# 
# One of the most important aspects of OOP is the ability to use previously written code. Inheritance is a perfect way to go. In fact, every object we created in Python inherits methods from "**object**" which is **super** class of all Python class. Let's understand how it is by examining the following code.
# 

# In[ ]:


class MyDefinedClass:#Implicitly inheritance  
    pass

class YourDefinedClass(object):#Explicityl inheritance
    pass

print("The methods in MyDefinedClass:\n",dir(MyDefinedClass))
print()
print("The methods in YourDefinedClass:\n",dir(MyDefinedClass))
print()
print("The methods in 'object':\n",dir(object))


# Two different classes are defined in the above code. One is implicitly inherited while the other is explicitly  inherited. Using the predefined '**dir**' function, the methods belong to MyDefinedClass, YourDefinedClass and  '**object**'  is predefined class in Python, are printed on the screen. The methods of the three classes are the same.
# 
# If there are user-defined methods in the class, the dir function will show them too. The **methods** and **attributes** of the KaggleMember class are shown below. The **methods** and **attributes** we have described are at the end of the list.

# In[ ]:


print("The methods in KaggleMember:\n",dir(KaggleMember))


#  <a class="anchor" id="2."></a>**WORKING ON TITANIC DATASET** <=====>[Go to Contents Menu](#0.)
# 
# Now let's work on the Titanic dataset using classes
# 
# 
# Different classes have been created for the operations on the Titanic dataset; Information, Preprocess, PreprocessStrategy, GridSearchHelper, ObjectOrientedTitanic
# 
# '**Information**': This class prints summary information about the data set on the screen.
# 
# '**Preprocess**': The preprocessing on the data set is done using this class.
# 
# '**PreprocessStrategy**': Preprocessing is important in the Titanic data set. The PreprocessStrategy class was created to develop different pre-processing strategies.
# 
# '**GridSearchHelper**': Class for parameter optimization for machine learning algorithms.
# 
# '**ObjectOrientedTitanic**': The class for which all classes are managed.
# 
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing

#Visualization
import matplotlib.pyplot as plt

#Systems
import os
import warnings
print(os.listdir("../input"))


# In[ ]:


warnings.filterwarnings('ignore')
print("Warnings were ignored")


# <a class="anchor" id="2.1."></a>**'Information' class**   <=====>[Go to Contents Menu](#0.)

# In[ ]:


class Information():

    def __init__(self):
        """
        This class give some brief information about the datasets.
        Information introduced in R language style
        """
        print("Information object created")

    def _get_missing_values(self,data):
        """
        Find missing values of given datad
        :param data: checked its missing value
        :return: Pandas Series object
        """
        #Getting sum of missing values for each feature
        missing_values = data.isnull().sum()
        #Feature missing values are sorted from few to many
        missing_values.sort_values(ascending=False, inplace=True)
        
        #Returning missing values
        return missing_values

    def info(self,data):
        """
        print feature name, data type, number of missing values and ten samples of each feature
        :param data: dataset information will be gathered from
        :return: no return value
        """
        feature_dtypes=data.dtypes
        self.missing_values=self._get_missing_values(data)

        print("=" * 50)

        print("{:16} {:16} {:25} {:16}".format("Feature Name".upper(),
                                            "Data Format".upper(),
                                            "# of Missing Values".upper(),
                                            "Samples".upper()))
        for feature_name, dtype, missing_value in zip(self.missing_values.index.values,
                                                      feature_dtypes[self.missing_values.index.values],
                                                      self.missing_values.values):
            print("{:18} {:19} {:19} ".format(feature_name, str(dtype), str(missing_value)), end="")
            for v in data[feature_name].values[:10]:
                print(v, end=",")
            print()

        print("="*50)


# <a class="anchor" id="2.2."></a>**'Preprocess' class**   <=====>[Go to Contents Menu](#0.)

# In[ ]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
class Preprocess():

    def __init__(self):
        print("Preprocess object created")

    def fillna(self, data, fill_strategies):
        for column, strategy in fill_strategies.items():
            if strategy == 'None':
                data[column] = data[column].fillna('None')
            elif strategy == 'Zero':
                data[column] = data[column].fillna(0)
            elif strategy == 'Mode':
                data[column] = data[column].fillna(data[column].mode()[0])
            elif strategy == 'Mean':
                data[column] = data[column].fillna(data[column].mean())
            elif strategy == 'Median':
                data[column] = data[column].fillna(data[column].median())
            else:
                print("{}: There is no such thing as preprocess strategy".format(strategy))

        return data

    def drop(self, data, drop_strategies):
        for column, strategy in drop_strategies.items():
            data=data.drop(labels=[column], axis=strategy)

        return data

    def feature_engineering(self, data, engineering_strategies=1):
        if engineering_strategies==1:
            return self._feature_engineering1(data)

        return data

    def _feature_engineering1(self,data):

        data=self._base_feature_engineering(data)


        data['FareBin'] = pd.qcut(data['Fare'], 4)

        data['AgeBin'] = pd.cut(data['Age'].astype(int), 5)

        drop_strategy = {'Age': 1,  # 1 indicate axis 1(column)
                         'Name': 1,
                         'Fare': 1}
        data = self.drop(data, drop_strategy)

        return data

    def _base_feature_engineering(self,data):
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

        data['IsAlone'] = 1
        data.loc[(data['FamilySize'] > 1), 'IsAlone'] = 0

        data['Title'] = data['Name'].str.split(", ", expand=True)[1].str.split('.', expand=True)[0]
        min_lengtht = 10
        title_names = (data['Title'].value_counts() < min_lengtht)
        data['Title'] = data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

        return data

    def _label_encoder(self,data):
        labelEncoder=LabelEncoder()
        for column in data.columns.values:
            if 'int64'==data[column].dtype or 'float64'==data[column].dtype or 'int64'==data[column].dtype:
                continue
            labelEncoder.fit(data[column])
            data[column]=labelEncoder.transform(data[column])
        return data

    def _get_dummies(self, data, prefered_columns=None):

        if prefered_columns is None:
            columns=data.columns.values
            non_dummies=None
        else:
            non_dummies=[col for col in data.columns.values if col not in prefered_columns ]

            columns=prefered_columns


        dummies_data=[pd.get_dummies(data[col],prefix=col) for col in columns]

        if non_dummies is not None:
            for non_dummy in non_dummies:
                dummies_data.append(data[non_dummy])

        return pd.concat(dummies_data, axis=1)


# <a class="anchor" id="2.3."></a>**'PreprocessStrategy' class**   <=====>[Go to Contents Menu](#0.)

# In[ ]:


class PreprocessStrategy():
    """
    Preprocess strategies defined and exected in this class
    """
    def __init__(self):
        self.data=None
        self._preprocessor=Preprocess()

    def strategy(self, data, strategy_type="strategy1"):
        self.data=data
        if strategy_type=='strategy1':
            self._strategy1()
        elif strategy_type=='strategy2':
            self._strategy2()

        return self.data

    def _base_strategy(self):
        drop_strategy = {'PassengerId': 1,  # 1 indicate axis 1(column)
                         'Cabin': 1,
                         'Ticket': 1}
        self.data = self._preprocessor.drop(self.data, drop_strategy)

        fill_strategy = {'Age': 'Median',
                         'Fare': 'Median',
                         'Embarked': 'Mode'}
        self.data = self._preprocessor.fillna(self.data, fill_strategy)

        self.data = self._preprocessor.feature_engineering(self.data, 1)


        self.data = self._preprocessor._label_encoder(self.data)

    def _strategy1(self):
        self._base_strategy()

        self.data=self._preprocessor._get_dummies(self.data,
                                        prefered_columns=['Pclass', 'Sex', 'Parch', 'Embarked', 'Title', 'IsAlone'])

    def _strategy2(self):
        self._base_strategy()

        self.data=self._preprocessor._get_dummies(self.data,
                                        prefered_columns=None)#None mean that all feature will be dummied


# <a class="anchor" id="2.4."></a>**'GridSearchHelper' class**   <=====>[Go to Contents Menu](#0.)

# In[ ]:


import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
class GridSearchHelper():
    def __init__(self):
        print("GridSearchHelper Created")

        self.gridSearchCV=None
        self.clf_and_params=list()

        self._initialize_clf_and_params()

    def _initialize_clf_and_params(self):

        clf= KNeighborsClassifier()
        params={'n_neighbors':[5,7,9,11,13,15],
          'leaf_size':[1,2,3,5],
          'weights':['uniform', 'distance']
          }
        self.clf_and_params.append((clf, params))

        clf=LogisticRegression()
        params={'penalty':['l1', 'l2'],
                'C':np.logspace(0, 4, 10)
                }
        self.clf_and_params.append((clf, params))

        clf = SVC()
        params = [ {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                   {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
        self.clf_and_params.append((clf, params))

        clf=DecisionTreeClassifier()
        params={'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15],
          'min_samples_leaf':[1],
          'random_state':[123]}
        #Because of depricating warning for Decision Tree which is not appended.
        #But it give high competion accuracy score. You can append when you run the kernel
        #self.clf_and_params.append((clf,params))

        clf = RandomForestClassifier()
        params = {'n_estimators': [4, 6, 9],
              'max_features': ['log2', 'sqrt','auto'],
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10],
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }
        #Because of depricating warning for RandomForestClassifier which is not appended.
        #But it give high competion accuracy score. You can append when you run the kernel
        #self.clf_and_params.append((clf, params))

    def fit_predict_save(self, X_train, X_test, y_train, submission_id, strategy_type):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.submission_id=submission_id
        self.strategy_type=strategy_type

        clf_and_params = self.get_clf_and_params()

        for clf, params in clf_and_params:
            self.current_clf_name = clf.__class__.__name__
            grid_search_clf = GridSearchCV(clf, params, cv=5)
            grid_search_clf.fit(self.X_train, self.y_train)
            self.Y_pred = grid_search_clf.predict(self.X_test)
            clf_train_acc = round(grid_search_clf.score(self.X_train, self.y_train) * 100, 2)
            print(self.current_clf_name, " train accuracy:", clf_train_acc)

            self.save_result()
            print()

    def save_result(self):
        Submission = pd.DataFrame({'PassengerId': self.submission_id,
                                           'Survived': self.Y_pred})
        file_name="{}_{}.csv".format(self.strategy_type,self.current_clf_name.lower())
        Submission.to_csv(file_name, index=False)

        print("Submission saved file name: ",file_name)

    def get_clf_and_params(self):

        return self.clf_and_params

    def add(self,clf, params):
        self.clf_and_params.append((clf, params))


# <a class="anchor" id="2.e1."></a>**Visualizer class** <=====>[Go to Content Menu](#0.)

# In[ ]:


from yellowbrick.features import RadViz
class Visualizer:
    
    def __init__(self):
        print("Visualizer object created!")
    
    def RandianViz(self, X, y, number_of_features):
        if number_of_features is None:
            features=X.columns.values
        else:
            features=X.columns.values[:number_of_features]
        
        fig, ax=plt.subplots(1, figsize=(15,12))
        radViz=RadViz(classes=['survived', 'not survived'], features=features)
        
        radViz.fit(X, y)
        radViz.transform(X)
        radViz.poof()
        


# <a class="anchor" id="2.5."></a>**'ObjectOrientedTitanic' class**   <=====>[Go to Contents Menu](#0.)

# In[ ]:


class ObjectOrientedTitanic():

    def __init__(self, train, test):
        """

        :param train: train data will be used for modelling
        :param test:  test data will be used for model evaluation
        """
        print("ObjectOrientedTitanic object created")
        #properties
        self.testPassengerID=test['PassengerId']
        self.number_of_train=train.shape[0]

        self.y_train=train['Survived']
        self.train=train.drop('Survived', axis=1)
        self.test=test

        #concat train and test data
        self.all_data=self._get_all_data()

        #Create instance of objects
        self._info=Information()
        self.preprocessStrategy = PreprocessStrategy()
        self.visualizer=Visualizer()
        self.gridSearchHelper = GridSearchHelper()
        


    def _get_all_data(self):
        return pd.concat([self.train, self.test])

    def information(self):
        """
        using _info object gives summary about dataset
        :return:
        """
        self._info.info(self.all_data)



    def preprocessing(self, strategy_type):
        """
        Process data depend upon strategy type
        :param strategy_type: Preprocessing strategy type
        :return:
        """
        self.strategy_type=strategy_type

        self.all_data = self.preprocessStrategy.strategy(self._get_all_data(), strategy_type)

    def visualize(self, visualizer_type, number_of_features=None):
        
        self._get_train_and_test()
        
        if visualizer_type=="RadViz":
            self.visualizer.RandianViz(X=self.X_train, 
                                    y=self.y_train, 
                                    number_of_features=number_of_features)

    def machine_learning(self):
        """
        Get self.X_train, self.X_test and self.y_train
        Find best parameters for classifiers registered in gridSearchHelper
        :return:
        """
        self._get_train_and_test()

        self.gridSearchHelper.fit_predict_save(self.X_train,
                                          self.X_test,
                                          self.y_train,
                                          self.testPassengerID,
                                          self.strategy_type)



    def _get_train_and_test(self):
        """
        Split data into train and test datasets
        :return:
        """
        self.X_train=self.all_data[:self.number_of_train]
        self.X_test=self.all_data[self.number_of_train:]


# <a class="anchor" id="2.6."></a>**'Testing**   <=====>[Go to Contents Menu](#0.)

# <a class="anchor" id="2.6.1."></a>**Create  ObjectOrientedTitanic object**  <=====>[Go to Contents Menu](#0.)

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

objectOrientedTitanic=ObjectOrientedTitanic(train, test)


# <a class="anchor" id="2.6.2."></a>**Display R Type Information**  <=====>[Go to Contents Menu](#0.)

# In[ ]:


objectOrientedTitanic.information()


# <a class="anchor" id="2.6.3."></a>**Define preprocess strategy**  <=====>[Go to Contents Menu](#0.)

# In[ ]:


#There are currently two strategy type: strategy1, strategy2.
#We can select any of two
objectOrientedTitanic.preprocessing(strategy_type='strategy1')


# In[ ]:


objectOrientedTitanic.information()


# <a class="anchor" id="2.6.4."></a>**Visualize**  <=====>[Go to Contents Menu](#0.)

# In[ ]:


#Radian Viz figure
objectOrientedTitanic.visualize(visualizer_type="RadViz", number_of_features=None)


# <a class="anchor" id="2.6.5."></a>**Get GridSearchCV Results**  <=====>[Go to Contents Menu](#0.)

# In[ ]:


objectOrientedTitanic.machine_learning()


# **Notes:**
# 
# Please note, this kernel is currently in progress, but open to feedback and upvotes  (o____o)  ! Thanks!
