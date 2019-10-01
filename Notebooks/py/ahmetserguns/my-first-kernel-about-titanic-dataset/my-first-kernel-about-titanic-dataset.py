#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder,RobustScaler,MinMaxScaler,Normalizer
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV, StratifiedKFold, learning_curve

from speedml import Speedml


# In[ ]:


train_dataset=pd.read_csv("../input/train.csv")
test_dataset=pd.read_csv("../input/test.csv")
datasets=[train_dataset,test_dataset]


# In[ ]:


train_dataset.head()


# In[ ]:


for dataset in datasets: # datasets : 1st train_dataset, 2nd test_dataset
    total = dataset.isnull().sum().sort_values(ascending=False)
    percent_1 = dataset.isnull().sum()/dataset.isnull().count()*100
    percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
    missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
    print(missing_data.head(5))


# # DATA PROCESSING

# In[ ]:



def clean_data(dataset):
    
    
    #------ Embarked---------------
    dataset["Embarked"]=dataset["Embarked"].fillna("S")
    
    #-------------Fare-------------
    dataset.loc[dataset.Fare==0, "Fare"]="NaN" # Fare'e str değeri atadığımız için artık object'e dönüştü!!
    dataset["Fare"]=dataset["Fare"].astype(float) # Fare datasını yeniden float'a çeviriyoruz ki aritmetik hesaplamalar yapılabilelim
    dataset["Fare"].fillna(dataset.groupby("Pclass")["Fare"].transform("median"), inplace=True)
 
    #-------------Name-------------
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]   
    
    #-------------Age-------------
    
    dataset["Age"].fillna(dataset.groupby("Title")["Age"].transform("median"), inplace=True)
    #dataset['Age'] = dataset['Age'].astype(int)
    #-------------Title-------------
    dataset['Title'].replace(['Mlle','Mme','Ms',"Capt","Col","Don","Jonkheer","Rev","Major","Dr","Lady","Sir","the Countess"],
                               ["Miss","Miss","Mrs","Mr","Mr","Mr","Mr","Mr","Mr","Mr","Loyal","Loyal","Loyal"],inplace=True)
    
    #-------------Cabin-------------
    #dataset['Cabin'].fillna('U', inplace=True)
    #dataset['Cabin'] = dataset['Cabin'].apply(lambda x: x[0])
    
    #import re
    #deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
    #dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    #dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    #dataset['Deck'] = dataset['Deck'].map(deck)
    #dataset['Deck'] = dataset['Deck'].fillna(0)
    #dataset['Deck'] = dataset['Deck'].astype(int)
 
    dataset.drop(['Cabin'], axis=1,inplace = True)
        
   #-------------Age-------------
    #bins = [0, 11, 18, 22, 28, 33, 40, 66,100]
    #group_names = ['0-11', '12-18', '19-22', '20-28','28-33',"34-40","41-66","67-100" ]
    #dataset['AgeScala'] = pd.cut(train_dataset['Age'], bins, labels=group_names)
    #dataset["AgeScala"]=pd.qcut(train_dataset["Age"], 7).value_counts()
   
    #dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    #dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    #dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    #dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    #dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    #dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    #dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    #dataset.loc[ dataset['Age'] > 66, 'Age'] = 7
    #dataset['Age'] = dataset['Age'].astype(int)
    #
    #dataset.loc[ dataset['Fare'] <= 7.775, 'Fare'] = 0
    #dataset.loc[(dataset['Fare'] > 7.75) & (dataset['Fare'] <= 8.7), 'Fare'] = 1
    #dataset.loc[(dataset['Fare'] > 8.7) & (dataset['Fare'] <= 14.454), 'Fare']   = 2
    #dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 26), 'Fare']   = 3
    #dataset.loc[(dataset['Fare'] > 26) & (dataset['Fare'] <= 52), 'Fare']   = 4
    #dataset.loc[ dataset['Fare'] > 52, 'Fare'] = 5
    #dataset['Fare'] = dataset['Fare'].astype(int)

    #-------------Alone or Not-------------
    #dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    #dataset.loc[dataset['relatives'] > 0, 'alone'] = 0
    #dataset.loc[dataset['relatives'] == 0, 'alone'] = 1
    #dataset['alone'] = dataset['alone'].astype(int)  
    
    dataset['Age_Class']= dataset['Age']*dataset['Pclass']
    
if __name__=="__main__":
    clean_data(train_dataset)
    clean_data(test_dataset)


# In[ ]:


#---- Silinecek değişkenler-------
drop_column = ['PassengerId', 'Ticket', 'Name']
train_dataset.drop(drop_column, axis=1, inplace = True)
drop_column_ = ['Ticket', 'Name']
test_dataset.drop(drop_column_,axis=1, inplace=True)


# In[ ]:


#train_dataset.to_csv('deneme.csv',index=True,header=True)


# In[ ]:


columns_=["Age","Fare","Title","Deck","Pclass","SibSp","Parch","Sex","Age_Class","Embarked"]   
columns__=["Sex","Embarked","Title","Age_Class"]

for dataset in datasets:
    for i in columns__:
        dataset[i] = LabelEncoder().fit_transform(dataset[i])  # dönüşüm
      
    #for j in columns_:
    #    dataset[j]=dataset[j].astype(float)   # int to float dönüşüm uyarısını almamak için
    
    #for k in columns_:
       
    #    dataset[k] = StandardScaler().fit_transform(dataset[k].values.reshape(-1, 1))  # dönüşüm
        
    


# In[ ]:


train_dataset.sample(3)


# In[ ]:


test_dataset.head()


# # MODEL CALCULATIONS

# In[ ]:


variables_=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Title","Age_Class"]

features=train_dataset[variables_].values

target=train_dataset["Survived"].values

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=0)


# In[ ]:


y_test.shape


# In[ ]:


# train_dataset.corr()
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(train_dataset.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:



#Ensemble Methods
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

#Gaussian Processes
from sklearn.gaussian_process import GaussianProcessClassifier   

#GLM
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifierCV
#from sklearn.linear_model import SGDClassifier
#from sklearn.linear_model import Perceptron

#Navies Bayes
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

#Nearest Neighbor
from sklearn.neighbors import KNeighborsClassifier

#Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

#Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Support Vector
from sklearn import svm


#Neural Network
from sklearn.neural_network import MLPClassifier


#===========================================================================================================================

models = []

#Ensemble Methods
models.append(("AdaBoostClassifier", AdaBoostClassifier))
models.append(("BaggingClassifier", BaggingClassifier))
models.append(("RandomForestClassifier", RandomForestClassifier))
models.append(("GradientBoostingClassifier",  GradientBoostingClassifier))
models.append(("XGBClassifier",  XGBClassifier))


#Gaussian Processes
models.append(("GaussianProcessClassifier",  GaussianProcessClassifier))

#GLM
#models.append(("LinearRegression", LinearRegression))
models.append(("LogisticRegression", LogisticRegression))
#models.append(("LogisticRegressionCV", LogisticRegressionCV))
#models.append(("PassiveAggressiveClassifier",  PassiveAggressiveClassifier))
models.append(("RidgeClassifierCV",  RidgeClassifierCV))
#models.append(("SDK",  SGDClassifier))
#models.append(("PERS",  Perceptron))

#Navies Bayes
models.append(("BernoulliNB",  BernoulliNB))
models.append(("GaussianNB",  GaussianNB))

#Nearest Neighbor
models.append(("KNeighborsClassifier", KNeighborsClassifier))

#Tree
models.append(("DecisionTreeClassifier", DecisionTreeClassifier))              
models.append(("ExtraTreesClassifier", ExtraTreesClassifier))               

#Discriminant Analysis
models.append(("LinearDiscriminantAnalysis", LinearDiscriminantAnalysis))
models.append(("QuadraticDiscriminantAnalysis", QuadraticDiscriminantAnalysis))

# Support Vector
#models.append(("Support Vector Regression", svm))

#Neural network
#models.append(("MLPClassifier", MLPClassifier))


# In[ ]:


aim_score= 0.80         # determine success rate exam: 0.80

for description in models:
    short_name, model_name = description
    classifier = model_name()

    classifier.fit(x_train, y_train)             # eğitim
    y_pred = classifier.predict(x_test)          # tahmin
    score = accuracy_score(y_test, y_pred)       # tahmin ile test için ayrılan set karşılaştırılıyor
    #score1 = classifier.score(x_test, y_test)
    
    # cv'de belirtilen deneme sonrasında accuracy sonucunun ne olduğunu ölçüyoruz. Böylece ilk başta çok başarılı gibi
    # görünen modellerin gerçek değerlerine ulaşmış oluyoruz.
    #scores.mean()
        
    if score > aim_score:
            conf_matrix_ = confusion_matrix(y_test,y_pred)
            true_answer=conf_matrix_[0][0]+conf_matrix_[1][1]
            wrong_answer=conf_matrix_[0][1]+conf_matrix_[1][0]
            total=true_answer+wrong_answer
            scores=model_selection.cross_val_score(classifier, features, target, scoring="accuracy", cv=5)
            print(classification_report(y_test, y_pred))
            print("{} --> {:.2f} --> {:.2f} --> Total: {} ~ True: {}  ".format(short_name, score, scores.mean(), total,true_answer))
            print("--------------------")


# # Building Machine Learning Models (after try=50)

# işlemcisine güvenen cv değerini artırabilir.

# In[ ]:


gradient=GradientBoostingClassifier()
gradient.fit(x_train,y_train)
Y_pred=gradient.predict(x_test)
acc_gradinet=round(gradient.score(x_train, y_train)*100,2)
print(acc_gradinet)
score_gradinet=model_selection.cross_val_score(gradient, features, target, scoring="accuracy", cv=50)
print("After try: {}".format(round(score_gradinet.mean()*100,2)))


# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train) 
Y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train, y_train)*100,2)
print(acc_decision_tree)
score_decision_tree=model_selection.cross_val_score(decision_tree, features, target, scoring="accuracy", cv=50)
print("After try: {}".format(round(score_decision_tree.mean()*100,2)))


# In[ ]:


XG=XGBClassifier()
XG.fit(x_train,y_train)
Y_pred=XG.predict(x_test)
acc_XGBC=round(XG.score(x_train, y_train)*100,2)
print(acc_XGBC)
score_XGBC=model_selection.cross_val_score(XG, features, target, scoring="accuracy", cv=50)
print("After try: {}".format(round(score_XGBC.mean()*100,2)))


# In[ ]:


from sklearn.linear_model import SGDClassifier
from sklearn import linear_model
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(x_train, y_train)
Y_pred = sgd.predict(x_test)
sgd.score(x_train, y_train)
acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)
print(acc_sgd)
score_sgd=model_selection.cross_val_score(sgd, features, target, scoring="accuracy", cv=50)
print("After try: {}".format(round(score_sgd.mean()*100,2)))


# In[ ]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
Y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
print(acc_log)
score_log=model_selection.cross_val_score(logreg, features, target, scoring="accuracy", cv=50)
print("After try: {}".format(round(score_log.mean()*100,2)))


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train) 
Y_pred = knn.predict(x_test)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)
print(acc_knn)
score_knn=model_selection.cross_val_score(knn, features, target, scoring="accuracy", cv=50)
print("After try: {}".format(round(score_knn.mean()*100,2)))


# In[ ]:


gaussian = GaussianNB()
gaussian.fit(x_train, y_train) 
Y_pred = gaussian.predict(x_test) 
acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)
print(acc_gaussian)
score__gaussian=model_selection.cross_val_score(gaussian, features, target, scoring="accuracy", cv=50)
print("After try: {}".format(round(score__gaussian.mean()*100,2)))


# In[ ]:


from sklearn.linear_model import Perceptron
perceptron = Perceptron(max_iter=5)
perceptron.fit(x_train, y_train)
Y_pred = perceptron.predict(x_test)
acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)
print(acc_perceptron)
score_perceptron=model_selection.cross_val_score(perceptron, features, target, scoring="accuracy", cv=50)
print("After try:{}".format(round(score_perceptron.mean()*100,2)))


# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
Y_pred = decision_tree.predict(x_test) 
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
print(acc_decision_tree)
score_decision_tree_svc=model_selection.cross_val_score(decision_tree, features, target, scoring="accuracy", cv=50)
print("After try: {}".format(round(score_decision_tree.mean()*100,2)))


# # Random Forest

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100,oob_score = True)
random_forest.fit(x_train, y_train)
Y_prediction = random_forest.predict(x_test)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
print(acc_random_forest)
score_random_forest=model_selection.cross_val_score(random_forest, features, target, scoring="accuracy", cv=50)
print("After try: {}".format(round(score_random_forest.mean()*100,2)))


# In[ ]:


print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


# In[ ]:


random_forest = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

random_forest.fit(x_train, y_train)
Y_prediction = random_forest.predict(x_test)

random_forest.score(x_train, y_train)

print("oob score:", round(random_forest.oob_score_, 4)*100, "%")


# In[ ]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, x_train, y_train, cv=3)
confusion_matrix(y_train, predictions)


# In[ ]:


from sklearn.metrics import precision_score, recall_score,f1_score

print("Precision\t:", round(precision_score(y_train, predictions)*100,2))
print("Recall\t:",round(recall_score(y_train, predictions)*100,2))
print("f1 score\t:", round(f1_score(y_train, predictions)*100,2))


# In[ ]:


#Apply our prediction to test data
#predictions = gradient.predict(test_dataset[variables_])

# Create a new dataframe with only the columns Kaggle wants from the dataset
#submission_DFs = pd.DataFrame({ 
#    "PassengerId" : test_dataset["PassengerId"],
#    "Survived" : predictions
    })
#print(submission_DFs.head(2))


# In[ ]:


# prepare file for submission
#submission_DFs.to_csv("submission_gradient.csv", index=False)


# In[ ]:


# Only for XGRB
#Apply our prediction to test data
#predictionx = XG.predict(test_dataset[variables_].as_matrix())

# Create a new dataframe with only the columns Kaggle wants from the dataset
#submission_DFx = pd.DataFrame({ 
#    "PassengerId" : test_dataset["PassengerId"],
#    "Survived" : predictionx
    })
#print(submission_DFx.head(2))


# In[ ]:


# prepare file for submission
#submission_DFx.to_csv("submission_XBGC.csv", index=False)


# # -------------------------------------------------/-------------------------------------------------------

# Precision:
# Our model predicts 72 % of the time, a passengers survival correctly (precision).
# 
# Recall:
# The recall tells us that it predicted the survival of 72 % of the people who actually survived.
# 
# F-Score:
# You can combine precision and recall into one score, which is called the F-score. The F-score is computed with the harmonic mean of precision and recall. Note that it assigns much more weight to low values. As a result of that, the classifier will only get a high F-score, if both recall and precision are high.

# # Precision Recall Curve

# In[ ]:


from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = random_forest.predict_proba(x_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()


# # ROC AUC Curve
# The red line in the middel represents a purely random classifier (e.g a coin flip) and therefore your classifier should be as far away from it as possible. 

# In[ ]:


from sklearn.metrics import roc_curve
# compute true positive rate and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)
# plotting them against each other
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()


# The ROC AUC Score is the corresponding score to the ROC AUC Curve.
# It is simply computed by measuring the area under the curve, which is called AUC.
# A classifiers that is 100% correct, would have a ROC AUC Score of 1
# and a completely random classiffier would have a score of 0.5.

# In[ ]:


from sklearn.metrics import roc_auc_score
r_a_score = roc_auc_score(y_train, y_scores)
print("ROC-AUC-Score: {:2f}".format( r_a_score))


# In[ ]:


#results = pd.DataFrame({
#    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
#              'Random Forest', 'Naive Bayes', 'Perceptron', 
#              'Stochastic Gradient Decent', 
#              'Decision Tree'],
#    'Score': [acc_linear_svc, acc_knn, acc_log, 
#              acc_random_forest, acc_gaussian, acc_perceptron, 
#              acc_sgd, acc_decision_tree],
#    })
#result_df = results.sort_values(by='Score', ascending=False)
#result_df = result_df.set_index('Score')
#result_df.head(9)


# # Keras

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical

output = train_dataset['Survived'].values
output = to_categorical(output, 2)

X_train, X_validation, y_train, y_validation = train_test_split(features, output, test_size=0.05)
import numpy
numpy.random.seed(7)


# In[ ]:


model = Sequential()
model.add(Dense(32, input_dim=9, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


# In[ ]:


#model.summary()


# In[ ]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.sgd(),
              metrics=['accuracy'])


# In[ ]:


model.fit(X_train, y_train, 
          batch_size=200, 
          epochs=100)


# In[ ]:


acc = model.evaluate(X_validation, y_validation)
print('Hata Toplami(LOSS):', acc[0])
print('Basari(ACC):', acc[1])


# In[ ]:


#prediction_= model.predict_classes(test_dataset[variables_])


# In[ ]:


#submission = pd.DataFrame()
#submission['PassengerId'] = test_dataset["PassengerId"]
#submission['Survived'] = prediction_


# In[ ]:


#submission.shape


# In[ ]:


#submission.to_csv('submission.csv', index=False)


# In[ ]:




