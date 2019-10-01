#!/usr/bin/env python
# coding: utf-8

# I was trying to understand Receiver Operator Curve (ROC) and Area Under the Curve (AUC). This notebook is the result. 
# There are many fine notebooks around here, with plenty of feature engineering, classifier optimization, etc. I mainly wanted to explore ROC here, so this notebook only includes some very basics for running a classifier. Then playing with ROC and a bit of accuracy.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# In[ ]:


df_train.shape


# In[ ]:


df_test.shape


# In[ ]:


df_train.describe(include="all").round(2)


# Combining training + test data to do all preprocessing at once:

# In[ ]:


df = df_train.append(df_test, ignore_index=True, sort=False)


# In[ ]:


df.describe()


# To see which columns have nulls that we should do something about.

# In[ ]:


df.isnull().sum()


# In[ ]:


df.shape


# Out of 1309 rows/people, 1014 Cabin values are null, which is quite a large percentage:

# In[ ]:


(df['Cabin'].isnull().sum()/df.shape[0])*100 


# Age also has quite a few nulls, but not so bad:

# In[ ]:


(df['Age'].isnull().sum()/df.shape[0])*100 


# Fare and Embarked are 1 and 2 missing values. Will definitely keep those. Age seems relevant, so a look at values and will impute Age with median:

# In[ ]:


df["Age"].mean(skipna=True)


# In[ ]:


df["Age"].median(skipna=True)


# Embarked is just missing 2 values, so use the most common value:

# In[ ]:


df['Embarked'].value_counts()


# I could just pick S from above as most frequent value for missing Embedded values but idxmax() seems like a fancy way to do it

# In[ ]:


df['Embarked'].value_counts().idxmax()


# Impute missing values, drop Cabin column (too many missing)

# In[ ]:


df2 = df.copy()
df2["Age"].fillna(df["Age"].median(skipna=True), inplace=True)
df2["Embarked"].fillna(df['Embarked'].value_counts().idxmax(), inplace=True)
df2.drop('Cabin', axis=1, inplace=True)


# In[ ]:


df2.head()


# In[ ]:


df2.isnull().sum()


# Fare still remains. Who is the one without fare info?

# In[ ]:


df2[df2["Fare"].isnull()==True]


# In[ ]:


df2["Fare"].mean()


# In[ ]:


df2["Fare"].median()


# In[ ]:


df2["Fare"].fillna(df["Fare"].median(skipna=True), inplace=True)


# In[ ]:


df2.isnull().sum()


# To verify Mr. Thomas Storey got his median fare: 

# In[ ]:


df2.iloc[1043]


# In[ ]:


df2[1043:1044]


# Just a quick look now to see if something still needs to be done:

# In[ ]:


df2.head()


# Encode embarked and sex, but if you are male, you are not female, so just need one of the two numbers...

# In[ ]:


df3=pd.get_dummies(df2, columns=["Embarked","Sex"])
#0 or 1 for sex_male covers sex_female so can drop the other
df3.drop('Sex_female', axis=1, inplace=True)
#will ignore name and ticket, even if some parse title etc from name
df3.drop('Name', axis=1, inplace=True)
df3.drop('Ticket', axis=1, inplace=True)


# In[ ]:


df3.head()


# Done preprocessing, so separate the training and test data back again (after merge for preprocessing in beginning)

# In[ ]:


df3_train = df3[:df_train.shape[0]]


# In[ ]:


df3_train.shape


# In[ ]:


df3_test = df3[df_train.shape[0]:]


# In[ ]:


df3_test.shape


# In[ ]:


df3_train.isnull().sum()


# In[ ]:


df3_test.isnull().sum()


# In[ ]:


y = df3_train["Survived"]


# In[ ]:


X = df3_train[["Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked_C", "Embarked_Q", "Embarked_S", "Sex_male"]]


# In[ ]:


X2 = df3_test[["Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked_C", "Embarked_Q", "Embarked_S", "Sex_male"]]


# Using a logistic regression classifier to try out the ROC:

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score 
from sklearn.metrics import confusion_matrix, roc_curve, auc


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[ ]:


predictions = logreg.predict(X_test)
predicted_probabilities = logreg.predict_proba(X_test)


# In[ ]:


predicted_probabilities.shape


# In[ ]:


X_test.shape


# "predicted_probabilities" has two numbers for each row of features: [probability of false, probability of true].
# This just takes the true probability for each row (false probability is 1-true so just need one).

# In[ ]:


pred_probs = predicted_probabilities[:, 1]


# roc_curve() is the sklearn implementation of ROC curve calculation. I find gives is a good start to start looking into ROC. roc_curve() returns a list of false positive rates (FPR) and true positives rates (TPR) for different configurations of the classifier used to plot the ROC.

# In[ ]:


[fpr, tpr, thr] = roc_curve(y_test, pred_probs)


# In[ ]:


plt.figure(figsize=(10, 6), dpi=80)
plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=14)
plt.ylabel('True Positive Rate (TPR)', fontsize=14)
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()


# The following gives accuracy for the default classifier configuration, and the AUC for the sklearn roc_curve() function results. Useful to compare against later.

# In[ ]:


print("accuracy: %2.3f" % accuracy_score(y_test, predictions))
print("AUC: %2.3f" % auc(fpr, tpr))


# The confusion matrix gives me access to number of true positives, false positives, true negatives, and false negatives. Again, useful for looking into ROC and classifier configurations.

# In[ ]:


confusion_matrix(y_test, predictions)


# In[ ]:


accuracy_score(y_test, predictions)


# In[ ]:


y_test.count()


# Just to be sure, I sum up confusion matrix to see the result matches y_test.count()

# In[ ]:


90+10+30+49


# To get passenger indices with predicted survival probability 50% or higher:

# In[ ]:


#ss is my survivor-set
ss1 = np.where(pred_probs >= 0.5)


# Assuming ss1 now contains indices into pred_probs to rows where pred_probs >= 0.5. Since pred_probs indices should match y_test indices, which should match X_text indices, the ss1 indices should work on all of them to find rows where predicted survival probability >= 0.5. 
# 
# Sounds complicated? Just think of ss1 as an array that contains numbers. Each of those numbers in the array is an index into y_test. Think about it from there..

# In[ ]:


ss1


# As shown above, ss1 is a tuple with an array for the indices at index 0. The array from index 0[](http://) is what we really want.

# In[ ]:


len(ss1[0])


# And for indices where prediction is non-survival:

# In[ ]:


ss2 = np.where(pred_probs < 0.5)


# In[ ]:


len(ss2[0])


# In[ ]:


y_test.value_counts()


# It seems to be predicting more non-survivors and less survivors than there really were. 
# 
# How many predicted survivors did actually survive?

# In[ ]:


sum(y_test.values[i] for i in ss1[0])


# And how many predicted fatalities actually did not survive?

# In[ ]:


sum(1 for i in ss2[0] if y_test.values[i] == 0)


# In[ ]:


y_test.shape


# How many did we get right (TP+TN)? What is the accuracy ( (TP+TN)/row_count)?

# In[ ]:


(49+90)/179


# That is a perfect match on accuracy_score calculated with sklearn above, so can just consider I am still going at it right.

# To calculate my own TRP and FRP values using thresholds from 1-99, to re-build the ROC curve, see if this is how ROC is built

# In[ ]:


tpr2 = []
fpr2 = []
temp_data = []
#do the calculations for range of 1-100 percent confidence, increments of 1
for threshold in range(1, 100, 1):
    ss = np.where(pred_probs >= (threshold/100))
    count_tp = 0
    count_fp = 0
    for i in ss[0]:
        if y_test.values[i] == 1:
            count_tp += 1
        else:
            count_fp += 1
    tpr2.append(count_tp/79.0)
    fpr2.append(count_fp/100.0)
    temp_data.append((threshold, count_tp, count_fp))
    #using temp dataframe to make it look nicer, but this print works too
    #print("threshold="+str(threshold)+", count_tp="+str(count_tp)+", count_fp="+str(count_fp))
col_names = ["threshold", "tp", "fp"]
temp_df = pd.DataFrame.from_records(temp_data, columns=col_names)
display(temp_df)


# And to visualize the values for the thresholds I just created: 

# In[ ]:


plt.figure(figsize=(10, 6), dpi=80)
plt.plot(fpr2, tpr2, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr2, tpr2))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC test')
plt.legend(loc="lower right")
plt.show()


# Comparing this figure to the one drawn from sklearn roc_curve(), this is very close. Drawing the two on the same chart should illustrate it better:

# In[ ]:


plt.figure(figsize=(10, 6), dpi=80)
plt.plot(fpr2, tpr2, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr2, tpr2))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC test')
plt.legend(loc="lower right")
plt.show()


# There is just a small difference, with my own version drawing a slightly smoother line. To see why, I look into the data used to draw the curves:

# In[ ]:


#this is the fpr from roc_curve
fpr.shape


# In[ ]:


#and this is the fpr from my own code (1-99 thresholds)
len(fpr2)


# "fpr" has fewer elements, so at least having fewer points in "fpr2" is not the cause. Look into contents:

# In[ ]:


fpr


# In[ ]:


np.array(fpr2)


# The sklearn roc_curve() values seem to pack more information into fewer points (much less duplicates). The "thr" values from roc_curve() are actually the thresholds sklearn used to calculate the ROC curve points. Looking at those:

# In[ ]:


thr


# Using the "thr" threshold values, I can calculate FPR and TPR points as before:

# In[ ]:


tpr3 = []
fpr3 = []
temp_data = []
#do the calculations for sklearn thresholds
for threshold in thr:
    ss = np.where(pred_probs >= threshold)
    count_tp = 0
    count_fp = 0
    for i in ss[0]:
        if y_test.values[i] == 1:
            count_tp += 1
        else:
            count_fp += 1
    tpr3.append(count_tp/79.0)
    fpr3.append(count_fp/100.0)
    #print("threshold="+str(threshold)+", count_tp="+str(count_tp)+", count_fp="+str(count_fp))
    temp_data.append((threshold, count_tp, count_fp))
col_names = ["threshold", "tp", "fp"]
temp_df = pd.DataFrame.from_records(temp_data, columns=col_names)
display(temp_df)


# This is the exact same ROC curve point calculation as I did for the 1-99 thresholds, but using the sklearn thresholds. If the difference in the charts from my code vs the chart from sklearn roc_curve() was due to the thresholds, drawing the above results should look like the sklearn curve:

# In[ ]:


plt.figure(figsize=(10, 6), dpi=80)
plt.plot(fpr3, tpr3, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr3, tpr3))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC test')
plt.legend(loc="lower right")
plt.show()


# Looks like exactly the same, to verify I draw the sklearn roc_curve() based chart on top of this self-calculated one:

# In[ ]:


plt.figure(figsize=(10, 6), dpi=80)
plt.plot(fpr3, tpr3, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr3, tpr3))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.3f)' % auc(fpr3, tpr3))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC test')
plt.legend(loc="lower right")
plt.show()


# It is 100% overlap, so this should verify that this is how the curve is drawn and this is how ROC is calculated and interpreted. The only difference is that sklearn has some fancier way to calculate roc_curve points, or to select the threshold values that have impact. I will leave it at that. 
# 
# Finally a look into how using these different thresholds would impact accuracy with this classifier and these probabilities:

# In[ ]:


tpr3 = []
fpr3 = []
accuracies = []
accs_ths = []
temp_data = []

#do the calculations for sklearn thresholds
for threshold in thr:
    ss = np.where(pred_probs >= threshold)
    count_tp = sum(y_test.values[i] for i in ss[0])
    count_fp = len(ss[0]) - count_tp
    my_tpr = count_tp/79.0
    my_fpr = count_fp/100.0

    ssf = np.where(pred_probs < threshold)
    count_tn = sum(1 for i in ssf[0] if y_test.values[i] == 0)
    
    tpr3.append(my_tpr)
    fpr3.append(my_fpr)
    acc = (count_tp+count_tn)/y_test.count()
    accuracies.append(acc)
    threshold = round(threshold, 5)
    acc = round(acc, 5)
    my_tpr = round(my_tpr, 5)
    accs_ths.append((acc, threshold, count_tp, count_tn, count_tp+count_tn, my_tpr, my_fpr))
    #print("threshold="+str(threshold)+", tp="+str(count_tp)+", fp="+str(count_fp), ", tn="+str(count_tn), ", acc="+str(acc), "tpr="+str(my_tpr)+", fpr="+str(my_fpr))
    temp_data.append((threshold, count_tp, count_fp, count_tn, acc, my_tpr, my_fpr))

col_names = ["threshold", "tp", "fp", "tn", "accucary", "tpr", "fpr"]
temp_df = pd.DataFrame.from_records(temp_data, columns=col_names)
display(temp_df)


# Graphing the accuracy vs FPR and TPR with the same thresholds:

# In[ ]:


plt.figure(figsize=(10, 6), dpi=80)
plt.plot(fpr3, tpr3, color='blue', label='ROC curve (area = %0.3f)' % auc(fpr3, tpr3))
plt.plot(fpr3, accuracies, color='coral', label='Accuracy')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate/Accuracy', fontsize=14)
plt.title('ROC test')
plt.legend(loc="lower right")
plt.show()


# The accuracies at different thresholds, sorted to have the highest accuracy at the top:

# In[ ]:


accs_ths.sort(reverse=True)
accs_ths
#accuracy, threshold, count tp, count tn, count tp+count tn, tpr, fpr

col_names = ["accuracy", "threshold", "tp", "tn", "tn+tp", "tpr", "fpr"]
temp_df = pd.DataFrame.from_records(accs_ths, columns=col_names)
display(temp_df)


# Compare against the "default" classifier threshold of 0.5:

# In[ ]:


ssx = np.where(pred_probs >= 0.5)
count_tp = sum(y_test.values[i] for i in ssx[0])
count_fp = len(ssx[0]) - count_tp
my_tpr = count_tp/87.0
my_fpr = count_fp/175.0

ssxf = np.where(pred_probs < 0.5)
count_tn = sum(1 for i in ssxf[0] if y_test.values[i] == 0)

acc = (count_tp+count_tn)/y_test.count()
acc = round(acc, 5)
my_tpr = round(my_tpr, 5)
my_fpr = round(my_fpr, 5)
print("threshold="+str(0.5)+", tp="+str(count_tp)+", fp="+str(count_fp), ", tn="+str(count_tn), ", acc="+str(acc), "tpr="+str(my_tpr)+", fpr="+str(my_fpr)+" tp+tn="+str(count_tp+count_tn))
    


# Looking at the chart accuracy vs TPR vs FPR, I guess it makes sense that higher accuracy would be concentrated where FPR is lower.  Looking at the sorted accuracy values above and comparing against the scores at the "default" of 0.5, I find a few interesting values at the top of the sorted accuracy/thresholds list. At 0.5 threshold, we get the accuracy of 0.77654 which also matches the sklearn accuracy_score() of 0.777 from higher above. Still, there are a number of thresholds in the sorted accuracy list with a higher accuracy score than that. The two top ones are:
# 

# In[ ]:


temp_df.head(2)


# 
# Which translates to:
# 
# * Threshold of 0.2849 giving an accuracy of 0.7933, classifying 142 instances correctly. 60 true positives, 82 true negatives. 3 more correct predictions than default of 0.5 (142 vs 139).
# * Threshold of 0.52806 giving an accuracy of 0.78212, classifying 140 instances correctly. 48 true positives, 92 true negatives. 1 more correct prediction than default of 0.5 (140 vs 139).
# 
# Compared to the threshold of 0.5, both of these are higher in accuracy/number of correct classifications. Not sure how that would play out with with some other set of unseen data vs overfitting this test data, but anyway..
# 
# * The highest accuracy score is using a treshold (0.2849) that is quite a bit lower than 0.5, and puts much more focus on identifying true positives. 
# * The second highest score uses a threshold that is slightly above the 0.5 value (0.52806) and puts much more focus on identifying true negatives over true positives. 
# 
# Perhaps this highlights one of the possible uses for such thresholds, in selecting which focus (true positives vs true negatives) is more important in the use case at hand?
# 

# Conclusions? This exercise certainly helped me understand ROC/AUC much better..  Hope others find it interesting/useful :)

# In[ ]:




