#!/usr/bin/env python
# coding: utf-8

# #My first notebook on Kaggle :)
# ### In this notebook, we will do the following:
# 1. Preprocess the Titanic dataset
# 2. Build different classifiers without tweaking parameters (i.e. the most naive way)
# 3. Compare the performances of different classifiers
# 4. Pick the best classifier

# In[ ]:


import pandas as pd
import sklearn

# all the classifiers that we will use
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# ### Load the input data as pandas datatable

# In[ ]:


raw_train_df = pd.read_csv("../input/train.csv")

# Let's first split raw_train_df into training_set: validation_set = 8 : 2
# We are going to use this validation set to see how well a classifier performs later
[train_df, valid_df] = sklearn.model_selection.train_test_split(raw_train_df, test_size=0.2)

# Let's look at the data to see how we should preprocess it
train_df.head()


# ### Preprocess data
# 
# #### From observing the above datatable, we notice that
# 1. PassengerId, Name and Ticket columns seem to be useless (by intuition)
# 2. Age and Cabin columns have empty rows
# 3. Some values are numerical while others are categorical (i.e. not a number)
# 
# #### We do the following to preprocess the data
# 1. Drop Name and Ticket columns
# 2. Populate Age and Cabin columns
# 3. Encode categorical values (e.g. map "male" to 0, "female" to 1)

# In[ ]:


def drop_cols(df, cols_to_drop):
    for col in cols_to_drop:
        df = df.drop(col, 1)
    return df
        
def encode_cols(df, cols_to_encode):
    for col in cols_to_encode:
        df[col] = df[col].astype('category').cat.codes
    return df

non_null_cabin_col = train_df['Cabin'][train_df['Cabin'].notnull()]
def get_random_cabin():
    return non_null_cabin_col.sample(n=1).values[0]

def preprocess(train_df):
    train_df = drop_cols(train_df, ['PassengerId', 'Name', 'Ticket'])
    train_df = encode_cols(train_df, ['Cabin', 'Embarked', 'Sex'])

    # Fill Columns
    train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())
    train_df['Cabin'] = train_df['Cabin'].apply(lambda x: get_random_cabin() if pd.isnull(x) else x)
    return train_df

train_df = preprocess(train_df)
train_df.head()


# ### We are now ready to build a classifier. Let's start with a linear SVM without any parameters.

# In[ ]:


# X = all feature columns
# Y = label column ("Survived")
train_df_x = train_df.drop('Survived', 1)
train_df_y = train_df['Survived']

# Build and train the model with the training set
svm = SVC()
svm.fit(train_df_x, train_df_y)
svm


# ### See how the trained SVM classifier performs with a validation set

# In[ ]:


# test_df = pd.read_csv("../input/test.csv")
# We make a validation set by splitting the training set

valid_df = preprocess(valid_df)
valid_df_x = valid_df.drop('Survived', 1)
valid_df_y = valid_df['Survived']

def get_accuracy(trained_classifier, x, y):
    predicted_vals = trained_classifier.predict(x)
    result = (y == predicted_vals).value_counts()
    return float(result[True]) / float(len(predicted_vals))

get_accuracy(svm, valid_df_x, valid_df_y)


# ### Let's build different classifiers and see if we can achieve a better accuracy.

# In[ ]:


classifiers = {
    "Nearest Neighbors": KNeighborsClassifier(3),
    "Most Naive SVM": SVC(), # our initial classifier
    "Linear SVM": SVC(kernel="linear", C=0.025),
    "RBF SVM": SVC(gamma=2, C=1),
    "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "Neural Net": MLPClassifier(alpha=1),
    "AdaBoost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis(),
}

# Reuse train_df_x and train_df_y
trained_classifiers = {}
for key in classifiers.keys():
    classifier = classifiers[key]
    trained_classifiers[key] = classifier.fit(train_df_x, train_df_y)


# In[ ]:


# Compare performances of different classifiers
for classifier_name in trained_classifiers.keys():
    classifier = trained_classifiers[classifier_name]
    print(classifier_name, get_accuracy(classifier, valid_df_x, valid_df_y))


# ## Based on the above results, QDA classifier seems to do well.
# 
# - We want to run k-fold cross validations to see if we're overfitting.
# - We can tweak the parameters to see if we can do better.
# 
# For now, let's just try to submit predicted values using the QDA classifier and see where we rank.

# In[ ]:


# Test set is same as training set, except it doesn't have the labels ('Survived')
# Use our classifier to predict the Survived column
test_df = pd.read_csv("../input/test.csv")
passenger_ids = test_df['PassengerId']
test_df_x = preprocess(test_df)
# test_df_x has a row where Fare is empty. Let's populate it with a mean
# Q. How do I find/populate such a row in the data set less manually?
test_df_x['Fare'] = test_df_x['Fare'].fillna(test_df_x['Fare'].mean())

# We are now ready to make our predictions
classifier = trained_classifiers['QDA']
predicted_values = classifier.predict(test_df_x)


# In[ ]:


# Let's make our predicted_values in a submission format
submission = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission['PassengerId'] = passenger_ids
submission['Survived'] = predicted_values
submission


# ### Let's create a link to download the csv and submit it to Kaggle.

# In[ ]:


from IPython.display import FileLink, FileLinks

submission.to_csv('titanic.csv', index=False)
FileLink('titanic.csv')

