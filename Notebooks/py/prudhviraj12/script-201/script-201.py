import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
#print(train.head())

print("\n\nSummary statistics of training data")
#print(train.describe())

#Any files you save will be available in the output tab below
#train.to_csv('copy_of_the_training_data.csv', index=False)
# # Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
# PassengerId =np.array(test["PassengerId"]).astype(int)
# my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
# print(my_solution)
# 
# # Check that your data frame has 418 entries
# print(my_solution.shape)
# 
# # Write your solution to a csv file with the name my_solution.csv
# my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])




train.head(5)
train.Survived.value_counts()
train.Pclass.value_counts()
train.Sex.value_counts()
train['Sex'].loc[train['Sex'] == 'male'] = 0
train['Sex'].loc[train['Sex'] == 'female'] = 1
train.columns
import numpy as np
age_for_na_values = int(np.mean(train.Age))
train['Age'] = train['Age'].fillna(age_for_na_values)
train.SibSp.value_counts()
train.columns
train.Parch.value_counts()
train.Embarked.value_counts()

train["Embarked"].loc[train["Embarked"] == "S"] = 0
train["Embarked"].loc[train["Embarked"] == "C"] = 1
train["Embarked"].loc[train["Embarked"] == "Q"] = 2

train.Embarked = train["Embarked"].fillna(0)

X_train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']].values
y_train = train['Survived'].values


test.columns
test.Age = test.Age.fillna(np.mean(train.Age))
test["Embarked"].loc[test["Embarked"] == "S"] = 0
test["Embarked"].loc[test["Embarked"] == "C"] = 1
test["Embarked"].loc[test["Embarked"] == "Q"] = 2
test['Sex'].loc[test['Sex'] == 'male'] = 0
test['Sex'].loc[test['Sex'] == 'female'] = 1

X_test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']].values

#train.columns
train_copy = train.copy()
Fare_high_female = []
Fare_high_female = (train_copy['Fare'] > 31) & (train_copy['Sex'] == 1)
#[0]
train_copy['High_Fare_Female'] = Fare_high_female
columns = list(train_copy.columns)
first_class_female = []
first_class_female = (train_copy['Pclass']) == 1 & (train_copy['Sex'] == 1)
train_copy['first_class_female'] = first_class_female
age_division = []
age_division = ((train_copy['Age'] < 18) | (train_copy['Age'] > 40))
train_copy['age_division'] = age_division
train_copy.columns
X_train_pesci = train_copy[['High_Fare_Female', 'first_class_female', 'age_division', 'Fare', 'Sex']].values


test_copy = test.copy()
Fare_high_female = []
Fare_high_female = (test_copy['Fare'] > 31) & (test_copy['Sex'] == 1)
test_copy['High_Fare_Female'] = Fare_high_female

first_class_female = []
first_class_female = (test_copy['Pclass']) == 1 & (test_copy['Sex'] == 1)
test_copy['first_class_female'] = first_class_female

age_division = []
age_division = ((test_copy['Age'] < 18) | (test_copy['Age'] > 40))
test_copy['age_division'] = age_division

X_train_pesci = train_copy[['High_Fare_Female', 'first_class_female', 'age_division', 'Fare', 'Sex']].values
X_test_pesci = test_copy[['High_Fare_Female', 'first_class_female', 'age_division', 'Fare', 'Sex']].values


train_copy.columns
train_copy.head()

for n in range(len(train_copy)):
    if (train_copy['High_Fare_Female'][n] == False):
        train_copy['High_Fare_Female'][n] = 0
    else:
        train_copy['High_Fare_Female'][n] = 1

for n in range(len(train_copy)):
    if (train_copy['first_class_female'][n] == False):
        train_copy['first_class_female'][n] = 0
    else:
        train_copy['first_class_female'][n] = 1

for n in range(len(train_copy)):
    if (train_copy['age_division'][n] == False):
        train_copy['age_division'][n] = 0
    else:
        train_copy['age_division'][n] = 1
    
print(123)

print(train_copy.head())

#from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_pesci, y_train)

y_pred = lr.predict(X_test_pesci)
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(y_pred, PassengerId, columns = ["Survived"])
print(my_solution)
# 
#Check that your data frame has 418 entries
print(my_solution.shape)
# 
#Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_scaled_lr.csv", index_label = ["PassengerId"])




