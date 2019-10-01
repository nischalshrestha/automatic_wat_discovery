#!/usr/bin/env python
# coding: utf-8

# # Impact of Features on simple Neural Networks
# 
# 
# ## Introduction
# 
# After a lecture about ML I found out about Kaggle and I thought it might be a motivating way to practice and apply what I've learned.
# 
# To get a feeling about how the __quantity__ and __quality__ of features influences the performance of a simple neural network I decided to try a few constellations and measure the output. I hope sharing my results might help another beginner. However, I'm also new to ML, so feedback is very welcome.
# 
# At first I'm going to introduce the basic program and afterwards I'm going to augment it step by step. 

# ## First Approach
# 
# For this project I used **Keras** to simplify the construction of the NN. To load the data I used pandas:

# In[ ]:


# Import Dataset
data = pd.read_csv('train.csv')
df = pd.DataFrame(data)

# Replace Sex with 1(male) and 0(female)
df.replace({'male': 1, 'female': 0}, inplace=True)


# I replaced missing features (e.g. Age) with -1 and trained a three layer, fully connected network with 1500 nodes on each layer an dropout. Afterwards, I tried dropping some values randomly.

# In[ ]:


# Shuffle Data 
df = df.sample(frac=1.0)

# Delete Cabin (Data to sparse) and ID (pure Random)
# print(df['Survived'].isnull().sum())  # Age:177, Cabin: 687, Embarked:2
df.drop(['PassengerId', 'Cabin', 'Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

# Create train and test
train_data = df.values[:800]
test_data = df.values[800:]

x_train = train_data[:, 1:]
y_train = np_utils.to_categorical(train_data[:, 0])

x_test = test_data[:, 1:]
y_test = np_utils.to_categorical(test_data[:, 0])

# Setup the Network
model = Sequential()
model.add(Dense(1500,
                activation='relu',
                input_shape=(x_train.shape[1],),
                kernel_regularizer=regularizers.l2(0.1)
                ))
model.add(Dropout(0.5))
model.add(Dense(2000, activation='relu',
                kernel_regularizer=regularizers.l2(0.1)
                ))
model.add(Dropout(0.5))
model.add(Dense(1500, activation='relu'))
model.add(Dense(2, activation='softmax',
                kernel_regularizer=regularizers.l2(0.1)
                ))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

tb = TensorBoard(log_dir='logs/{}'.format(time()))

# Train
model.fit(x=x_train, y=y_train, batch_size=200, verbose=2, epochs=25, callbacks=[tb])

# Eval
score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: {}".format(score[1]))


# Some features reduced the accuracy slightly when added or dropped, but in general the accuracy was not exceeding 61%.

# ## Extracting New Features
# After the first approach was not very successful I decided to engineer some new features, inspired by [Megan Risdals script](https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic). 
# 
# The first step was counting how often a certain title occurs:

# In[ ]:


# Extract title from the name

def getTitle(name):
    '''
    :param name: Name of the format Firstname, Title Surename
    :return: Title
    '''

    m = re.search('(?<=,\s)\w+', name)
    return (m.group(0))


# Extract titles and count
title_set = {}
for name in df['Name']:
    title = getTitle(name)
    if title in title_set:
        title_set[title] = title_set[title] + 1
    else:
        title_set[title] = 1
print(title_set) 
# Output:
# {'Mr': 517, 'Ms': 1, 'Don': 1, 'the': 1, 
# 'Mlle': 2, 'Jonkheer': 1, 'Rev': 6, 
# 'Dr': 7, 'Miss': 182, 'Major': 2, 
# 'Sir': 1, 'Lady': 1, 'Mme': 1, 'Mrs': 125, 
# 'Master': 40, 'Col': 2, 'Capt': 1}


# To simplify the classes I merged *Col, Capt* and *Major* to the title-group _Military_, *Mme, Mlle, Ms* to _UnmarriedWoman_, and eventually *Lady, Sir* and *the* to _nobility_. Other titles are considered to be _miscellaneous_. 
# 
# These classes were used to add a new (numerical) attribute to the data:

# In[ ]:


def getTitleNum(name):
    '''
    Assign a numeral according to the title
    :param name: 
    :return: numeral according to title
    '''

    title = getTitle(str(name).upper())

    title_dict = {}
    title_dict["MR"] = 0
    title_dict["MRS"] = 1
    title_dict["COL"] = 2
    title_dict["CAPT"] = 2
    title_dict["MAJOR"] = 2
    title_dict["MME"] = 3
    title_dict["MLLE"] = 3
    title_dict["MS"] = 3
    title_dict["MISS"] = 3
    title_dict["LADY"] = 4
    title_dict["SIR"] = 4
    title_dict["THE"] = 4
    title_dict["MASTER"] = 5
    title_dict["REV"] = 6
    title_dict["DR"] = 7

    if title in title_dict:
        return title_dict[title]
    else:
        return -1


df['Title'] = df.apply(lambda row: getTitleNum(row['Name']), axis=1)


# The old network increased its accuracy on the Test-Data by 10% to 71%. So apparently the title is a good predictor. Even after removing sex and age the accuracy didn't differ significantly. This was strange since Cameron told us that it's women and children first. 
# 
# Therefore, I discretized the age into *toddler (<3)*, *child (<12)*, *teenager (<17)*, *adult (<50)* and *senior (>50)*. With this data I trained another, similar NN to predict the age of the passengers with missing data.

# In[ ]:


# Discretize the Age
def discretAge(age):
    if age < 3:
        return 0
    if age < 12:
        return 1
    if age < 17:
        return 2
    if age < 50:
        return 3
    if age >= 50:
        return 4
    # Keep the missing values
    return age


df['DisAge'] = df.apply(lambda row: discretAge(row['Age']), axis=1)

# Replace Sex with 1(male) and 0(female)
df.replace({'male': 1, 'female': 0}, inplace=True)

# Shuffle Data and extract rows with missing age
df = df.sample(frac=1.0)
age_missing = df[df['Age'].isnull()]
age_complete = df[df['Age'].notnull()]


# Create train and test
ages = age_complete['DisAge'].values

x_train = age_complete[['Title','Pclass', 'Sex']].values[:650]
y_train = np_utils.to_categorical(ages[:650])

x_test = age_complete[['Title','Pclass', 'Sex']].values[650:]
y_test = np_utils.to_categorical(ages[650:])

# Setup the Network
model = Sequential()
model.add(Dense(800,
                activation='relu',
                input_shape=(x_train.shape[1],),
                kernel_regularizer=regularizers.l2(0.1)
                ))

model.add(Dropout(0.5))
model.add(Dense(800, activation='relu'))
model.add(Dense(5, activation='softmax',
                kernel_regularizer=regularizers.l2(0.1)
                ))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

tb = TensorBoard(log_dir='logs/{}'.format(time()))

# Train
model.fit(x=x_train, y=y_train, batch_size=200, verbose=2, epochs=25, callbacks=[tb])

# Eval
score = model.evaluate(x_test, y_test, verbose=0) # ~75%


# In[ ]:


# Apply the new model to predict the missing values

def predictAge(row):
    '''
    Use the trained network to predict the discrete Age
    :param row:
    :return:
    '''

    if math.isnan(float(row['DisAge'])):
        v = np.array(row[['Title', 'Pclass', 'Sex']].values)
        pred = model_age_prediction.predict(v.reshape((1,3)))
        return np.argmax(pred)
    return row['DisAge']

df['DisAge'] = df.apply(lambda row: predictAge(row), axis=1)


# The same network increased its accuracy by another 15% to 87% accuracy. By adding some other values (e. g. Fare) the accuracy fell again, sometimes dramatically. 

# ## Conclusion
# 
# This example shows, that the quality of features is very important. High quantity, however, could do more harm than good. Maybe that's just an artifact from this dataset or maybe I did something terribly wrong. Anyway, feedback is very welcome. =)  
# 
