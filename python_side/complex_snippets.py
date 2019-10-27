# TODO Normalize and add to rsnips
# subsetting on a particular column with a filter
df['Age'][df['Survived'] == 1]
df['Isalone'].loc[df['FamilyMembers'] > 1]
# subsetting on main dataframe then referencing a column(s)
df[vd['PassengerId'] == 1284]['Survived']
df[DataSetTrain['Survived'] == 0][['Name', 'Sex', 'Age', 'Pclass', 'Cabin']]
# subsetting using which with chained conditions where rhs is either numeric or string
df[(df.Pclass == 3) & (df.Embarked == 'S') & (df.SibSp == 0) & (df.Parch == 0) & (df.Sex == 'male') & (df.Age > 50.0)]
# subsetting using chained conditions using mix of boolean operators?

# subsetting with a mix of a functions inside [] (notnull, isnull, isin)
df['Cabin'][df['Cabin'].notnull()].head()
df.loc[(df.Fare.isnull()), :]
x_test.loc[x_test['Fare'].isnull()]['Pclass']
df[fulldata.Fare.isnull()][['Fare', 'Pclass', 'Embarked']]
df[df.Fare.isnull() == True]
df[(df['NoPerTicket'] == 1) & (df['Age'].isnull() == False) & ((df['Parch'] > 0) | (df['SibSp'] > 0))]
df[df.Title.isin(['Rev', 'Dr', 'Col', 'Capt', 'Major']) & (df.Sex != 'male')]
df[df.Name.isin(['Connolly, Miss. Kate', 'Kelly, Mr. James'])].sort_values(by='Name')

# function call on subsetting operation(s)
df[numeric_variables].head(3)
df[(df['Embarked'] == 'C') & (df.Fare > 200)].head()
df[(df.Sex == 'female') & (df.Pclass == 3) & (df.Survived == 1)].head(20)
df[['Age', 'Age_0', 'Age_1', 'Age_2', 'Age_3', 'Age_4', 'Age_5', 'Age_6']].head(10)
df[(df['Age'] > 8.0) & (df['Age'] <= 10.0)].sort_values('Age')
df.loc[df['SharedTicket'] == 1, 'Ticket'].sort_values()
df[df.Age.notnull()].query("Title =='MASTER' and Age >= 10")

# chained function calls
df.sort_values(by='Name').head(10)
df.sort_values(by='SibSp').describe(percentiles=[0.68, 0.69])
df.drop(categorical_columns, axis=1).describe(include=[np.number])
df.query('CabinCount > 1').head()

# computation using dimensions of two dataframes
df[(df.Fare == 0) & (df.Pclass == 2)].shape[0] / (1.0 * df[df.Fare == 0].shape[0])

# inverse boolean vector using ~ (contrast to R, ~ equivalent to ! is higher in precedence so parens are needed)
# for e.g
"""
>>> df
   a  b
0  1  4
1  2  5
2  3  6
>>> ~df.a > 1
0    False
1    False
2    False
Name: a, dtype: bool
>>> ~(df.a > 1)
0     True
1    False
2    False
Name: a, dtype: bool
"""
df[~((df.Embarked == 'S') | (df.Embarked == 'C') | (df.Embarked == 'Q'))]


# Stats
# summary on a chained subsetting operation
df[df['Pclass'] == 3]['Fare'].describe()
train.loc[train['Pclass'] == 3]['Deck'].describe()
