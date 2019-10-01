titanic.train <- read.csv('../input/train.csv')

titanic.test <- read.csv('../input/test.csv')



# Creating a column flag (0 for train, 1 for test data). We will 

# separate the data into train test on the basis of this variable.

titanic.train[,'flag'] <- 0

titanic.test[,'flag'] <- 1



# In order to bind train and test data, both should have

# same number of columns so adding survived to test data.

titanic.test[, 'Survived'] <- 0

whole.data <- rbind(titanic.train,titanic.test)
head(whole.data)
summary(whole.data)
m <- regexpr("(?<=, ).+?(?=\\. )", whole.data$Name,perl = TRUE)

#get titles

unique(as.factor(regmatches(whole.data$Name, m)))
whole.data[,'Title'] <- regmatches(whole.data$Name, m)

others = c("Don", "Rev", "Dr", "Major", "Lady", "Sir", "Col", "Capt", "the Countess", "Jonkheer", "Dona")

whole.data[whole.data$Title == 'Mlle' | whole.data$Title == 'Ms', 'Title'] <- 'Miss'

whole.data[whole.data$Title == 'Mme', 'Title'] <- 'Mrs'

whole.data[whole.data$Title %in% others, 'Title'] <- 'Others'

whole.data[,'Title'] <- as.factor(whole.data[,'Title'])
library(ggplot2)

g <- ggplot(whole.data, aes(x = factor(Survived))) + geom_bar() + facet_wrap( ~ Title)

g
# Adding number of children for a person and its sibling and spouse. +1 for himself/herself 

whole.data[,'FamilySize'] <- whole.data$SibSp + whole.data$Parch +1



family.survived <- ggplot(whole.data[whole.data$flag==0,], aes(x = FamilySize, fill = factor(Survived))) + geom_bar(position = 'dodge') + scale_x_continuous(breaks = c(1:11))

family.survived
whole.data[whole.data$FamilySize == 1,'FamilyGroup'] <- 'single'

whole.data[whole.data$FamilySize > 1 & whole.data$FamilySize < 5,'FamilyGroup'] <- 'small size'

whole.data[whole.data$FamilySize > 4,'FamilyGroup'] <- 'large'
class.paid.survival <- ggplot(whole.data[whole.data$flag==0 & !is.na(whole.data$Age), ], aes(Fare, fill = factor(Survived))) + geom_histogram(binwidth = 10, position = "dodge") + facet_wrap(~Pclass) + scale_x_continuous(limits = c(1,250), breaks = seq(0, 250, 20))

class.paid.survival
summary(whole.data)
g <- ggplot(whole.data, aes(Age, fill = Survived)) + geom_histogram(binwidth = 5) + facet_wrap(~Title)

g
titles <- unique(whole.data$Title)

for (i in titles){

  whole.data[is.na(whole.data$Age) & whole.data$Title == i,'Age'] <- mean(whole.data[!is.na(whole.data$Age) & whole.data$Title == i,'Age'])

}
summary(whole.data)
for (i in unique(whole.data$Pclass)){(

  whole.data[whole.data[,'Pclass']==i & is.na(whole.data$Fare), 'Fare'] <- median(whole.data[whole.data[,'Pclass']==i & !is.na(whole.data$Fare), 'Fare']))

}
whole.data[whole.data$Embarked == "",]
females.embarked <- ggplot(whole.data[whole.data$Embarked != "" & whole.data$Sex == 'female' & whole.data$Survived == 1 & whole.data$Pclass == 1 & whole.data$Fare > 75 & whole.data$Fare < 85,], aes(Age)) + geom_histogram(binwidth = 1) + facet_grid(~ Embarked)

females.embarked
females.paid <- ggplot(whole.data[whole.data$Pclass==1 & whole.data$Embarked != "",], aes(x = Embarked, y = Fare)) + geom_boxplot() + geom_hline(aes(yintercept = 80))

females.paid
whole.data[whole.data$Embarked == "", 'Embarked'] <- 'C'
age.survived <- ggplot(whole.data[whole.data$flag==0,], aes(Age, fill = factor(Survived))) + geom_histogram(binwidth = 10, position = 'dodge')

age.survived
whole.data[whole.data$Age>=0 & whole.data$Age<21, 'AgeBin'] <- '1'

whole.data[whole.data$Age>=21 & whole.data$Age<28, 'AgeBin'] <- '2'

whole.data[whole.data$Age>=28 & whole.data$Age<39, 'AgeBin'] <- '3'

whole.data[whole.data$Age>=39, 'AgeBin'] <- '4'
whole.data[whole.data$Age<18, 'Child'] <- '0'

whole.data[whole.data$Age>=18, 'Child'] <- '1'
whole.data[whole.data$Fare>=0 & whole.data$Fare<8, 'FareBin'] <- '1'

whole.data[whole.data$Fare>=8 & whole.data$Fare<15, 'FareBin'] <- '2'

whole.data[whole.data$Fare>=15 & whole.data$Fare<31, 'FareBin'] <- '3'

whole.data[whole.data$Fare>=31, 'FareBin'] <- '4'
for (name in c("Survived", "Pclass", "SibSp", "Parch", "FamilyGroup", "FareBin", "AgeBin", "Child")){

  whole.data[, name] <- as.factor(whole.data[, name])

}

head(whole.data)
whole.data <- whole.data[,!(names(whole.data) %in% c("Name", "Cabin", "Ticket", "FamilySize","PassengerId", "SibSp", "Parch", "Age", "Fare"))]

head(whole.data)
titanic.train <- whole.data[whole.data$flag == 0,]

titanic.train <- titanic.train[,!(names(titanic.train) %in% c("flag"))]

titanic.test <- whole.data[whole.data$flag == 1,]

titanic.test <- titanic.test[,!(names(titanic.test) %in% c("flag", "Survived"))]

head(titanic.train)
library(tree)

titanic.tree <- tree(Survived ~ ., titanic.train)

summary(titanic.tree)
plot(titanic.tree)

text(titanic.tree, pretty = TRUE)
library(randomForest)

titanic.tree <- randomForest(Survived ~ ., data = titanic.train, importance = TRUE)

titanic.tree
varImpPlot(titanic.tree)