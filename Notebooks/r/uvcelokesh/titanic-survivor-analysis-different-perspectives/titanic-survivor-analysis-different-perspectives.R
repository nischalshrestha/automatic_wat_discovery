# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#system("ls ../input")



# Any results you write to the current directory are saved as output.



# My first attempt at writing a kernel in Kaggle. I have chosen the notebook approach... why? well

# just as an experiment. Here I will try to analyze the Titanic dataset and improve the prediction

# accuracy and overall score in steps.



library(dplyr)

library(rpart)

library(randomForest)

library(MASS)

library(mice)
# Read the training and test data

train.data <- read.csv("../input/train.csv", stringsAsFactors = F)

test.data <- read.csv("../input/test.csv", stringsAsFactors = F)



dim(train.data)

str(train.data)

dim(test.data)

str(test.data)
# Clean the title

rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')



train.data$Title <- gsub('(.*, )|(\\..*)', '', train.data$Name)

train.data$Title[train.data$Title == 'Mlle']        <- 'Miss' 

train.data$Title[train.data$Title == 'Ms']          <- 'Miss'

train.data$Title[train.data$Title == 'Mme']         <- 'Mrs' 

train.data$Title[train.data$Title %in% rare_title]  <- 'Rare Title'

train.data$FamilySize <- train.data$SibSp + train.data$Parch + 1



test.data$Title <- gsub('(.*, )|(\\..*)', '', test.data$Name)

test.data$Title[test.data$Title == 'Mlle']        <- 'Miss' 

test.data$Title[test.data$Title == 'Ms']          <- 'Miss'

test.data$Title[test.data$Title == 'Mme']         <- 'Mrs' 

test.data$Title[test.data$Title %in% rare_title]  <- 'Rare Title'

test.data$FamilySize <- test.data$SibSp + test.data$Parch + 1
# Merge the datasets

traintest.data <- bind_rows(train.data, test.data)



dim(traintest.data)

str(traintest.data)
# Some bar plot to analyze the data

barplot(table(train.data$Survived, train.data$Sex), col=c("Red", "Green"), legend=c("Dead", "Alive"), main="Survival By Gender")

barplot(table(train.data$Survived, train.data$Age), col=c("Red", "Green"), legend=c("Dead", "Alive"), main="Survival By Age")

barplot(table(train.data$Survived, train.data$Embarked), col=c("Red", "Green"), legend=c("Dead", "Alive"), main="Survival By Embarked")

barplot(table(train.data$Survived, train.data$Pclass), col=c("Red", "Green"), legend=c("Dead", "Alive"), main="Survival By Class")

barplot(table(train.data$Survived, train.data$Title), col=c("Red", "Green"), legend=c("Dead", "Alive"), main="Survival By Title")

barplot(table(train.data$Survived, train.data$FamilySize), col=c("Red", "Green"), legend=c("Dead", "Alive"), main="Survival By Family Size")
# Solution 1

# Just enter random values for the survival

test.data$SurvivedRandom <- rbinom(418, 1, 0.5)

final.data <- data.frame(PassengerId = test.data$PassengerId, Survived = test.data$SurvivedRandom)

write.csv(final.data, file="Output_Random.csv", row.names = FALSE)
# Solution 2

# Only females survive. This is based on the gender based survival graph shown above

test.data$SurvivedGender <- 0

test.data$SurvivedGender[test.data$Sex == "female"] <- 1

test.data$SurvivedGender[test.data$Sex == "female"] <- 1

final.data <- data.frame(PassengerId = test.data$PassengerId, Survived = test.data$SurvivedGender)

write.csv(final.data, file="Output_Gender.csv", row.names = FALSE)
# Update missing values for fare, embarkment and age. Use rpart for age computation

colSums(is.na(test.data))

traintest.data$Fare[1044] <- median(traintest.data$Fare, na.rm = TRUE)

traintest.data$Embarked[c(62, 830)] <- 'C'

predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked, data = traintest.data[!is.na(traintest.data$Age),], method = "anova")

# predicted_age

traintest.data$Age[is.na(traintest.data$Age)] <- predict(predicted_age, traintest.data[is.na(traintest.data$Age),])

colSums(is.na(traintest.data))
# Solution 3... use rpart to compute survival

train.data1 <- traintest.data[1:891,]

test.data1 <- traintest.data[892:1309,]

predicted_survival <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize, data = train.data1, method = "class")

rpart_prediction <- predict(predicted_survival, newdata = test.data1, type = "class")

final.data <- data.frame(PassengerId = test.data1$PassengerId, Survived = rpart_prediction)

write.csv(final.data, file="Output_RPart.csv", row.names = FALSE)
# Solution 4... use randomforest to compute survival

train.data2 <- traintest.data[1:891,]

test.data2 <- traintest.data[892:1309,]

train.data2$Title <- factor(train.data2$Title)

train.data2$Sex <- factor(train.data2$Sex)

train.data2$Survived <- factor(train.data$Survived)

train.data2$Survived <- factor(train.data2$Survived)

train.data2$Embarked <- factor(train.data2$Embarked)

test.data2$Title <- factor(test.data2$Title)

test.data2$Sex <- factor(test.data2$Sex)

test.data2$Survived <- factor(test.data2$Survived)

test.data2$Embarked <- factor(test.data2$Embarked)

predicted_survival <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Title + Embarked + FamilySize, data = train.data2)

rf_prediction <- predict(predicted_survival, test.data2)

final.data <- data.frame(PassengerId = test.data2$PassengerId, Survived = rf_prediction)

write.csv(final.data, file="Output_RF.csv", row.names = FALSE)
# Solution 5... use randomforest to compute survival + some simple feature engineering

traintest.data$Child[traintest.data$Age < 18] <- 'Child'

traintest.data$Child[traintest.data$Age >= 18] <- 'Adult'

traintest.data$Mother <- 'Not Mother'

traintest.data$Mother[traintest.data$Sex == 'female' & traintest.data$Parch > 0 & traintest.data$Age > 18 & traintest.data$Title != 'Miss'] <- 'Mother'

traintest.data$FamilySizeCategory[traintest.data$FamilySize == 1] <- 'Single'

traintest.data$FamilySizeCategory[traintest.data$FamilySize > 1] <- 'Small'

traintest.data$FamilySizeCategory[traintest.data$FamilySize > 4] <- 'Large'



train.data3 <- traintest.data[1:891,]

test.data3 <- traintest.data[892:1309,]

train.data3$Title <- factor(train.data3$Title)

train.data3$Sex <- factor(train.data3$Sex)

train.data3$Survived <- factor(train.data3$Survived)

train.data3$Embarked <- factor(train.data3$Embarked)

train.data3$Child <- factor(train.data3$Child)

train.data3$Mother <- factor(train.data3$Mother)

train.data3$FamilySizeCategory <- factor(train.data3$FamilySizeCategory)

test.data3$Title <- factor(test.data3$Title)

test.data3$Sex <- factor(test.data3$Sex)

test.data3$Survived <- factor(test.data3$Survived)

test.data3$Embarked <- factor(test.data3$Embarked)

test.data3$Child <- factor(test.data3$Child)

test.data3$Mother <- factor(test.data3$Mother)

test.data3$FamilySizeCategory <- factor(test.data3$FamilySizeCategory)

predicted_survival <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Title + Embarked + FamilySizeCategory + Child + Mother, data = train.data3)

rf_prediction <- predict(predicted_survival, test.data3)

final.data3 <- data.frame(PassengerId = test.data2$PassengerId, Survived = rf_prediction)

write.csv(final.data3, file="Output_RF_FE1.csv", row.names = FALSE)
# Solution 6... compute age using MICE & use Surname... still use RandomForest

train.data <- read.csv("../input/train.csv", stringsAsFactors = F)

test.data <- read.csv("../input/test.csv", stringsAsFactors = F)

rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

train.data$Title <- gsub('(.*, )|(\\..*)', '', train.data$Name)

train.data$Title[train.data$Title == 'Mlle']        <- 'Miss' 

train.data$Title[train.data$Title == 'Ms']          <- 'Miss'

train.data$Title[train.data$Title == 'Mme']         <- 'Mrs' 

train.data$Title[train.data$Title %in% rare_title]  <- 'Rare Title'

train.data$FamilySize <- train.data$SibSp + train.data$Parch + 1

test.data$Title <- gsub('(.*, )|(\\..*)', '', test.data$Name)

test.data$Title[test.data$Title == 'Mlle']        <- 'Miss' 

test.data$Title[test.data$Title == 'Ms']          <- 'Miss'

test.data$Title[test.data$Title == 'Mme']         <- 'Mrs' 

test.data$Title[test.data$Title %in% rare_title]  <- 'Rare Title'

test.data$FamilySize <- test.data$SibSp + test.data$Parch + 1

traintest.data <- bind_rows(train.data, test.data)

traintest.data <- bind_rows(train.data, test.data)

traintest.data$Fare[1044] <- median(traintest.data$Fare, na.rm = TRUE)

traintest.data$Embarked[c(62, 830)] <- 'C'

traintest.data$Child[traintest.data$Age < 18] <- 'Child'

traintest.data$Child[traintest.data$Age >= 18] <- 'Adult'

traintest.data$Mother <- 'Not Mother'

traintest.data$Mother[traintest.data$Sex == 'female' & traintest.data$Parch > 0 & traintest.data$Age > 18 & traintest.data$Title != 'Miss'] <- 'Mother'

traintest.data$FamilySizeCategory[traintest.data$FamilySize == 1] <- 'Single'

traintest.data$FamilySizeCategory[traintest.data$FamilySize > 1] <- 'Small'

traintest.data$FamilySizeCategory[traintest.data$FamilySize > 4] <- 'Large'

traintest.data$Surname <- sapply(traintest.data$Name, function(x) strsplit(x, split = '[,.]')[[1]][1])

traintestcopy <- traintest.data

factor_vars <- c('PassengerId','Pclass','Sex','Embarked','Title','Surname','FamilySizeCategory')

full.data <- traintest.data

full.data[factor_vars] <- lapply(full.data[factor_vars], function(x) as.factor(x))

set.seed(123)

mice_mod <- mice(full.data[, !names(full.data) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf')

mice_output <- complete(mice_mod)

full.data$Age <- mice_output$Age



full.data$Child[full.data$Age < 18] <- 'Child'

full.data$Child[full.data$Age >= 18] <- 'Adult'



train.data3 <- full.data[1:891,]

test.data3 <- full.data[892:1309,]

train.data3$Title <- factor(train.data3$Title)

train.data3$Sex <- factor(train.data3$Sex)

train.data3$Survived <- factor(train.data3$Survived)

train.data3$Embarked <- factor(train.data3$Embarked)

train.data3$Child <- factor(train.data3$Child)

train.data3$Mother <- factor(train.data3$Mother)

train.data3$FamilySizeCategory <- factor(train.data3$FamilySizeCategory)

train.data3$Surname <- factor(train.data3$Surname)

test.data3$Title <- factor(test.data3$Title)

test.data3$Sex <- factor(test.data3$Sex)

test.data3$Survived <- factor(test.data3$Survived)

test.data3$Embarked <- factor(test.data3$Embarked)

test.data3$Child <- factor(test.data3$Child)

test.data3$Mother <- factor(test.data3$Mother)

test.data3$FamilySizeCategory <- factor(test.data3$FamilySizeCategory)

test.data3$Surname <- factor(test.data3$Surname)

predicted_survival <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Title + Embarked + FamilySizeCategory + Child + Mother, data = train.data3)

rf_prediction <- predict(predicted_survival, test.data3)

final.data3 <- data.frame(PassengerId = test.data2$PassengerId, Survived = rf_prediction)

write.csv(final.data3, file="Output_RF_MICE.csv", row.names = FALSE)