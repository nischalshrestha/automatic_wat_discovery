library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(tidyverse)

library(rpart)

library(rpart.plot)

library(randomForest)

library(party)

library(mice)

library(caret)



system("ls ../input")



train = read.csv("../input/train.csv")

test = read.csv("../input/test.csv")
test$Survived <- NA

combi <- rbind(train, test)

combi$Name <- as.character(combi$Name)



# summary(combi)
# Use rpart for age prediction

# Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,

#                   data=combi[!is.na(combi$Age),], 

#                   method="anova")

# 

# combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])

# combi$Age <- as.numeric(combi$Age)
set.seed(226)

mice_mod <- mice(combi[, !names(combi) %in% c('PassengerId','Name','Ticket','Cabin','FamilySize','Surname','Survived', 'Fare', 'Child')], method='rf') 

mice_output <- complete(mice_mod)



# plot to see difference

plot(combi$Age)

plot(mice_output$Age)



combi$Age <- mice_output$Age

summary(combi$Embarked)

which(is.na(combi$Embarked))



# substitute with mode, "S"

combi$Embarked[c(62,830)] = "S"

combi$Embarked <- factor(combi$Embarked)



summary(combi$Fare)

#substitute with median

combi$Fare[1044] <- median(combi$Fare, na.rm=TRUE)

combi$Fare <- as.numeric(combi$Fare)
combi$FamilySize <- combi$SibSp + combi$Parch + 1

combi$FamilySize <- as.numeric(combi$FamilySize)
combi$Title <- gsub('(.*, )|(\\..*)', '', combi$Name)



#remove very rare titles

combi$Title[combi$Title %in% c('Mme', 'Mlle')] <- 'Mlle'

combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'

combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'



# Alternative: group titles by class

combi$TitleCat <- combi$Title

combi$TitleCat[combi$Title %in% c('Capt', 'Col', 'Major', 'Rev', 'Dr')] <- 'Officer'

combi$TitleCat[combi$Title %in% c('Mrs', 'Ms', 'Mme')] <- 'Mrs'

combi$TitleCat[combi$Title %in% c('Mlle', 'Miss')] <- 'Miss'

combi$TitleCat[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Don', 'Sir', 'Jonkheer')] <- 'Royal'



combi$Title <- factor(combi$Title)

combi$TitleCat <- factor(combi$TitleCat)



cat('Original titles:')

table(combi$Title)

cat('By category:')

table(combi$TitleCat)
combi$Child[combi$Age < 18] <- '1'

combi$Child[combi$Age >= 18] <- '0'

combi$Child <- factor(combi$Child)



combi$Mother <- '0'

combi$Mother[combi$Sex == 'female' & combi$Parch > 0 & combi$Age > 18 & combi$Title != 'Miss'] <- '1'

combi$Mother <- factor(combi$Mother)
# combi$Deck<-factor(sapply(combi$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

# running mice code here again

# plot(combi$Deck)

# plot(mice_output$Deck)
which(is.na(combi$Embarked))

which(is.na(combi$Fare))

which(is.na(combi$Sex))

which(is.na(combi$SibSp))

which(is.na(combi$FamilySize))

which(is.na(combi$Title))

which(is.na(combi$TitleCat))
# split back into the train & test

train <- combi[1:891,]

test <- combi[892:1309,]



# Regular decision tree

fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize,

             data=train,

             method="class")



fit$confusion

# Random forest

fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +

                       Embarked + Title + FamilySize,

                     data=train,

                     importance=TRUE,

                     ntree=2000)

fit$confusion
# conditional inference tree

 fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +

                  Embarked + Title + FamilySize,

                data = train,

                controls=cforest_unbiased(ntree=2000, mtry=3))

set.seed(126)

fit <- train(as.factor(Survived) ~ Pclass + Sex + Age + Child + SibSp + Fare +

                Embarked + Title + FamilySize, 

  data = train, 

  method = "cforest", 

  tuneGrid = data.frame(.mtry = 7),

  trControl = trainControl(method = "oob"))

fit

plot(varImp(fit))
Prediction <- predict(fit, test, OOB=TRUE, type = "raw")

Prediction



submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)



write.csv(submit, file = "inferenceTree", row.names = FALSE)