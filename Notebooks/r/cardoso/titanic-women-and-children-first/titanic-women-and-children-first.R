library('rpart')

library('randomForest')

library('party')

library('ggplot2')

library('ggthemes') 

library('scales') 

library('mice')

library('randomForest') 

library('magrittr')
train <- read.csv("../input/train.csv")

test <- read.csv("../input/test.csv")
test$Survived <- NA

fullComb <- rbind(train, test)
str(fullComb)
fullComb$Name <- as.character(fullComb$Name)
fullComb$Title <- sapply(fullComb$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})

fullComb$Title <- sub(' ', '', fullComb$Title)
fullComb$Title[fullComb$Title %in% c('Mme', 'Mlle')] <- 'Mlle'

fullComb$Title[fullComb$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'

fullComb$Title[fullComb$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
fullComb$Title <- factor(fullComb$Title)
fullComb$FamilySize <- fullComb$SibSp + fullComb$Parch + 1



ggplot(fullComb[1:891,], aes(x = FamilySize, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') +

  scale_x_continuous(breaks=c(1:11)) +

  labs(x = 'Family Size') +

  theme_few()
fullComb$FsizeD[fullComb$FamilySize == 1] <- 'singleton'

fullComb$FsizeD[fullComb$FamilySize < 5 & fullComb$Fsize > 1] <- 'small'

fullComb$FsizeD[fullComb$FamilySize > 4] <- 'large'
fullComb$Surname <- sapply(fullComb$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})

fullComb$FamilyID <- paste(as.character(fullComb$FamilySize), fullComb$Surname, sep="")

fullComb$FamilyID[fullComb$FamilySize <= 2] <- 'Small'
famIDs <- data.frame(table(fullComb$FamilyID))

famIDs <- famIDs[famIDs$Freq <= 2,]

fullComb$FamilyID[fullComb$FamilyID %in% famIDs$Var1] <- 'Small'
fullComb$FamilyID <- factor(fullComb$FamilyID)
summary(fullComb$Age)

Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize, 

                data=fullComb[!is.na(fullComb$Age),], method="anova")

fullComb$Age[is.na(fullComb$Age)] <- predict(Agefit, fullComb[is.na(fullComb$Age),])
summary(fullComb)
summary(fullComb$Embarked)

which(fullComb$Embarked == '')
fullComb$Embarked[c(62,830)] = "S"
summary(fullComb$Fare)

which(is.na(fullComb$Fare))

fullComb$Fare[1044] <- median(fullComb$Fare, na.rm=TRUE)



ggplot(fullComb[fullComb$Pclass == '3' & fullComb$Embarked == 'S', ], 

       aes(x = Fare)) +

  geom_density(fill = '#99d6ff', alpha=0.4) + 

  geom_vline(aes(xintercept=median(Fare, na.rm=T)),

             colour='red', linetype='dashed', lwd=1) +

  scale_x_continuous(labels=dollar_format()) +

  theme_few()
fullComb$FamilyID2 <- fullComb$FamilyID



fullComb$FamilyID2 <- as.character(fullComb$FamilyID2)

fullComb$FamilyID2[fullComb$FamilySize <= 3] <- 'Small'



fullComb$FamilyID2 <- factor(fullComb$FamilyID2)



ggplot(fullComb[1:891,], aes(Age, fill = factor(Survived))) + 

  geom_histogram() + 

  facet_grid(.~Sex) + 

  theme_few()
fullComb$Child[fullComb$Age < 16] <- 'Child'

fullComb$Child[fullComb$Age >= 16] <- 'Adult'
table(fullComb$Child, fullComb$Survived)
fullComb$Child  <- factor(fullComb$Child)

md.pattern(fullComb)
train <- fullComb[1:891,]

test <- fullComb[892:1309,]
set.seed(2501)

fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2,

                    data=train, importance=TRUE, ntree=2000)
varImpPlot(fit)
Prediction <- predict(fit, test)

submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)

write.csv(submit, file = "one.csv", row.names = FALSE)
set.seed(2501)

fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,

               data = train, controls=cforest_unbiased(ntree=2000, mtry=3)) 

Prediction <- predict(fit, test, OOB=TRUE, type = "response")

submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "two.csv", row.names = FALSE)