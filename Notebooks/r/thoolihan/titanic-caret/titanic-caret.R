col.classes <- c('numeric', #PassengerId

                 'numeric', #Survived

                 'numeric', #Pclass

                 'character', #Name

                 'factor', #Sex

                 'numeric', #Age

                 'integer', #SibSp

                 'integer', #Parch

                 'character', #Ticket

                 'numeric', #Fare

                 'character', #Cabin

                 'factor') #Embarked



# Training Set

df <- read.csv('../input/train.csv',  

                       colClasses = col.classes,

                       row.names = 1,

                       header = TRUE)



# Kaggle Validation Set

vdf <- read.csv('../input/test.csv',  

                       colClasses = col.classes[-2],  # Survived column doesn't exist in test

                       header = TRUE)
df$Pclass <- factor(df$Pclass, labels = "class")

vdf$Pclass <- factor(vdf$Pclass, labels = "class")



df$Survived <- factor(df$Survived, labels = c('No', 'Yes'))



summary(df[, c('Pclass', 'Survived')])
library(caret)



trctl <- trainControl(method = 'cv', repeats = 10, savePredictions = TRUE)



model <- train(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + 

               Pclass:Age:Sex + Pclass:Sex + Sex:SibSp:Parch, 

               data = df,

               method = "glm",

               preProcess = c('knnImpute', 'pca'),

               na.action = na.pass,

               trControl = trctl)



model
cm <- confusionMatrix(model$pred$pred, model$pred$obs, positive = 'Yes')

cm$byClass[['F1']]
vdf$Survived <- predict(model, vdf, na.action = na.pass)

vdf$Survived <- ifelse(vdf$Survived == 'Yes', 1, 0)

write.csv(vdf[, c('PassengerId', 'Survived')], file = 'submission.csv', row.names = FALSE)

Sys.sleep(1)