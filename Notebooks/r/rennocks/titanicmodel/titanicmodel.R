# Training Set

df <- read.csv('../input/train.csv',

                       row.names = 1,

                       header = TRUE)



# Kaggle Validation Set

vdf <- read.csv('../input/test.csv',  # Survived column doesn't exist in test

                       header = TRUE)



df$Pclass <- factor(df$Pclass, labels = "class")

vdf$Pclass <- factor(vdf$Pclass, labels = "class")



# important to get 1 and 0 as a classification

df$Survived <- factor(df$Survived, labels = c('No', 'Yes'))



summary(df[, c('Pclass', 'Survived')])
library(caret)



resampledataPart <- createDataPartition(df$Survived, list = FALSE)

trainTable <- df[resampledataPart,]

trainTable
# train Serviced column BASED on the Sex column

model <- train(Survived ~ Sex, 

              data = trainTable,

               method = "glm")

model



val <- df[-resampledataPart,]

val_results <- predict(model, val)



val_results
cm <- confusionMatrix(val_results, val$Survived, positive = 'Yes')

cm$byClass[['F1']]
cm
vdf$Survived <- predict(model, vdf, na.action = na.pass)

vdf$Survived <- ifelse(vdf$Survived == 'Yes', 1, 0)

vdf$Survived
write.csv(vdf[, c('PassengerId', 'Survived')], file = 'submission.csv', row.names = FALSE)

Sys.sleep(1)