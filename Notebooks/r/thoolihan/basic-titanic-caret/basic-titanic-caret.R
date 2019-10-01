# Training Set

df <- read.csv('../input/train.csv',

                       row.names = 1,

                       header = TRUE)



# Kaggle Validation Set

vdf <- read.csv('../input/test.csv',  # Survived column doesn't exist in test

                       header = TRUE)
df$Pclass <- factor(df$Pclass, labels = "class")

vdf$Pclass <- factor(vdf$Pclass, labels = "class")



df$Survived <- factor(df$Survived, labels = c('No', 'Yes'))



summary(df[, c('Pclass', 'Survived')])
library(caret)



idx <- createDataPartition(df$Survived, list = FALSE)



train <- df[idx,]



model <- train(Survived ~ Sex, 

              data = train,

               method = "glm")



val <- df[-idx,]

val_results <- predict(model, val)
cm <- confusionMatrix(val_results, val$Survived, positive = 'Yes')

cm$byClass[['F1']]
cm
vdf$Survived <- predict(model, vdf, na.action = na.pass)

vdf$Survived <- ifelse(vdf$Survived == 'Yes', 1, 0)

write.csv(vdf[, c('PassengerId', 'Survived')], file = 'submission.csv', row.names = FALSE)

Sys.sleep(1)