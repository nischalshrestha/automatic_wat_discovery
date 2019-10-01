# This notebook is very much inspired from : https://machinelearningmastery.com/machine-learning-ensembles-with-r/

library(mlbench)

library(caret)

library(caretEnsemble)

library(data.table)

library(dplyr)

system("ls ../input")



# Any results you write to the current directory are saved as output.
train <- fread("../input/train.csv", stringsAsFactors = TRUE)

test <- fread("../input/test.csv", stringsAsFactors = TRUE)

ids <- test$PassengerId

all <- rbind(train%>%dplyr::select(-Survived), test)
Survived = train$Survived

PassengerId = test$PassengerId

all$Name = as.character(all$Name)

all$Title <- sapply(all$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})

all$Title <- sub(' ', '', all$Title)

all[!all$Title %in% c("Mr", "Miss", "Mrs", "Master")]$Title = "Rare"

all$Title <- as.factor(all$Title)
age_group <- all %>%

  filter(!is.na(Age)) %>%

  group_by(Title, Pclass, Sex) %>%

  dplyr::summarise(avgAge = mean(Age), count = n())
all <- all %>%

  left_join(age_group) %>%

  mutate(Age = ifelse(is.na(Age), avgAge, Age)) %>%

  dplyr::select(-avgAge, -count)
all <- all%>%dplyr::select(-PassengerId, -Name, -Ticket, -Cabin)

dmy <- dummyVars(" ~ .", data = all, fullRank = T)

all <- data.frame(predict(dmy, newdata = all))



train <- cbind(all[1:nrow(train),], Survived)

test <- all[nrow(train)+1 : nrow(test), ]

train$Survived <- as.factor(ifelse(train$Survived == 0, "N", "Y") )
# Example of Stacking algorithms

# create submodels

control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)

algorithmList <- c('lda', 'rpart', 'glm', 'knn', 'svmRadial')

set.seed(3)

models <- caretList(Survived~., data=train, trControl=control, methodList=algorithmList)

results <- resamples(models)

summary(results)
dotplot(results)
modelCor(results)
# stack using glm

stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)

set.seed(3)

stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)

print(stack.glm)
set.seed(3)

stack.rf <- caretStack(models, method="rf", metric="Accuracy", trControl=stackControl)

print(stack.rf)
pred_test <- ifelse(predict(stack.rf, newdata =test) == "Y",1,0)

Submission <- as.data.frame(cbind(ids, pred_test))

names(Submission) <- c("PassengerId", "Survived")

write.csv(Submission, "stacked_submission.csv", row.names = FALSE)