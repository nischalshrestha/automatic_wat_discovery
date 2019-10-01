# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(keras)
raw_train <- read.csv(file="../input/train.csv", header=TRUE, sep=",", na.strings=c("")) #891 reg

raw_train <- raw_train[!is.na(raw_train$Age), ] # 714 reg

raw_train <- raw_train[!is.na(raw_train$Embarked), ] # 712 reg
sapply(raw_train,function(x) sum(is.na(x)))
# Encodes categorical variables to numeric

raw_train$NumSex <- as.numeric(factor(raw_train$Sex,labels=c(1,2)))

raw_train$NumEmbarked <- as.numeric(factor(raw_train$Embarked,labels=c(1,2,3)))
train_nn <- raw_train[ , c("Pclass", "NumSex", "Age", "SibSp", "Parch", "NumEmbarked", "Fare")]

label_nn <- raw_train$Survived
train_nn$Pclass <-  (train_nn$Pclass - min(train_nn$Pclass)) /

                    (max(train_nn$Pclass) - min(train_nn$Pclass))

train_nn$NumSex <-  (train_nn$NumSex - min(train_nn$NumSex)) /

                    (max(train_nn$NumSex) - min(train_nn$NumSex))

train_nn$Age <-  (train_nn$Age - min(train_nn$Age)) /

                    (max(train_nn$Age) - min(train_nn$Age))

train_nn$SibSp <-  (train_nn$SibSp - min(train_nn$SibSp)) /

                    (max(train_nn$SibSp) - min(train_nn$SibSp))

train_nn$Parch <-  (train_nn$Parch - min(train_nn$Parch)) /

                    (max(train_nn$Parch) - min(train_nn$Parch))

train_nn$NumEmbarked <-  (train_nn$NumEmbarked - min(train_nn$NumEmbarked)) /

                    (max(train_nn$NumEmbarked) - min(train_nn$NumEmbarked))

train_nn$Fare <-  (train_nn$Fare - min(train_nn$Fare)) /

                    (max(train_nn$Fare) - min(train_nn$Fare))
colnames(train_nn)=NULL

for (i in 1:7){train_nn[,i]=as.numeric(train_nn[,i])}

train_nn=as.matrix(train_nn)

label_nn=as.matrix(label_nn)
set.seed(0)

model <- keras_model_sequential() 



model %>% 

  layer_dense(units = 256, activation = 'relu', input_shape = dim(train_nn)[2]) %>% 

  layer_dropout(rate = 0.5) %>% 

  layer_dense(units = 256, activation = 'relu') %>%

  layer_dropout(rate = 0.5) %>%

  layer_dense(units = 1, activation = 'sigmoid')



model %>% compile(

  loss = 'binary_crossentropy',

  optimizer = optimizer_rmsprop(),

  metrics = c('accuracy')

)



history <- model %>% fit(

  train_nn,label_nn,

  epochs = 70, batch_size = 128

)
plot(history)


eval <- model %>% evaluate(train_nn, label_nn)

eval
raw_test <- read.csv(file="../input/test.csv", header=TRUE, sep=",", na.strings=c(""))
sapply(raw_test,function(x) sum(is.na(x)))
raw_test$Age[is.na(raw_test$Age)] <- mean(raw_test$Age,na.rm=T)

raw_test$Fare[is.na(raw_test$Fare)] <- mean(raw_test$Fare,na.rm=T)
raw_test$NumSex <- as.numeric(factor(raw_test$Sex,labels=c(1,2)))

raw_test$NumEmbarked <- as.numeric(factor(raw_test$Embarked,labels=c(1,2,3)))

test_nn <- raw_test[ , c("Pclass", "NumSex", "Age", "SibSp", "Parch", "NumEmbarked", "Fare")]

test_nn$Pclass <-  (test_nn$Pclass - min(test_nn$Pclass)) /

                    (max(test_nn$Pclass) - min(test_nn$Pclass))

test_nn$NumSex <-  (test_nn$NumSex - min(test_nn$NumSex)) /

                    (max(test_nn$NumSex) - min(test_nn$NumSex))

test_nn$Age <-  (test_nn$Age - min(test_nn$Age)) /

                    (max(test_nn$Age) - min(test_nn$Age))

test_nn$SibSp <-  (test_nn$SibSp - min(test_nn$SibSp)) /

                    (max(test_nn$SibSp) - min(test_nn$SibSp))

test_nn$Parch <-  (test_nn$Parch - min(test_nn$Parch)) /

                    (max(test_nn$Parch) - min(test_nn$Parch))

test_nn$NumEmbarked <-  (test_nn$NumEmbarked - min(test_nn$NumEmbarked)) /

                    (max(test_nn$NumEmbarked) - min(test_nn$NumEmbarked))

test_nn$Fare <-  (test_nn$Fare - min(test_nn$Fare)) /

                    (max(test_nn$Fare) - min(test_nn$Fare))

colnames(test_nn)=NULL

for (i in 1:7){test_nn[,i]=as.numeric(test_nn[,i])}

test_nn=as.matrix(test_nn)
result <- model %>% predict(test_nn)
result_bin <- ifelse(result < 0.5, 0, 1)
out=cbind(raw_test$PassengerId,result_bin)

colnames(out)=c("PassengerId", "Survived")
write.table(out, file = "Resultado_Titanic_NN_20180104.csv", row.names=FALSE, sep=",")