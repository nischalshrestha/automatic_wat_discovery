# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 

library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(e1071) # For Bayesian approach
raw_train <- read.csv(file="../input/train.csv", header=TRUE, sep=",", na.strings=c("")) #891 reg
# Number of NAs for each attribute:

sapply(raw_train,function(x) sum(is.na(x)))
# Function to calculate the mode of a distribution

moda <- function(x) {

  tab <- table(x)

  return(names(tab[which.max(tab)]))

}
# Assign the average of the Age attribute:

raw_train$Age[is.na(raw_train$Age)] <- mean(raw_train$Age,na.rm=T)



# Assign the mode of the Embarked attribute:

moda_embarked <- moda(raw_train$Embarked[!is.na(raw_train$Embarked)])

raw_train$Embarked[is.na(raw_train$Embarked)] <- moda_embarked

train_nb <- raw_train

train_glm <- raw_train
# Another option would be to remove the NA values:

# train_nb <- raw_train[!is.na(raw_train$Age), ]
# Age Histogram, and the mean marked with red-dashed line.

ggplot(train_nb, aes(x=Age)) +

    geom_histogram(binwidth=5, colour="black", fill="white") +

    geom_vline(aes(xintercept=mean(Age, na.rm=T)),   # Ignore NA values for mean

               color="red", linetype="dashed", size=1)
# Generating a new variable discretizing the Age:

train_nb$AgeTramo <- cut(train_nb$Age, breaks = seq(0,80,5))



#Assigning factors to discrete variables, including de label feature (Survived):

train_nb$Pclass_f <- factor(train_nb$Pclass)

train_nb$SibSp_f <- factor(train_nb$SibSp)

train_nb$Parch_f <- factor(train_nb$Parch)

train_nb$Survived_f <- factor(train_nb$Survived)
clasificador <- naiveBayes(Survived_f ~ Pclass_f + Sex + AgeTramo + SibSp_f + Parch_f + Embarked,

                            data=train_nb)

train_nb_features <- train_nb[ , c("Pclass_f", "Sex", "AgeTramo", "SibSp_f", "Parch_f", "Embarked")]

predicted_train_nb<-predict(clasificador, train_nb_features)

matrizconf<-table(predicted_train_nb, train_nb$Survived_f)

#Resultados

matrizconf

sum(diag(matrizconf))/sum(matrizconf)
raw_train <- read.csv(file="../input/train.csv", header=TRUE, sep=",", na.strings=c("")) # 891 reg

train_glm <- raw_train[!is.na(raw_train$Age), ] # 712 reg
# Applying GLM regression model with the binomial family:

clasificador_glm <- glm(formula = Survived ~ Pclass + Sex + Age + SibSp + Parch + Embarked + Fare,

                        data = train_glm,

                        family=binomial())

summary(clasificador_glm) #output

train_glm_features <- train_glm[ , c("Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Fare")]

predicted_train_glm <- predict(clasificador_glm, newdata=train_glm_features)



# Segmentation of the positive and negative predictions.

predicted_train_glm_bin <- ifelse(predicted_train_glm < 0.5, 0, 1)

# Confusion matrix to calculate the accuracy of the model:

matrizconf<-table(predicted_train_glm_bin, train_glm$Survived)

matrizconf

sum(diag(matrizconf))/sum(matrizconf)
library(keras) # For Neural Network deployment
raw_train <- read.csv(file="../input/train.csv", header=TRUE, sep=",", na.strings=c("")) #891 reg

raw_train <- raw_train[!is.na(raw_train$Age), ] # 714 reg

raw_train <- raw_train[!is.na(raw_train$Embarked), ] # 712 reg
# Transform categorical variables to numeric

raw_train$NumSex <- as.numeric(factor(raw_train$Sex,labels=c(1,2)))

raw_train$NumEmbarked <- as.numeric(factor(raw_train$Embarked,labels=c(1,2,3)))



# Generate the feature matrix and the label vector:

train_nn <- raw_train[ , c("Pclass", "NumSex", "Age", "SibSp", "Parch", "NumEmbarked", "Fare")]

label_nn <- raw_train$Survived



# Rescale the values:

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



# Convert dataframe to numeric matrix:

colnames(train_nn)=NULL

for (i in 1:7){train_nn[,i]=as.numeric(train_nn[,i])}

train_nn=as.matrix(train_nn)

label_nn=as.matrix(label_nn)
set.seed(300)

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