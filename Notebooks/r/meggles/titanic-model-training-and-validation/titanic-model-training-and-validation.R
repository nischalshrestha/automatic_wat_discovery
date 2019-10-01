#load packages

library("ggplot2") # data visualization

library("caret") # multiple model training/testing functions

library("readr") # CSV file I/O, e.g. the read_csv function

library("dplyr") # several Hadley Wickham data-wrangling packages

library("mice") # imputing missing values

library("VIM") # visualizing missing values

library("stringr") # feature engineering

library("arules") # feature engineering

library("corrplot") # correlogram 

library("randomForest") # random forest model



options(warn=-1) # turn warnings off 
# read in the data

train_full <- read_csv('../input/train.csv')



# train split into train and validate

inTrain <- createDataPartition(train_full$Survived, times = 1, p = 0.8, list=F)



train <- train_full[inTrain,]

val <- train_full[-inTrain,]
nrow(train) # number of training observations

nrow(val) # number of training observations
   

    train <- mutate(train,

                Cabin_Deck = str_sub(Cabin,1,1),

                Ticket_Digit = nchar(Ticket),

                Ticket_Alpha = str_detect(Ticket, '[[:alpha:]]'),

                Family_Size = Parch+SibSp,

                Name_Family = gsub(",.*$", "", Name),

                Title = str_sub(Name, 

                                str_locate(Name, ",")[ , 1] + 2, 

                                str_locate(Name, "\\.")[ , 1] - 1)

               )

    # credit to https://www.kaggle.com/c/titanic/discussion/30733 for Title regex



    train_sub <- select(train,

                          Survived,Pclass,Sex,Age,SibSp,Parch,Fare,

                          Embarked,Cabin_Deck,Ticket_Digit,Ticket_Alpha,Name_Family,

                          Title,Family_Size)



    train_mm <- model.matrix(~Pclass+Sex+Age+SibSp+Parch+Fare+

                               Embarked+Cabin_Deck+Title+Family_Size+Ticket_Alpha,

                                          train_sub)



    train_imp <- mice(train_sub, 

                        m = 1,

                        method = "cart",

                        seed = 5,

                        printFlag=F)



    train <- complete(train_imp)



    train <- mutate(train, 

                    Cabin_Deck_i = ifelse(!is.na(Cabin_Deck),

                                    Cabin_Deck,

                                    ifelse(Pclass == 1,# read in the data

                                           'ABCD', 

                                            # not including T because only one passenger

                                            # in the training set was assigned cabin T

                                           ifelse(Pclass == 2,

                                                  'E',

                                                 'F'))))



    train_Pclass1 <- filter(train, Pclass == 1) 



    cuts <- discretize(train_Pclass1$Fare,

                       method = 'cluster',

                       categories = 4,

                       ordered = T,

                       onlycuts = T)



    train <- mutate(train, Cabin_Deck_i2 = ifelse(Cabin_Deck_i != "ABCD",

                                           Cabin_Deck_i,

                                           ifelse(Fare < cuts[2],

                                                 "D",

                                                 ifelse(Fare < cuts[3],

                                                       "C",

                                                       ifelse(Fare < cuts[4],

                                                             "B", 

                                                             "A")))))

    train <- mutate(train, Cabin_Deck_i3 = ifelse(Cabin_Deck_i2 == 'A',1,

                                    ifelse(Cabin_Deck_i2 == 'B',2,

                                          ifelse(Cabin_Deck_i2 == 'C',3,

                                                ifelse(Cabin_Deck_i2 == 'D',4,

                                                      ifelse(Cabin_Deck_i2 == 'E',5,

                                                            ifelse(Cabin_Deck_i2 == 'F',6,

                                                                  ifelse(Cabin_Deck_i2 == 'G',7,8))))))))

    train <- mutate(train, 

                    Embarked = ifelse(is.na(Embarked),

                                     'S', Embarked))


    val <- mutate(val,

                Cabin_Deck = str_sub(Cabin,1,1),

                Ticket_Digit = nchar(Ticket),

                Ticket_Alpha = str_detect(Ticket, '[[:alpha:]]'),

                Family_Size = Parch+SibSp,

                Name_Family = gsub(",.*$", "", Name),

                Title = str_sub(Name, 

                                str_locate(Name, ",")[ , 1] + 2, 

                                str_locate(Name, "\\.")[ , 1] - 1)

               )

    # credit to https://www.kaggle.com/c/titanic/discussion/30733 for Title regex



    val_sub <- select(val,

                          Survived,Pclass,Sex,Age,SibSp,Parch,Fare,

                          Embarked,Cabin_Deck,Ticket_Digit,Ticket_Alpha,Name_Family,

                          Title,Family_Size)



    val_mm <- model.matrix(~Pclass+Sex+Age+SibSp+Parch+Fare+

                               Embarked+Cabin_Deck+Title+Family_Size+Ticket_Alpha,

                                          val_sub)



    val_imp <- mice(val_sub, 

                        m = 1,

                        method = "cart",

                        seed = 5,

                        printFlag=F)



    val <- complete(val_imp)



    val <- mutate(val, 

                    Cabin_Deck_i = ifelse(!is.na(Cabin_Deck),

                                    Cabin_Deck,

                                    ifelse(Pclass == 1,

                                           'ABCD', 

                                            # not including T because only one passenger

                                            # in the training set was assigned cabin T

                                           ifelse(Pclass == 2,

                                                  'E',

                                                 'F'))))



    val_Pclass1 <- filter(val, Pclass == 1) 



    cuts <- discretize(val_Pclass1$Fare,

                       method = 'cluster',

                       categories = 4,

                       ordered = T,

                       onlycuts = T)



    val <- mutate(val, Cabin_Deck_i2 = ifelse(Cabin_Deck_i != "ABCD",

                                           Cabin_Deck_i,

                                           ifelse(Fare < cuts[2],

                                                 "D",

                                                 ifelse(Fare < cuts[3],

                                                       "C",

                                                       ifelse(Fare < cuts[4],

                                                             "B", 

                                                             "A")))))

    val <- mutate(val, Cabin_Deck_i3 = ifelse(Cabin_Deck_i2 == 'A',1,

                                    ifelse(Cabin_Deck_i2 == 'B',2,

                                          ifelse(Cabin_Deck_i2 == 'C',3,

                                                ifelse(Cabin_Deck_i2 == 'D',4,

                                                      ifelse(Cabin_Deck_i2 == 'E',5,

                                                            ifelse(Cabin_Deck_i2 == 'F',6,

                                                                  ifelse(Cabin_Deck_i2 == 'G',7,8))))))))

    val <- mutate(val, 

                    Embarked = ifelse(is.na(Embarked),

                                     'S', Embarked))
train <- mutate(train,

                Survived = as.factor(Survived),

                Sex = as.factor(Sex),

                Embarked = as.factor(Embarked),

                Cabin_Deck_i2 = as.factor(Cabin_Deck_i2),

                Name_Family = as.factor(Name_Family),

                Ticket_Alpha = as.factor(Ticket_Alpha),

                Title = as.factor(Title))



val <- mutate(val,

               Survived = as.factor(Survived),

               Sex = as.factor(Sex),

                Embarked = as.factor(Embarked),

                  Ticket_Alpha = as.factor(Ticket_Alpha),

                Cabin_Deck_i2 = factor(Cabin_Deck_i2, levels = levels(train$Cabin_Deck_i2)),

                Name_Family = factor(Name_Family, levels = levels(train$Name_Family)),

                Title = factor(Title, levels = levels(train$Title)))
dt_model <- train(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+

                  Embarked+Cabin_Deck_i2+Ticket_Digit+Ticket_Alpha+

                  Family_Size+Title,

                  data=train,  method = "rpart",

                  tuneLength = 10)

dt_model$finalModel



val_dt <- select(val, Survived,Pclass,Sex,Age,SibSp,Parch,Fare,

                 Embarked,Cabin_Deck_i2,Ticket_Digit,Ticket_Alpha,

                 Family_Size,Title) %>%

filter(!is.na(Cabin_Deck_i2) & !is.na(Ticket_Digit) & !is.na(Title))



dt_pred <- predict(dt_model, val_dt[,-1])

confusionMatrix(dt_pred, val_dt$Survived, positive = "1")
rf_model <- randomForest(Survived~

                         Pclass+Sex+Age+SibSp+Parch+Fare+

                         Embarked+Cabin_Deck_i2+Ticket_Digit+Ticket_Alpha+

                         Family_Size+Title,

                      data=train, 

                      importance=T, 

                      ntree=2000)

varImpPlot(rf_model)

val_rf <- select(val, Survived,Pclass,Sex,Age,SibSp,Parch,Fare,

                           Embarked,Cabin_Deck_i2,Ticket_Digit,Ticket_Alpha,Family_Size,Title)

rf_pred <- predict(rf_model, val_rf[,-1])

confusionMatrix(rf_pred, val_rf$Survived, positive = "1")
library(e1071)

val_nb <- select(val,Survived,Pclass,Sex,Age,SibSp,Parch,Fare,

                           Embarked,Cabin_Deck_i2,Ticket_Digit,Ticket_Alpha,Family_Size,Title)

nb_model <- naiveBayes(Survived~

                           Pclass+Sex+Age+SibSp+Parch+Fare+

                           Embarked+Cabin_Deck_i2+Ticket_Digit+Ticket_Alpha+

                            Family_Size+Title,

           data = train)

nb_pred <- predict(nb_model,val_nb[,-1])

confusionMatrix(nb_pred, val_nb$Survived, positive = "1")
val_svm <- select(val,Survived,Pclass,Sex,Age,SibSp,Parch,Fare,

                           Embarked,Cabin_Deck_i2,Ticket_Digit,Ticket_Alpha,Family_Size,Title) %>%

filter(!is.na(Cabin_Deck_i2) & !is.na(Ticket_Digit) & !is.na(Title))



svm_model <- svm(Survived~

                           Pclass+Sex+Age+SibSp+Parch+Fare+

                           Embarked+Cabin_Deck_i2+Ticket_Digit+Ticket_Alpha+

                            Family_Size+Title,

           data = train)

svm_pred <- predict(svm_model,val_svm[,-1])

confusionMatrix(svm_pred, val_svm$Survived, positive = "1")
library(sgd)



val_sgd <- select(val,Survived,Pclass,Sex,Age,SibSp,Parch,Fare,

                           Embarked,Cabin_Deck_i2,Ticket_Digit,Ticket_Alpha,Family_Size,Title) %>%

filter(!is.na(Cabin_Deck_i2) & !is.na(Ticket_Digit) & !is.na(Title))



sgd_model <- svm(Survived~

                           Pclass+Sex+Age+SibSp+Parch+Fare+

                           Embarked+Cabin_Deck_i2+Ticket_Digit+Ticket_Alpha+

                            Family_Size+Title,

           data = train)

sgd_pred <- predict(sgd_model,val_sgd[,-1])

confusionMatrix(sgd_pred, val_sgd$Survived, positive = "1")