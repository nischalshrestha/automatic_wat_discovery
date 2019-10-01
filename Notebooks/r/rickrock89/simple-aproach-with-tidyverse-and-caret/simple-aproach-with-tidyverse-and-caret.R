library(tidyverse)

library(caret)

library(randomForest)


SummarY_Vars <-function(VECTOR){

  Exit <- array(-99999,6)

  if(is.numeric(VECTOR))

  {

  Exit[1] <-  min(VECTOR,na.rm=TRUE)

  Exit[2] <-  max(VECTOR,na.rm=TRUE)

  Exit[3] <-  mean(VECTOR,na.rm=TRUE)

  Exit[4] <-  median(VECTOR,na.rm=TRUE)

  Exit[5] <-  sd(VECTOR,na.rm=TRUE)

  Exit[6] <-  sum(is.na(VECTOR))

  }

  return(Exit)

}



Adj.NA_NUM <- function(VECTOR){

  Exit <- VECTOR

  if(sum(is.na(VECTOR))>0){

    Exit[is.na(Exit)] <- mean(Exit, na.rm=TRUE)

  }

  return(Exit)

}



Adj.NA_NON_NUM <- function(VECTOR){

  Exit <- VECTOR

  if(sum(is.na(VECTOR))>0){

    

    Exit[is.na(Exit)] <- "MISSING"

  }

  return(as.factor(Exit))

}





IS_NOT_NUMERIC <- function(VECTOR){

  return(ifelse(is.numeric(VECTOR),FALSE,TRUE))

}





Adj_Var <- function(DATA){

  DATA_NUM <- DATA %>% 

    purrr::keep(is.numeric) %>% 

    purrr::map(Adj.NA_NUM)

  

  DATA_NON_NUM <- DATA %>% 

    purrr::keep(IS_NOT_NUMERIC)%>% 

    purrr::map(Adj.NA_NON_NUM)%>% 

    as_tibble

  

  Exit <- cbind(DATA_NON_NUM,DATA_NUM)

  

  return(Exit)

}
Data <- read_delim("../input/train.csv",",")

Data.test_out <- read_delim("../input/test.csv",",")
Sum.Vars <- Data %>% 

  map(SummarY_Vars) %>% 

  as_tibble()

Sum.Vars
inTrain <- createDataPartition(y = Data$Survived, p = .7,list = FALSE)



Data.Train <- Data[inTrain,] 

Data.Test <- Data[-inTrain,] 



Data.Train_Adj <- Adj_Var(Data.Train)%>% 

  mutate(Survived_F = as.factor(Survived)) %>% 

  select(-Survived,-PassengerId,-Name,-Ticket,-Cabin) %>% 

  select(Survived_F, everything())



Data.Test_Adj <- Adj_Var(Data.Test)%>% 

  mutate(Survived_F = as.factor(Survived)) %>% 

  select(-Survived,-PassengerId,-Name,-Ticket,-Cabin) %>% 

  select(Survived_F, everything())

cvCtrl <- trainControl(method = "repeatedcv",

                       repeats = 2,

                       savePredictions="final")



RfFit <- train(Survived_F ~ .,

                  data = Data.Train_Adj,

                  method = "rf",

                  tuneLength = 2,

                  trControl = cvCtrl)

Preds <- predict.train(RfFit,Data.Test_Adj)

confusionMatrix(Preds, Data.Test$Survived)