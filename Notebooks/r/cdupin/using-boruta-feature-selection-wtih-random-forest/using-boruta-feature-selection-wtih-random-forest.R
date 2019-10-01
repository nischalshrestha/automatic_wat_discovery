# contains R code to:

# -read in Kaggle's Titanic data csv files

# -clean, analyze and standardize dataset

# -apply Boruta package to downselect important variables

# -train data using Caret's random forest interface

# -make predictions on test data 

# -write out test predictions to csv file



# read data

train.df <- read.csv("../input/train.csv", header = TRUE, na.strings = "")

test.df <- read.csv("../input/test.csv", header = TRUE, na.strings = "")



# load libraries

suppressPackageStartupMessages({

    library(plyr)

    library(lattice)

    library(ggplot2)

    library(ranger)

    library(randomForest)

    library(caret) 

    library(Boruta)

})
# merge to simplify data standardization

combined.df <- rbind(train.df, 

                     cbind(test.df, Survived = rep(NA, nrow(test.df))))  



# adjust data classes

combined.df$Survived <- as.factor(combined.df$Survived) 

combined.df$Pclass <- as.factor(combined.df$Pclass)  

combined.df$Name <- as.character(combined.df$Name)  # adjust for string split



# create new FamilySize feature 

combined.df$FamilySize <- combined.df$SibSp + combined.df$Parch + 1



# create CabinLetter variable by extracting letters from Cabin

combined.df$CabinLetter <- substring(combined.df$Cabin, 1, 1)

combined.df$CabinLetter <- ifelse(is.na(combined.df$CabinLetter), "None",  

                                  combined.df$CabinLetter)

combined.df$CabinLetter <- as.factor(combined.df$CabinLetter) 



# extract Titles from Name 

combined.df$Title <- sapply(combined.df$Name, 

                            FUN = function(x) {strsplit(x, split = '[,.]')[[1]][2]})

combined.df$Title <- sub(" ", "", combined.df$Title)  # clean up Title spacing



# consolidate Titles into higher-level categories

combined.df$TitleGroup <- combined.df$Title   

combined.df$TitleGroup[which(combined.df$Title %in% c("Rev"))] <- "MrRev" 



combined.df$TitleGroup[which(combined.df$Title %in% c("Capt", 

                                                      "Col", 

                                                      "Major",

                                                      "Don", 

                                                      "Jonkheer", 

                                                      "Sir"))] <- "MrSpec" 



combined.df$TitleGroup[which(combined.df$Title %in% c("Dona", 

                                                      "Lady", 

                                                      "Mme", 

                                                      "the Countess"))] <- "MrsSpec"



combined.df$TitleGroup[which(combined.df$Title %in% c("Mlle", 

                                                      "Ms"))] <- "Miss"



combined.df$TitleGroup[which(combined.df$Title == "Dr" & 

                             combined.df$Sex == "female")] <- "MrsSpec"



combined.df$TitleGroup[which(combined.df$Title == "Dr" & 

                             combined.df$Sex == "male")] <- "MrSpec"



# create new Surname variable; can this be useful?

combined.df$Surname <- sapply(combined.df$Name, 

                              function(x) strsplit(x, split = '[,.]')[[1]][1])



# transform new Title features into factor classes

combined.df$Title <- as.factor(combined.df$Title)

combined.df$TitleGroup <- as.factor(combined.df$TitleGroup)

combined.df$Surname <- as.factor(combined.df$Surname)
# impute values for 2 Embarked NAs; assigning 'C' given $80 Fare is median value for 'C'

combined.df$Embarked[c(62, 830)] <- 'C'



# impute values for 1 Fare NA; assigning median fare for 3rd class/'S' embarkment

combined.df$Fare[1044] <- median(combined.df[combined.df$Pclass == '3' 

                                             & combined.df$Embarked == 'S', ]$Fare, 

                                 na.rm = TRUE)



# impute values for Age NAs via OLS model fit

lmAge.fit <- lm((Age ~ Pclass + TitleGroup + SibSp + Parch), data = combined.df)  



# assign imputed Age values to NAs 

for(i in 1:nrow(combined.df)) {

  if(is.na(combined.df[i, "Age"])) {

    combined.df[i, "Age"] <- predict(lmAge.fit, newdata = combined.df[i, ])  

  }

}
# transform factor variables to standardized dummy structure

dummyTrnsfrm <- dummyVars(" ~ Pclass + Sex + Embarked + CabinLetter + TitleGroup", 

                          data = combined.df, 

                          fullRank = T)  # set to avoid perfect collinearity 

dummyData.df <- data.frame(predict(dummyTrnsfrm, newdata = combined.df))  

combinedTrnsfrm.df <- cbind(combined.df, dummyData.df)  # merge dummies with all variables
# clean up newly standardized dataset; remove unnecessary variables

combinedTrnsfrm.df  <- subset(combinedTrnsfrm.df, 

                              select = -c(Pclass,  # dummyVars transform

                                          Sex,  # dummyVars transform

                                          Embarked,  # dummyVars transform

                                          CabinLetter,  # dummyVars transform

                                          TitleGroup,  # dummyVars transform

                                          Title,  # TitleGroup instead

                                          Cabin,  # CabinLetter extract

                                          Ticket,  # not using

                                          Surname,  # not using

                                          Name))  # not using



# break combinedTrnsfrm.df back into train.df and test.df  

train.df <- combinedTrnsfrm.df[which(!is.na(combinedTrnsfrm.df$Survived)), ]

test.df <- data.frame(combinedTrnsfrm.df[which(is.na(combinedTrnsfrm.df$Survived)), ], 

                      row.names = NULL)



# adjust factor form

levels(train.df$Survived) <- make.names(levels(factor(train.df$Survived)))  
##############################

# Variable Importance & Downselect

##############################



# run the Boruta package and downselect variables

set.seed(10)

borutaAttr <- Boruta(Survived ~ . - PassengerId, 

                     data = train.df, 

                     maxRuns = 200, 

                     doTrace = 0)

borutaVars <- getSelectedAttributes(borutaAttr)

boruta.formula <- formula(paste("Survived ~ ", 

                                paste(borutaVars, collapse = " + ")))

print(boruta.formula)



plot(borutaAttr, 

     whichShadow = c(FALSE, FALSE, FALSE), 

     cex.axis = .7, 

     las = 2, 

     boxwex = 0.6, 

     xlab = "", main = "Variable Importance")
##############################

# Parameter Controls

##############################



# set general control pararameters

fitControl = trainControl(method = "repeatedcv",

                          classProbs = TRUE,

                          number = 10,

                          repeats = 5, 

                          index = createResample(train.df$Survived, 50),

                          summaryFunction = twoClassSummary,

                          verboseIter = FALSE)
##############################

# Random Forest 

##############################



set.seed(110)

rfBoruta.fit <- train(boruta.formula, 

                      data = train.df, 

                      trControl = fitControl,

                      tuneLength = 4,  # final value was mtry = 4

                      method = "rf",

                      metric = "ROC")

print(rfBoruta.fit$finalModel) 
##############################

# Predictions

##############################



# generate predictions on train data

# trainYhat.df <- cbind(train.df[1:2]) 

# trainYhat.df$rf.boruta <- predict(rfBoruta.fit, newdata = train.df, type = "raw")



# generate predictions on test data

# testYhat.df <- cbind(test.df[1]) 

# testYhat.df$rf.boruta <- predict(rfBoruta.fit, newdata = test.df, type = "raw")
##############################

# Submission

##############################



# extract test predictions and bind relevant vectors per Kaggle submission rules

# testYhat.df$submit <- testYhat.df$rf.boruta  # select the prediction to submit 

# testSubmission.df <- data.frame(cbind(testYhat.df$PassengerId, 

#                                       testYhat.df$submit)) 

# colnames(testSubmission.df) <- c("PassengerId", "Survived")

# testSubmission.df$Survived <- ifelse(testSubmission.df$Survived == 1, 0, 1)  



# save csv file for submission

# write.csv(testSubmission.df, 

#           "~/Submissions/testSubmission_v5.csv", 

#           row.names = FALSE)