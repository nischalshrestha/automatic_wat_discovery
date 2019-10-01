library(Hmisc)

library("caret")

library("rpart")

library("tree")

library("e1071")

library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(randomForest)

set.seed(1)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Read data

train <- read.csv("../input/train.csv" , stringsAsFactors = FALSE)

test  <- read.csv("../input/test.csv",   stringsAsFactors = FALSE)



# Any results you write to the current directory are saved as output.
head(train)
str(train)
describe(train)
# A function to extract features

    # Missing data imputation

extractFeatures <- function(data) {

  features <- c("Pclass",

                "Age",

                "Sex",

                "Parch",

                "SibSp",

                "Fare",

                "Embarked")

  fea <- data[,features]

  fea$Age[is.na(fea$Age)] <- -1

  fea$Fare[is.na(fea$Fare)] <- median(fea$Fare, na.rm=TRUE)

  fea$Embarked[fea$Embarked==""] = "S"

  fea$Sex      <- as.factor(fea$Sex)

  fea$Embarked <- as.factor(fea$Embarked)

  #fea <- cbind(fea, fea$Age * fea$Age)

    return(fea)

}
summary(extractFeatures(train))
summary(extractFeatures(test))
rf <- randomForest(extractFeatures(train), as.factor(train$Survived), ntree=100, importance=TRUE)

rf
# create submission file

submission <- data.frame( PassengerId= test$PassengerId )  # create a dataframe

# using model rf fit on training data to predict test data

submission$Survived <- predict( rf, extractFeatures(test) )  

# write results to CSV file

write.csv(submission, file = "1_random_forest_r_submission.csv", row.names=FALSE)

# plot importance of preditors

imp <- importance(rf, type=1 )

imp
# Use ggplot to plot the importance

p <- ggplot(featureImportance, aes(x= reorder(Feature, Importance) , y = Importance) ) +

            geom_bar(stat = "identity", fill = "#52cfff") +

            coord_flip() +

            theme_light(base_size = 20) +

            xlab("") + 

            ylab("Importance")+

            ggtitle("Ramdom Forest Feature Importance\n") +

            theme(plot.title= element_text(size=18))

           

ggsave("2_feature_importance.png", p)

p
print(row.names(imp))

featureImportance <- data.frame(Feature = row.names(imp), Importance = imp[, 1])
# Classification Tree with rpart

library(rpart)



# grow tree 

fol= formula( as.factor(Survived) ~ Pclass + Age + Sex + Parch + SibSp + Fare + Embarked)

fit <- rpart( fol, data=train, method= "class")
print(fit)     #print results  

printcp(fit)   #display cp table  

plotcp(fit)    #plot cross-validation results 

#rsq.rpart(fit) #plot approximate R-squared and relative error for different splits (2 plots). labels are only appropriate for the "anova" method.  

summary(fit)   #detailed results including surrogate splits  
 # plot tree 

plot(fit, uniform=TRUE, main="Classification Tree")      #plot decision tree  

text(fit, use.n=FALSE, all=TRUE, cex=.8 )      #label the decision tree plot  
library(e1071) 

Survived= train$Survived

train1= cbind(Survived,  extractFeatures(train))

SVMmodel <- svm(fol, data= train1)



#print(SVMmodel) 

summary(SVMmodel) # view results 

# create submission file

submSVM <- data.frame( PassengerId= test$PassengerId )  # create a dataframe

# using model rf fit on training data to predict test data

submSVM$Survived <- predict( SVMmodel, extractFeatures(test) )  

# write results to CSV file

write.csv(submSVM, file = "3_SVM_r_submission.csv", row.names=FALSE)
