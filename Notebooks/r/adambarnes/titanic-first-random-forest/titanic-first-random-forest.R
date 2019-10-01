options(stringsAsFactors = FALSE)



library(dplyr)

library(ggplot2)



trainData <- read.csv("../input/train.csv")

testData <- read.csv("../input/test.csv")



fullData <- trainData %>%

    bind_rows(testData)



str(trainData)
# might be easier to use cut next time?

ageClass <- function(Age) {

    if_else(Age <10,"0-9",

           if_else(Age < 20,"10-19",

                  if_else(Age < 30,"20-29",

                         if_else(Age < 40,"30-39",

                                if_else(Age < 50,"40-49",

                                       if_else(Age < 60,"50-59","60+"))))))

}
workData <- trainData %>%

    bind_rows(testData) %>%

    mutate(AgeClass = ageClass(Age),

          CabinClass = substr(Cabin,1,1),

          Title = substr(Name,unlist(gregexpr(",",Name)) + 2,unlist(gregexpr(",",Name)) + 4),

          Title = gsub('\\.','',Title),

          Title = factor(gsub(' ','',Title)),

          Sex = factor(Sex),

          CabinClass = factor(CabinClass),

          Embarked = factor(Embarked),

          Survived = factor(Survived),

          Pclass = factor(Pclass)) %>%

        anti_join(testData,by = "PassengerId")



str(workData)

ggplot(workData,aes(x = Sex)) + geom_bar(aes(fill = factor(Survived))) + 

    scale_x_discrete(labels = c("F","M")) + 

    facet_grid(Pclass~AgeClass)
# average age by sex, Pclass,Parch

avAges <- workData %>%

    filter(!is.na(Age)) %>%

    group_by(Sex,Pclass,Parch) %>%

    summarize(AvAge = round(mean(Age),0))



avFare <- workData %>%

    filter(!is.na(Fare)) %>%

    group_by(Pclass,Embarked) %>%

    summarize(AvFare = mean(Fare))



head(avFare,20)
# replace NA ages with averages, recalc age class

workData <- workData %>%

    left_join(avAges, by = c("Sex" = "Sex","Pclass" = "Pclass","Parch" = "Parch")) %>%

    left_join(avFare,by = c("Pclass" = "Pclass","Embarked" = "Embarked")) %>%

    mutate(Age = if_else(is.na(Age),round(AvAge,0),Age),

         AgeClass = factor(ageClass(Age)),

         Fare = if_else(is.na(Fare),round(AvFare,0),Fare))



head(workData)

rownames(workData) <- NULL
library(randomForest)



forData <- workData %>%

    select(Survived,Sex,Fare,Pclass,Age,Title,SibSp,Parch,Embarked) 



rf1 <- randomForest(Survived~., data=forData, mtry=2, ntree=50, importance=TRUE)

importance(rf1,type=1)
resultSet <- data.frame(ID = workData$PassengerId,

           Act = workData$Survived,

           Pred = predict(rf1,workData)) 



#check accuracy

sum(resultSet$Act == resultSet$Pred) / nrow(resultSet)

# 91% using Survived,Sex,Fare,Pclass,AgeClass,Title,SibSp,Parch,CabinClass,Embarked

# 89% using Survived,Pclass,Sex,AgeClass,SibSp,Parch,Fare,Title

# 90% using Survived,Pclass,Sex,AgeClass,SibSp,Parch,Fare,CabinClass,Title
# prepare submission file

avAgesTest <- fullData %>%

    filter(!is.na(Age)) %>%

    group_by(Pclass,Parch) %>%

    summarize(AvAge = mean(Age))



avFareTest <- testData %>%

    filter(!is.na(Fare)) %>%

    group_by(Pclass,Embarked) %>%

    summarize(AvFare = mean(Fare))



modTestData <- testData %>%

    bind_rows(trainData) %>%

    left_join(avAgesTest, by = c("Pclass" = "Pclass","Parch" = "Parch")) %>%

    left_join(avFareTest,by = c("Pclass" = "Pclass","Embarked" = "Embarked")) %>%

    mutate(Age = if_else(is.na(Age),round(AvAge,0),Age),

           Age = if_else(is.na(Age),mean(Age,na.rm = TRUE),Age),

           Fare = if_else(is.na(Fare),round(AvFare,0),Fare),

            AgeClass = ageClass(Age),

            CabinClass = substr(Cabin,1,1),

            Title = substr(Name,unlist(gregexpr(",",Name)) + 2,unlist(gregexpr(",",Name)) + 4),

            Title = gsub('\\.','',Title),

            Title = factor(gsub(' ','',Title)),

            Sex = factor(Sex),

            CabinClass = factor(CabinClass),

            Embarked = factor(Embarked),

            Pclass = factor(Pclass)) %>%

    anti_join(trainData,by = "PassengerId") %>%

    select(-Survived)



rownames(modTestData) <- NULL



submitData <- data.frame(PassengerId = modTestData$PassengerId,

            Survived = predict(rf1,modTestData),

                       row.names = NULL) 





str(submitData)

head(submitData)
write.csv(submitData,"submission.csv",row.names = FALSE)