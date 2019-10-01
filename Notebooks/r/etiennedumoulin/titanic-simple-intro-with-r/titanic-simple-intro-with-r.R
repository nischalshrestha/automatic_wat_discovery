library(ggplot2)

library(dplyr,warn.conflicts = FALSE)

library(formattable,warn.conflicts = FALSE)

library(rpart,warn.conflicts = FALSE)

library(mice,warn.conflicts = FALSE)

library(randomForest,warn.conflicts = FALSE)

library(caret,warn.conflicts = FALSE)



tr = read.csv("../input/train.csv")

pr = read.csv("../input/test.csv")

pr$Survived <- NA

mergedData <- rbind(tr,pr)
formattable(head(tr,20))
summary(tr)
trPclass <- tr %>% group_by(Pclass) %>% summarize(cnt=n(),surv=mean(Survived)) %>% arrange(-surv) %>% as.data.frame()

formattable(trPclass)
trSex <- tr %>% group_by(Sex) %>% summarize(cnt=n(),surv=mean(Survived)) %>% arrange(-surv) %>% as.data.frame()

formattable(trSex)

ggplot(subset(tr,!is.na(Age)), aes(x = Age, fill = factor(Survived))) + geom_histogram(bins=15,position = 'dodge')

ggplot(tr, aes(x = SibSp, fill = factor(Survived))) + geom_bar(position='dodge')
ggplot(tr, aes(x = Parch, fill = factor(Survived))) + geom_bar(position='dodge')
ggplot(tr, aes(x = Fare, fill = factor(Survived))) + geom_histogram(bins=15,position='dodge')
trEmbarked <- tr %>% group_by(Embarked) %>% summarize(cnt=n(),surv=mean(Survived)) %>% arrange(-surv) %>% as.data.frame()

formattable(trEmbarked)
t <- table(mergedData$Ticket)

mergedData$sharedTicket <- sapply(mergedData$Ticket,function(x) pmin(t[[x]],4) )

trSharedTicket <- mergedData[!is.na(mergedData$Survived),] %>% group_by(sharedTicket) %>% summarize(cnt=n(),surv=mean(Survived)) %>% arrange(-cnt) %>% as.data.frame()

formattable(trSharedTicket)

t <- NULL
mergedData$surname <- gsub(',.*', '', mergedData$Name)

t <- table(mergedData$surname)

mergedData$sharedSurname <- sapply(mergedData$surname,function(x) pmin(t[[x]],4))

trSharedSurname <- mergedData[!is.na(mergedData$Survived),] %>% group_by(sharedSurname) %>% summarize(cnt=n(),surv=mean(Survived)) %>% arrange(-cnt) %>% as.data.frame()

formattable(trSharedSurname)



mergedData$surname <- NULL

t <- NULL
mergedData$sharedCabin <- 'N'

mergedData$Cabin <- as.character(mergedData$Cabin)

mergedData[nchar(mergedData$Cabin) > 0,"sharedCabin"] <- duplicated(mergedData[nchar(mergedData$Cabin) > 0,"Cabin"])|duplicated(mergedData[nchar(mergedData$Cabin) > 0,"Cabin"],fromLast=TRUE)



trSharedCabin <- mergedData[!is.na(mergedData$Survived),] %>% group_by(sharedCabin) %>% summarize(cnt=n(),surv=mean(Survived)) %>% arrange(-cnt) %>% as.data.frame()

formattable(trSharedCabin)
mergedData$title <- gsub('(.*, )|(\\..*)', '', mergedData$Name)

mergedData$title[mergedData$title == "Ms"] <- "Miss"

mergedData$title[mergedData$title == "Mlle"] <- "Miss"

mergedData$title[!mergedData$title %in% c("Miss","Mrs","Mr")] <- "Other"



trTitle <- mergedData[!is.na(mergedData$Survived),] %>% group_by(title) %>% summarize(cnt=n(),surv=mean(Survived)) %>% arrange(-surv) %>% as.data.frame()

formattable(trTitle)
formattable(subset(mergedData,nchar(as.character(Embarked)) == 0))
mergedData$Name <- NULL

mergedData$Ticket <- NULL

mergedData$Cabin <- NULL

mergedData$ticketsCnt <- NULL

mergedData$ticketLetter <- NULL

trClean1 <- mergedData

class_emb_mod <- rpart(Embarked ~ . - Survived, data=subset(trClean1,nchar(as.character(Embarked)) > 0), method="class", na.action=na.omit) 

emb_pred <- predict(class_emb_mod, subset(trClean1,nchar(as.character(Embarked)) == 0))

trClean1$Embarked[nchar(as.character(trClean1$Embarked)) == 0] <- colnames(emb_pred)[apply(emb_pred, 1, which.max)]
formattable(head(subset(tr,is.na(Age))))
trClean2 <- trClean1

# perform mice imputation, based on random forests

miceMod <- mice(trClean2[,!names(trClean2) %in% "Survived"], method="rf",printFlag=FALSE) 

# generate the completed data

trClean2[,!names(trClean2) %in% "Survived"] <- complete(miceMod)

#anyNA(miceOutput)

head(trClean2)
set.seed(415)

trClean <- trClean2

#lapply(trClean, class)

trClean$Survived <- as.factor(trClean$Survived)

trClean$title <- as.factor(trClean$title)

trClean$sharedCabin <- as.factor(trClean$sharedCabin)



trRF <- trClean[!is.na(trClean$Survived),]

fit <- randomForest(Survived ~ . - PassengerId,

                      data=trRF, 

                      importance=TRUE, 

                      ntree=1000)
fit$confusion



successRate <- (fit$confusion[1,1]+fit$confusion[2,2]) / (fit$confusion[1,1]+fit$confusion[2,2]+fit$confusion[2,1]+fit$confusion[1,2])

print(successRate)
flds <- createFolds(trRF$PassengerId, k = 10, list = TRUE, returnTrain = FALSE)

#head( trRF[flds[[1]],!names(trRF) %in% c("PassengerId","Survived") ])

conf <- NA

for(i in 1:10){

    fitTest <- randomForest(Survived ~ . - PassengerId,

                      data=trRF[-flds[[1]], ], 

                      importance=TRUE, 

                      ntree=1000,

                      xtest = trRF[flds[[1]],!names(trRF) %in% c("PassengerId","Survived") ],

                      ytest = trRF[flds[[1]], "Survived"])

    if(!is.matrix(conf)){

        conf <- fitTest$test$confusion

    }else{

        conf <- conf + fitTest$test$confusion

    }

}

conf[1,3] <- conf[1,2] /(conf[1,1]+conf[1,2])

conf[2,3] <- conf[2,1] /(conf[2,1]+conf[2,2])

conf
varImpPlot(fit)
prRF <- trClean[is.na(trClean$Survived),]

prediction <- predict(fit, prRF)

prRF$Survived <-prediction

formattable(head(prRF,20))
solution <- data.frame(PassengerId = prRF$PassengerId, Survived = prRF$Survived)

write.csv(solution, file = 'output.csv')