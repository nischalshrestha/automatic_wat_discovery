#Load data and libs 



library(Amelia)

library(rattle)

library(rpart.plot)

library(RColorBrewer)

library(rpart)

library(randomForest)





titanic <- read.csv("../input/train.csv", header = T, na.strings=c(""))

titanicTest <- read.csv("../input/test.csv", header = T, na.strings=c(""))

#We can use sapply function to get the nº of missing values in our dataset

sapply(titanic,function(x) sum(is.na(x)))

    

#missmap help us to get these information in a graph



missmap(titanic, main = "Missing values vs observed")

    
sapply(titanic, function(x) length(unique(x)))

    

barplot(table(titanic$Survived),

        names.arg = c("Murio", "Vivio"),

        main="Survived", col="black")



barplot(table(titanic$Pclass), 

        names.arg = c("Primera", "Segunda", "Tercera"),

        main="Pclass (clase del viajero)", col="firebrick")



barplot(table(titanic$Sex),

        names.arg = c("Mujer", "Hombre"),

        main="Sex (genero)", col="darkviolet")



hist(titanic$Age, main="Age", xlab = NULL, col="brown")



barplot(table(titanic$SibSp), main="SibSp (hijos y esposa a bordo)", 

        col="darkblue")



barplot(table(titanic$Parch), main="Parch (Padres e hijos abordo)", 

        col="gray50")



hist(titanic$Fare, main="Fare (Precio del ticket)", xlab = NULL, 

     col="darkgreen")



barplot(table(titanic$Embarked), 

        names.arg = c("Cherbourg", "Queenstown", "Southampton"),

        main="Embarked (Lugar donde embarcó)", col="sienna")

    

    
#We load in titanic2 the data without the columns that we are speak about (-c(columns))



titanic2 <- titanic[,-c(1,4, 9, 11)]

head(titanic2)

#With Age, we can assign the mean value of the other examples



titanic2$Age[is.na(titanic2$Age)] <- mean(titanic2$Age,na.rm=T)



#On the other hand, with Embarked, at the moment we can remove the two examples with MV.

titanic2 <- titanic2[!is.na(titanic2$Embarked),]

rownames(titanic2) <- NULL
#Train de regression model 



model <- glm(Survived ~.,family=binomial(link='logit'),data=titanic2)



summary(model)



#Apply anova test



anova(model, test="Chisq")
#Table is a useful command for this



table(titanic2$Survived)



#También podemos verlo en forma de proporción



prop.table(table(titanic2$Survived))
prop.table(table(titanic2$Sex, titanic2$Survived),1) 
summary(titanic2$Age)
titanic2$Child<-0

titanic2$Child[titanic2$Age<18]<-1



aggregate(Survived ~ Child + Sex, data=titanic2, FUN=function(x) {sum(x)/length(x)})
titanic2$Child<-0

titanic2$Child[titanic2$Age<14]<-1





aggregate(Survived ~ Child + Sex, data=titanic2, FUN=function(x) {sum(x)/length(x)})
titanic2$Fare2 <- '30+'

titanic2$Fare2[titanic2$Fare < 30 & titanic2$Fare >= 20] <- '20-30'

titanic2$Fare2[titanic2$Fare < 20 & titanic2$Fare >= 10] <- '10-20'

titanic2$Fare2[titanic2$Fare < 10] <- '<10'
aggregate(Survived ~ Fare2 + Pclass + Sex, data=titanic2, FUN=function(x) {sum(x)/length(x)})

aggregate(Survived ~ Child + Fare2 + Pclass + Sex, data=titanic2, FUN=function(x) {sum(x)/length(x)})
