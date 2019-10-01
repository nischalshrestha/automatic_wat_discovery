train<-read.csv("/kaggle/input/train.csv", sep=",")

test<-read.csv("/kaggle/input/test.csv", sep=",")

test$Survived<-0 

tot<-rbind(train,test) #Combine train and test dataframes



library(corrplot) #for correlation chart

library(mice) #for NA fulfillment

library(randomForest) #for modeling

library(ggplot2) #for data visualization
summary(tot)
tot$FamilyAboardS<-1+tot$SibSp+tot$Parch #Count familysize

aggregate(tot$Age, by=list(tot$FamilyAboardS), FUN=mean, na.rm=TRUE) #Average age by family size
ggplot(tot[1:891,], aes(x=factor(FamilyAboardS), fill=factor(Survived)))+geom_bar(stat='count', position='dodge')
aggregate(tot$Fare, by=list(tot$Pclass), FUN=mean, na.rm=TRUE)
ggplot(tot[1:891,], aes(x=factor(Pclass), fill=factor(Survived)))+geom_bar(stat='count', position='dodge')
ggplot(tot[1:891,], aes(x=factor(Sex), fill=factor(Survived)))+geom_bar(stat='count', position='dodge')
ggplot(tot[1:891,], aes(x=factor(Embarked), y=Fare, fill=factor(Survived)))+geom_boxplot()
tot$Name<-as.character(tot$Name)

tot$Title<-sapply(tot$Name, function(x) strsplit(x, split='[,.]')[[1]][2])

tot$Title<-sub(' ', '', tot$Title)

tot$Title[tot$Title %in% c('Miss', 'Mlle', 'Lady', 'Ms')]<-'Miss'

tot$Title[tot$Title %in% c('Dona', 'Dr', 'Mme', 'Mrs')]<-'Mrs'

tot$Title[tot$Title=='Mrs' & tot$Sex=='male']<-'Mr'

tot$Title[tot$Title %in% c('Don', 'Capt', 'Col', 'Dr', 'Major', 'Master', 'Rev', 'Sir', 'Jonkheer')]<-'Mr'

tot$Title[tot$Title %in% c('the Countess')]<-'Mrs' #Noël Leslie, Countess of Rothes была was marriage at the disaster moment

aggregate(tot$Age, by=list(tot$Title), FUN=mean, na.rm=TRUE)
ggplot(tot[1:891,], aes(x=factor(Title), fill=factor(Survived)))+geom_bar(stat='count', position='dodge' )
#Fare has 1 NA in 3d class

tot[is.na(tot$Fare),] 

#Fulfill Fare NA by median Fare for 3dclass

tot$Fare[is.na(tot$Fare)]<-median(tot$Fare[tot$Pclass %in% c(3)], na.rm=TRUE) 
#Embarked has 2 NA with average Fare $80 - one o the most expensive. As we know - it is Cherbourg

aggregate(tot$Fare, by=list(tot$Embarked), FUN=mean ,na.rm=TRUE)

#Fulfill it with 'C'

tot$Embarked<-as.character(tot$Embarked)

tot$Embarked[tot$Embarked %in% c('')]<-'C'

tot$Embarked<-as.factor(tot$Embarked)
#There are some Fare filled by zero

tot[tot$Fare==0,]
#Fulfill zero- Fare with median Fare for class

for(i in 1:3){

	tot$Fare[tot$Fare==0&tot$Pclass==i]<-median(tot$Fare[tot$Pclass %in% c(i)], na.rm=TRUE)

	}
#I use the package "mice". It detectes NAs then uses other variables to restore target variable.







#imp<-mice(tot, seed=1234, method="rf")

#imp$imp$Age

#I chose 5th set

#mice_tot<-complete(imp, action=5)

#tot$Age<-mice_tot$Age



#To avoid a Kaggle server overload I will load Age variable directlt as a vector

mice_v<-c(22,38,26,35,35,26,54,2,27,14,4,58,20,39,14,55,2,31,31,26,35,34,15,28,8,38,21,19,38,20,40,38,55.5,66,28,42,26,21,18,14,40,27,20,3,19,26,40,21,22,18,7,21,49,29,65,19,21,28.5,5,11,22,38,45,4,24,1,29,19,17,26,32,16,21,26,32,25,21,21,0.83,30,22,29,22,28,17,33,16,28,23,24,29,20,46,26,59,22,71,23,34,34,28,28,21,33,37,28,21,27,38,18,47,14.5,22,20,17,21,70.5,29,24,2,21,30,32.5,32.5,54,12,17,24,4,45,33,20,47,29,25,23,19,37,16,24,34,22,24,19,18,19,27,9,36.5,42,51,22,55.5,40.5,21,51,16,30,24,36,44,40,26,17,1,9,33,45,56,28,61,4,1,21,56,18,8,50,30,36,3,19,9,1,4,44,18,45,40,36,32,19,19,3,44,58,30,42,21,24,28,2,34,45.5,18,2,32,26,16,40,24,35,22,30,18,31,27,42,32,30,16,27,51,29,38,22,19,20.5,18,40,35,29,59,5,24,22,44,8,19,33,22,18,29,22,30,44,25,24,37,54,34.5,29,62,30,41,29,60,30,35,50,18,3,52,40,22,36,16,25,58,35,42,25,41,37,34,63,45,42,7,35,65,28,16,19,54,33,30,22,42,22,26,19,36,24,24,62,23.5,2,4,50,15,21,19,44,15,0.92,32,17,30,30,24,18,26,28,43,26,24,54,31,40,22,27,30,22,2,36,61,36,31,16,22,45.5,38,16,30,25,29,41,45,45,2,24,28,25,36,24,40,15,3,42,23,38,15,25,23,28,22,38,22,23,40,29,45,35,16,30,60,22,21,24,25,18,19,22,3,40,22,27,20,19,42,1,32,35,24,18,1,36,35,17,36,21,28,23,24,22,31,46,23,28,39,26,21,28,20,34,51,3,21,21,21,20,33,25,44,22,34,18,30,10,29,21,29,28,18,30,28,19,18,32,28,31,42,17,50,14,21,24,64,31,45,20,25,28,20,4,13,34,5,52,36,28,30,49,21,29,65,35,50,31,48,34,47,48,28,38,25,56,19,0.75,21,38,33,23,22,40,34,29,22,2,9,32,50,63,25,8,35,58,30,9,21,21,55,71,21,36,54,21,25,24,17,21,22,37,16,18,33,20,28,26,29,20,36,54,24,47,34,22,36,32,30,22,23,44,25,40.5,50,67,39,23,2,20,17,24,30,7,45,30,36,22,36,9,11,32,50,64,19,34,33,8,17,27,22,22,22,62,48,27,39,36,14.5,40,28,21,21,24,19,29,22.5,32,62,53,36,30,16,19,34,39,29,32,25,39,54,36,26,18,47,60,22,20,35,52,47,18,37,36,28,49,21,49,24,25,48,44,35,36,30,27,22,40,39,20,22,18,35,24,34,26,4,26,27,42,20,21,21,61,57,21,26,29,80,51,32,32.5,9,28,32,31,41,21,20,24,2,23,0.75,48,19,56,21,23,20,18,21,30,18,24,25,32,23,58,50,40,47,36,20,32,25,18,43,63,40,31,70,31,31,18,24.5,18,43,36,20,27,20,14,60,25,14,19,18,15,31,4,32,25,60,52,44,22,49,42,18,35,18,25,26,39,45,42,22,1,24,39,48,29,52,19,38,27,20,33,6,17,34,50,27,20,30,60.5,25,25,29,11,27,23,23,28.5,48,35,24.5,20,22,36,21,24,31,70,16,30,19,31,4,6,33,23,48,0.67,28,18,34,33,24,41,20,36,16,51,40,30.5,27,32,24,48,57,25,54,18,24,5,23,43,13,17,29,16,25,25,18,8,1,46,20,16,2,38,25,39,49,31,30,30,34,31,11,0.42,27,31,39,18,39,33,26,39,35,6,30.5,33,23,31,43,10,52,27,38,27,2,18,24,1,45,62,15,0.83,20,23,18,39,21,15,32,35,20,16,30,34.5,17,42,10,35,28,18,4,74,9,16,44,18,45,51,24,20,41,21,48,6,24,42,27,31,27,4,26,47,33,47,28,15,20,19,23,56,25,33,22,28,25,39,27,19,40,26,32,34.5,47,62,27,22,14,30,26,18,21,9,46,23,63,47,24,35,21,27,45,55,9,47,21,48,50,22,22.5,41,21,50,24,33,47,30,18.5,60.5,21,25,32,39,47,41,30,45,25,45,22,60,36,24,27,20,28,25,10,35,25,29,36,17,32,18,22,13,30,18,47,31,60,24,21,29,28.5,35,32.5,21,55,30,24,6,67,49,13,30,20,27,18,21,2,22,26,27,23,25,25,76,29,20,33,43,27,26,26,16,28,21,20,25,18.5,41,28,36,18.5,63,18,20,1,36,29,12,21,35,28,55.5,17,22,27,42,24,32,53,36,22,43,24,26.5,26,23,40,10,33,61,28,42,31,36,22,29,30,23,25,60.5,36,13,24,29,23,42,26,21,7,26,23,41,26,48,18,61,22,13,27,23,21,40,15,20,54,36,64,30,37,18,20,27,40,21,17,18,40,34,50,11.5,61,8,33,6,18,23,19,27,0.33,47,8,25,31,35,24,33,25,32,22,17,60,38,42,59,57,50,21,30,21,22,21,53,34,23,31,40.5,36,14,21,21,22,39,20,64,20,18,48,55,45,45,21,42,41,22,42,29,21,0.92,20,27,24,32.5,21,55.5,28,19,21,36.5,21,29,1,30,35,39,21,5,17,46,30,26,36.5,32,20,28,40,30,22,23,0.75,21,9,2,36,21,24,25,29,33,30,20,53,36,26,1,16,30,29,32,30,43,24,25,64,30,0.83,55,45,18,22,21,37,55,17,57,19,27,22,26,25,26,33,39,23,12,46,29,21,48,39,23,19,27,30,32,39,25,24,18,32,40,58,13,16,26,38,24,31,45,25,18,49,0.17,50,59,30,26,30,14.5,24,31,27,25,29,38.5,22,45,29,21,31,49,44,54,45,22,21,55,5,26,26,16,19,21,24,24,57,21,6,23,51,13,47,29,18,24,48,22,31,30,38,22,17,43,20,23,50,29,3,30,37,28,22,39,38.5,24,18)

tot$Age<-mice_v
#Let's look at correlation between variables using package 'corrplot'

corM<-tot[,c('Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilyAboardS')]

M<-cor(corM)

corrplot(M, method="number")
#Transform variables

tot$Survived<-as.factor(tot$Survived)

tot$Pclass<-as.factor(tot$Pclass)

tot$Title<-as.factor(tot$Title)



my_train<-tot[1:891,]

my_test<-tot[892:1309,]



#Create a data farame for input and output parametrs storage

check<-data.frame(mtry=integer(0), nodesize=integer(0), ntrees=integer(0), OOB=numeric(0))

set.seed(12345)

for (i in 1:5) {

	for (j in 1:5){

		for (k in seq(50,200, by =10)){

			mod<-randomForest(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+Title, data=my_train, mtry=i, ntree=k, nodesize=j)

			check<-rbind(check, data.frame(mtry=i, nodesize=j, ntrees=k, OOB=mod$err.rate[nrow(mod$err.rate),1]))



		}

	}

}

#Show minimum OOB

check[check$OOB %in% c(min(check$OOB)),]

#Create the model with such parametrs

set.seed(12345)

mod<-randomForest(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+Title, data=my_train, mtry=4, ntree=130, nodesize=4)

#Make a prediction

my_pred <- predict(mod, my_test)

sol<-data.frame(PassengerId=my_test$PassengerId, Survived=my_pred)

write.csv(sol, file="my_solution", row.names=FALSE)

#The most important variables

importance(mod)