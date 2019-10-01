getwd()





#Load data

train=read.csv("train.csv", header = TRUE)

test=read.csv("test.csv", header = TRUE)



#View data

View(train)

View(test)



#Add survived variable to test to make it 12 columns(= no. of columns in train)

test.survived=data.frame(Survived=rep("None", nrow(test)), test[,])



#Reorder test.survived

test.survived=test.survived[,c(2,1,3:12)]



#Combine data sets

data.combined=rbind(train, test.survived)

str(data.combined)

nrow(data.combined)



#Change into factors

data.combined$Survived=as.factor(data.combined$Survived)

data.combined$Pclass=as.factor(data.combined$Pclass)

str(data.combined)



#Take a look at the gross survival rate

table(data.combined$Survived)



#Distribution across classes

table(data.combined$Pclass)



#Load up ggplot2

library(ggplot2)





#Hypothesis- Rich people survived at a higher rate

train$Pclass= as.factor(train$Pclass)



ggplot(train, aes(x=Pclass, fill=factor(Survived)))+

  geom_bar(width=0.5)+

  xlab("PClass")+

  ylab("Total Count")+

  labs(fill="Survived")



#Get unique names in train and test data sets

length(unique(as.character(data.combined$Name)))



#Get the duplicate names and store them as a vector

dup.names=as.character(data.combined[which(duplicated(as.character(data.combined$Name))), "name"])



#Take a look at the records in the combined data set

data.combined[which(data.combined$Name %in% dup.names),]



library(stringr)



#Any correlation between Mr. and Mrs. variable

Miss=data.combined[which(str_detect(data.combined$Name, "Miss.")),]

Miss[1:5,]



#Name title correlates with age

Mrses=data.combined[which(str_detect(data.combined$Name, "Mrs.")),]

Mrses[1:5,]



#Check out males to see if pattern continues

males=data.combined[which(train$Sex=="male"),]

males[1:5,]
















