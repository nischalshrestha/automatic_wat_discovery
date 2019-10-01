# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(stringr)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")





training <- read.csv("../input/train.csv")

test <- read.csv("../input/test.csv")

test$Survived <- rep("None")

data.combined <- rbind(training,test)

#data.combined[892,]

#table(training$Survived)

#table(training$Pclass)

training$Pclass <- as.factor(training$Pclass)

p <- ggplot(training, aes(x=Pclass,fill= factor(Survived)))

p <- p + geom_histogram(stat="count",width=0.5)+ xlab("Pclass")+ ylab("Total Count")+ labs(fill = "Survived")



dup_names <- as.character(data.combined[which(duplicated(data.combined$Name)),"Name"])

dup_names <- data.combined[which(data.combined$Name %in% dup_names),]

master <- data.combined[which(str_detect(data.combined$Name,"Master.")),]

misis <- data.combined[which(str_detect(data.combined$Name,"Miss.")),]

#misis[1:5,]

mises <- data.combined[which(str_detect(data.combined$Name,"Mrs")),]

#mises[1:5,]

mister <- data.combined[which(data.combined$Sex == "male"),]

#mister[1:5,]

extract_title <- function(name){

  name <- as.character(name)

  

  if(length(grep("Miss.",name)) > 0){

    return("Miss.")

  }

  else if(length(grep("Master.",name)) > 0){

    return("Master.")

  }

  else if(length(grep("Mrs.",name)) > 0){

    return("Mrs.")

  }

  else if(length(grep("Mr.",name)) > 0){

    return("Mr.")

  }

  else{

    return("Other")

  }

}



title <- NULL

for(i in 1:nrow(data.combined))

{

  title <- c(title,extract_title(data.combined[i,"Name"]))

}



data.combined$title <- as.factor(title)





# ggplot() for the newly data

ggplot(data.combined[1:891,],aes(x=title,fill = Survived)) + geom_bar(width=0.5) + facet_wrap(~ Pclass)



#table(data.combined$Sex)



#visualise the data through graph

ggplot(data.combined[1:891,],aes(x=Sex,fill=Survived)) +

  geom_bar(width=0.5) +

  facet_wrap(~ Pclass) +

  ggtitle("Pclass") +

  xlab("Sex") + 

  ylab("Total Count") +

  labs(fill = "Survived")



#Distribution on age criteria

#summary(data.combined$Age)

#summary(data.combined[1:891,"Age"])

 ggplot(data.combined[1:891,],aes(x=Age,fill=Survived)) +

  geom_histogram(binwidth=10) +

  facet_wrap(~ Sex + Pclass) +

  ggtitle("Pclass") +

  xlab("Age") + 

  ylab("Total Count") +

  labs(fill = "Survived")



# knowledge about master 

mast <- data.combined[which(data.combined$title == "Master."),]

summary(mast$Age)



# knowledge about Miss as it is complicated

miss <- data.combined[which(data.combined$title == "Miss."),]

summary(miss$Age)



#visualisation on miss

ggplot(miss[miss$Survived != "none",],aes(x=Age,fill=Survived)) +

  geom_histogram(binwidth=5) +

  facet_wrap(~ Pclass) +

  ggtitle("Age for 'mis' by Pclass") +

  xlab("Age") + 

  ylab("Total Count") +

  labs(fill = "Survived")



#More condition on misses

miss.alone <- miss[which(miss$SibSp==0 &miss$Parch == 0),]

#summary(miss.alone$Age)



#female equivalent to master boys

#length(which(miss.alone$Age <= 14.5))



#looking at sibsp

#summary(data.combined$SibSp)

 

#can we treat it as a factor

#length(unique(data.combined$SibSp))



# as it is only seven yes we can convert it into factor

data.combined$SibSp <- as.factor(data.combined$SibSp)



#visualising the survival rates by SibSp,pclass and title

ggplot(data.combined[1:891,],aes(x=SibSp,fill=Survived)) +

  geom_histogram(stat = "count",binwidth=1) +

  facet_wrap(~ title + Pclass) +

  ggtitle("Title & Pclass") +

  xlab("Sibsp") + 

  ylab("Total Count") +

  labs(fill = "Survived")



#Similarly we do for Parch



data.combined$SibSp <- as.factor(data.combined$Parch)



#visualising the survival rates by SibSp,pclass and title

ggplot(data.combined[1:891,],aes(x=Parch,fill=Survived)) +

  geom_histogram(stat = "count",binwidth=1) +

  facet_wrap(~ title + Pclass) +

  ggtitle("Title & Pclass") +

  xlab("Parch") + 

  ylab("Total Count") +

  labs(fill = "Survived")





# Let's make a family factor and add it to the frame (family size)



Sib <- c(train$SibSp,test$SibSp)

PAr <- c(train$Parch,test$Parch)



data.combined$family.size <- as.factor(Sib+PAr+1)



#visualising the survival rates by familySize,pclass and title

ggplot(data.combined[1:891,],aes(x=family.size,fill=Survived)) +

  geom_histogram(stat = "count",binwidth=1) +

  facet_wrap(~ title + Pclass) +

  ggtitle("Title & Pclass") +

  xlab("Family size") + 

  ylab("Total Count") +

  labs(fill = "Survived")





#experimenting on Ticket

#data.combined$Ticket[1:5]

tick <- ifelse(data.combined$Ticket == "","",substr(data.combined$Ticket,1,1))



#Adding a column in dat.combined according to ticket letter

data.combined$ticket.letter <- as.factor(tick)



ggplot(data.combined[1:891,],aes(x=ticket.letter,fill=Survived)) +

  geom_histogram(stat = "count",binwidth=1) +

  facet_wrap(~ title + Pclass) +

  ggtitle("Title & Pclass") +

  xlab("Ticket") + 

  ylab("Total Count") +

  labs(fill = "Survived")



#visualisation on Fares



ggplot(data.combined,aes(x=Fare)) +

  geom_histogram(binwidth=5) +

  ggtitle("Fare") +

  xlab("fare") + 

  ylab("Total Count") +

  ylim(0,200)



#visualisation based on Survival and Fares



ggplot(data.combined[1:891,],aes(x=Fare,fill=Survived)) +

  geom_histogram(binwidth=5) +

  facet_wrap(~ title + Pclass) +

  ggtitle("Title & Pclass") +

  xlab("Ticket") + 

  ylab("Total Count") +

  labs(fill = "Survived") +

  ylim(0,50)



# visualisation on Cabins

#str(data.combined$Cabin)



#cabin is not really a factor , hence converted into strings

data.combined$Cabin <- as.character(data.combined$Cabin)

#data.combined$Cabin[1:100]



#Lots of no empty cabins

#Making empty cabins as "U" Unknown

data.combined[which(data.combined$Cabin==""),"Cabin"] <- "U"

#data.combined$Cabin[1:100]



#look on first letter of cabin and making it a factor

fact <- as.factor(substr(data.combined$Cabin,1,1))

#str(fact)

#levels(fact)



#adding the variable to dataframe as cabinLeter



data.combined$cabinLetter <- fact



#High level plot



ggplot(data.combined[1:891,],aes(x=cabinLetter,fill=Survived)) +

  geom_histogram(stat="count",binwidth=5) +

  ggtitle("Cabin Letter Factors") +

  xlab("Cabin_Letter") + 

  ylab("Total Count") +

  labs(fill = "Survived") 



# Based on pclass



ggplot(data.combined[1:891,],aes(x=cabinLetter,fill=Survived)) +

  geom_histogram(stat="count",binwidth=5) +

  facet_wrap(~ Pclass) +

  ggtitle("Pclass") +

  xlab("Cabin_Letter") + 

  ylab("Total Count") +

  labs(fill = "Survived") +

  ylim(0,750)



# Based on pclass and Title

ggplot(data.combined[1:891,],aes(x=cabinLetter,fill=Survived)) +

  geom_histogram(stat="count",binwidth=5) +

  facet_wrap(~ title + Pclass) +

  ggtitle("Pclass") +

  xlab("Cabin_Letter") + 

  ylab("Total Count") +

  labs(fill = "Survived") +

  ylim(0,750)



# Visualisation on multiple cabins

library(stringr)

cht <- ifelse(str_detect(data.combined$Cabin," "),"MC","SC") #Mc for multiple cabins

#length(which(cht == "MC")) 



# Making it a variable of dataframe

data.combined$MulitpleCabin <- as.factor(cht)

data.combined[1:2,]



#Visualisation

ggplot(data.combined[1:891,],aes(x=MulitpleCabin,fill=Survived)) +

  geom_histogram(stat="count",binwidth=5) +

  facet_wrap(~ title + Pclass) +

  ggtitle("Pclass and Title") +

  xlab("Multiple Cabin") + 

  ylab("Total Count") +

  labs(fill = "Survived") +

  ylim(0,750)



# visualisation on places onboard

#data.combined$Embarked[1:10]



ggplot(data.combined[1:891,],aes(x=Embarked,fill=Survived)) +

  geom_histogram(stat="count",binwidth=5) +

  ggtitle("Survival") +

  xlab("Boarding_Place") + 

  ylab("Total Count") +

  labs(fill = "Survived") +

  ylim(0,750)





ggplot(data.combined[1:891,],aes(x=Embarked,fill=Survived)) +

  geom_histogram(stat="count",binwidth=5) +

  facet_wrap(~ Pclass + title) +

  ggtitle("Pclass and Title") +

  xlab("Boarding_Place") + 

  ylab("Total Count") +

  labs(fill = "Survived") +

  ylim(0,750)



   

    

#