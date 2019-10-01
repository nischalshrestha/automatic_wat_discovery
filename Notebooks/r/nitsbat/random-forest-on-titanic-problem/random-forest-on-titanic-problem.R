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



# as it is only seven yes we can convert it into factor

data.combined$SibSp <- as.factor(data.combined$SibSp)





data.combined$SibSp <- as.factor(data.combined$Parch)







# Let's make a family factor and add it to the frame (family size)



Sib <- c(training$SibSp,test$SibSp)

PAr <- c(training$Parch,test$Parch)



data.combined$family.size <- as.factor(Sib+PAr+1)





#experimenting on Ticket

#data.combined$Ticket[1:5]

tick <- ifelse(data.combined$Ticket == "","",substr(data.combined$Ticket,1,1))



#Adding a column in dat.combined according to ticket letter

data.combined$ticket.letter <- as.factor(tick)

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

library(stringr)

cht <- ifelse(str_detect(data.combined$Cabin," "),"MC","SC") #Mc for multiple cabins

#length(which(cht == "MC")) 



# Making it a variable of dataframe

data.combined$MulitpleCabin <- as.factor(cht)

data.combined[1:2,]



    

library(randomForest)



# creating the default parameters Pclass and title

rf.train <- data.combined[1:891,c("Pclass","title")]

#rf.train[1:50,]

rf.labels <- as.factor(training$Survived)



set.seed(1234)

# randomForest function takes x and y axis with importance as TRUE and number of decission tree as ntree

rf.1 <- randomForest(x=rf.train,y=rf.labels,importance = TRUE,ntree = 1000)

rf.1

varImpPlot(rf.1)





#creating random forest by adding sibsp

rf.train <- data.combined[1:891,c("Pclass","title","SibSp")]

set.seed(1234)

# randomForest function takes x and y axis with importance as TRUE and number of decission tree as ntree

rf.2 <- randomForest(x=rf.train,y=rf.labels,importance = TRUE,ntree = 1000)

#rf.2

varImpPlot(rf.2)





#creating random forest by adding 

rf.train <- data.combined[1:891,c("Pclass","title","Parch")]

set.seed(1234)

# randomForest function takes x and y axis with importance as TRUE and number of decission tree as ntree

rf.3 <- randomForest(x=rf.train,y=rf.labels,importance = TRUE,ntree = 1000)

#rf.3

varImpPlot(rf.3)



#creating random forest by adding sibsp and Parch

rf.train <- data.combined[1:891,c("Pclass","title","SibSp","Parch")]

set.seed(1234)

# randomForest function takes x and y axis with importance as TRUE and number of decission tree as ntree

rf.3 <- randomForest(x=rf.train,y=rf.labels,importance = TRUE,ntree = 1000)

#rf.3

varImpPlot(rf.3)



# By adding family member

rf.train <- data.combined[1:891,c("Pclass","title","family.size")]

set.seed(1234)

# randomForest function takes x and y axis with importance as TRUE and number of decission tree as ntree

rf.4 <- randomForest(x=rf.train,y=rf.labels,importance = TRUE,ntree = 1000)

#rf.4

varImpPlot(rf.3)



# All

rf.train <- data.combined[1:891,c("Pclass","title","family.size","SibSp","Parch")]

set.seed(1234)

# randomForest function takes x and y axis with importance as TRUE and number of decission tree as ntree

rf.5 <- randomForest(x=rf.train,y=rf.labels,importance = TRUE,ntree = 1000)

#rf.5

varImpPlot(rf.5)
