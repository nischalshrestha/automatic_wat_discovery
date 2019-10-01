#Library:

library(randomForest)
#Data importing:

train<-read.csv('../input/train.csv')

test<-read.csv('../input/test.csv')
#Appending the datasets:

test$Survived<-NA

train$istest<-FALSE

test$istest<-TRUE

data<-rbind(train,test)
# Data structure:

# Variable | Definition

# survival |	Survival 	0 = No, 1 = Yes

# pclass |	Ticket class 	1 = 1st, 2 = 2nd, 3 = 3rd

# sex |	Sex 	

# Age |	Age in years 	

# sibsp 	# of siblings / spouses aboard the Titanic 	

# parch  # of parents / children aboard the Titanic 	

# ticket |	Ticket number 	

# fare |	Passenger fare 	

# cabin |	Cabin number 	

# embarked |	Port of Embarkation



#Summary and visualization:

str(data) #NAs: Age (263), Fare (1) and Embarked has an empty cell



sapply(data, function(y) sum(length(which(is.na(y)))))

#Source:http://stackoverflow.com/questions/24027605/determine-the-number-of-na-values-in-a-column



unique(data$Embarked)