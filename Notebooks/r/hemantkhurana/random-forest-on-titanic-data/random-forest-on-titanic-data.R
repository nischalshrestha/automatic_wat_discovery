# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



# Load packages

library('ggplot2') # visualization

library('ggthemes') # visualization

library('scales') # visualization

library('dplyr') # data manipulation

library('mice') # imputation

library('randomForest') # classification algorithm

library(VIM) # for NA

library(dummies) #For String Index

# For modeling and predictions

library(caret)

library(glmnet)

library(ranger)

library(e1071)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")

traindata=read.csv("../input/train.csv",header = TRUE, stringsAsFactors  = FALSE,na.strings=c("NA","NaN", ""," ") )



testdata=read.csv("../input/test.csv",header = TRUE, stringsAsFactors  = FALSE,na.strings=c("NA","NaN", ""," "))

testdata$Survived=NA

fulldata=rbind(traindata,testdata)

#Check Null Data

aggr_plot <- aggr(fulldata, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(fulldata), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))

#Cabin has 80% data missing, Age has 20 % Missing and Embarked and Fare had less then 2%



#First Get Last name (family Name ) and Title details from name





fulldata$Title=sub(' ','',sapply(fulldata$Name,FUN=function(x){strsplit(x,split='[.,.]')[[1]][2]}))

fulldata$Lastname=sub(' ','',sapply(fulldata$Name,FUN=function(x){strsplit(x,split='[.,.]')[[1]][1]}))

#List of titles

unique(fulldata$Title)

table (fulldata$Sex,fulldata$Title)



# Map Title to smaller subset



map <- new.env(hash=T, parent=emptyenv())

key <- c( "Mr","Mrs"    ,"Lady"  ,"Mme" ,"Ms" ,"Miss" ,"Mlle","Master","Rev","Don","Sir","Dr","Col" ,"Capt","Major")  

keyvalue <- c("Mr","Mrs","Mrs","Mrs","Miss","Miss","Miss","Master","Rev","Mr","Sir","Dr","Col","Col","Col") 

for(keyv in key)

{

  map[[keyv]]=(keyvalue[key==keyv])

}



fulldata$Title=sapply(fulldata$Title,FUN=function(x){map[[x]]})

fulldata$Title=as.character(fulldata$Title)

fulldata$Title[c(760, 1306)] <- 'Mrs' 

fulldata$Title[c(823)] <- 'Mr'

fulldata$FamilySize=fulldata$SibSp+ fulldata$Parch+1



#ppl who were travlling on same ticket group 



fulldata$Groupticket=NA

fulldata <- (transform(fulldata, Groupticket = match(Ticket, unique(Ticket))))



#ppl sharing same family name



fulldata$Samesurname=NA

fulldata <- (transform(fulldata, Samesurname = match(Lastname, unique(Lastname))))

fulldata$Groupticket <- as.factor(fulldata$Groupticket)

fulldata$Samesurname <- as.factor(fulldata$Samesurname)





fulldata <- fulldata %>% 

  group_by(Groupticket) %>% 

  mutate(TicketGroupSize = n()) %>%

  ungroup()



fulldata <- fulldata %>% 

  group_by(Samesurname) %>% 

  mutate(FamilynameSize = n()) %>%

  ungroup()



 fulldata$Deck=sub(' ','',sapply(fulldata$Cabin,FUN=function(x){substring(x,0,1)}))





#Check Null Records



#Check Embarked



fulldata[(which(is.na(fulldata$Embarked))) ,c(1,3,6,10,20) ]





embark_fare <- fulldata %>%

  filter(PassengerId != 62 & PassengerId != 830)



# Use ggplot2 to visualize embarkment, passenger class, & median fare

ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +

  geom_boxplot() +

  geom_hline(aes(yintercept=80), 

             colour='red', linetype='dashed', lwd=2) +

  scale_y_continuous(labels=dollar_format()) +

  theme_few()

fulldata$Embarked[c(62, 830)] <- 'C'



#Check Fare



fulldata[(which(is.na(fulldata$Fare) | fulldata$Fare==0)) ,c(1,3,6,10,20) ]



#find value for Fare 0 also



fulldata$Fare[fulldata$Fare==0]=NA

str(fulldata)





#replace fare with Mean values     

         

fulldata[, names(fulldata) %in% c('Pclass','Deck','Embarked','Fare')]=  fulldata[, names(fulldata) %in% c('Pclass','Deck','Embarked','Fare')] %>%

  group_by(Pclass,Deck,Embarked) %>% 

  mutate_each(funs(replace(Fare, which(is.na(Fare)),

                           mean(Fare,na.rm = TRUE))))







# convert Values into factor



str(fulldata)





factor_vars <- c('Pclass','Sex','Embarked','Title','Survived')



fulldata[factor_vars] <- lapply(fulldata[factor_vars], function(x) as.factor(x))





#pridict Age based on Mice





mice_mod <- mice(fulldata[, names(fulldata) %in% c('Pclass','Sex','SibSp','Parch','Fare','Embarked','FamilySize','FamilynameSize','TicketGroupSize','Age')], method='rf') 

predicted_output <- complete(mice_mod)

#Check Age distribution 

par(mfrow=c(1,2))

hist(fulldata$Age, freq=F, main='Age: Original Data', 

     col='darkgreen', ylim=c(0,0.04))

hist(predicted_output$Age, freq=F, main='Age: MICE Output', 

     col='lightgreen', ylim=c(0,0.04))





fulldata$Age[is.na(fulldata$Age)] <- predicted_output$Age[is.na(fulldata$Age)]





#Predict Traveller Deck based on Fare 







predicted_deck <- train(

  Deck ~ Pclass +  Fare + Embarked ,

  tuneGrid = data.frame(mtry = c(2, 3, 7)),

  data = fulldata[!is.na(fulldata$Deck), ],

  method = "ranger",

  trControl = trainControl(

    method = "cv", number = 10,

    repeats = 10, verboseIter = TRUE),

  importance = 'impurity'

)



fulldata$Deck=as.factor(fulldata$Deck)

fulldata$Deck[is.na(fulldata$Deck)] <- predict(predicted_deck, fulldata[is.na(fulldata$Deck),])





ggplot(fulldata, aes(x = Deck, y = Fare, fill = factor(Pclass))) +

  geom_boxplot() +

  geom_hline(aes(yintercept=80), 

             colour='red', linetype='dashed', lwd=2) +

  scale_y_continuous(labels=dollar_format()) +

  theme_few()







#Check Female and kids





fulldata$isFemale=0



fulldata$isFemale[fulldata$Sex == 'female' & (fulldata$Parch > 0 | fulldata$TicketGroupSize>1 | fulldata$FamilynameSize>1) & fulldata$Age > 18 & fulldata$Title != 'Miss'] <- 1



fulldata$isKids=0



fulldata$isKids[fulldata$Age <=18  & (fulldata$Parch > 0 | fulldata$TicketGroupSize>1 | fulldata$FamilynameSize>1)  & fulldata$Title != 'Mrs'] <- 1



# Apply Predition



train <- fulldata[1:891,]

test <- fulldata[892:1309,]

myControl <- trainControl(

  method = "cv", 

  number = 10,

  repeats = 30, 

  verboseIter = TRUE

)

#Use RF Model

rf_model <- train(

  Survived ~ Age + Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize +Groupticket+ 

    Samesurname + TicketGroupSize+FamilynameSize+isFemale+Deck+isKids,

  tuneGrid = data.frame(mtry = c(2, 5, 8, 10, 15)),

  data = train, 

  method = "ranger", 

  trControl = myControl,

  importance = 'impurity'

)

test <- test %>%

  arrange(PassengerId)

my_prediction <- predict(rf_model, test)



output <- data.frame(PassengerID = test$PassengerId, Survived = my_prediction)



# Write the solution to a csv file 

write.csv(output, file = "titanic_summission.csv", row.names = FALSE)    

    

# Any results you write to the current directory are saved as output.