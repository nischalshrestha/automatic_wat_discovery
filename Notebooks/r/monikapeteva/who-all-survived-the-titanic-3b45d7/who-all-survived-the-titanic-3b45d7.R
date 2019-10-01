#Environment Setup and loading .csv data files into R data frames.

#Install and Load requied R packages and Libraries. Kaggle notebook is a pre-loaded env.

library(ggplot2)

library(readr) 

system("ls ../input")

#import test and training datasets/load .csv files in this case

train <- read.csv('../input/train.csv', stringsAsFactors = F)

test <- read.csv('../input/test.csv', stringsAsFactors = F)

#set a default value to Test$survived column as the value needs to be predicted part of our ML algoritham 

test$Survived <- NA

#Combine bth train and test dataset.

allData <- rbind(train,test)

#preview sample data along with header

head(allData, n= 3)
#Checking staructure of the dataset.

str(allData)
# Multiple plot function

#

# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)

# - cols:   Number of columns in layout

# - layout: A matrix specifying the layout. If present, 'cols' is ignored.

#

# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),

# then plot 1 will go in the upper left, 2 will go in the upper right, and

# 3 will go all the way across the bottom.

#

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {

  library(grid)



  # Make a list from the ... arguments and plotlist

  plots <- c(list(...), plotlist)



  numPlots = length(plots)



  # If layout is NULL, then use 'cols' to determine layout

  if (is.null(layout)) {

    # Make the panel

    # ncol: Number of columns of plots

    # nrow: Number of rows needed, calculated from # of cols

    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),

                    ncol = cols, nrow = ceiling(numPlots/cols))

  }



 if (numPlots==1) {

    print(plots[[1]])



  } else {

    # Set up the page

    grid.newpage()

    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))



    # Make each plot, in the correct location

    for (i in 1:numPlots) {

      # Get the i,j matrix positions of the regions that contain this subplot

      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))



      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,

                                      layout.pos.col = matchidx$col))

    }

  }

}



genderImpact <- data.frame(table(allData$Sex, allData$Survived))

names(genderImpact) <- c("Sex","Survived","Count")

sexVsSurvivedGraph <- ggplot(genderImpact, aes(x=Sex, y=Count, fill=Survived))

p1 <- sexVsSurvivedGraph + geom_bar(stat = "identity")



pClassImpact <- data.frame(table(allData$Pclass, allData$Survived))

names(pClassImpact) <- c("Pclass","Survived","Count")

pClassVsSurvivedGraph <- ggplot(pClassImpact, aes(x=Pclass, y=Count, fill=Survived))

p2 <- pClassVsSurvivedGraph + geom_bar(stat = "identity")



sibSpImpact <- data.frame(table(allData$SibSp+allData$Parch, allData$Survived))

names(sibSpImpact) <- c("FamilyMembers","Survived","Count")

sibSpVsSurvivedGraph <- ggplot(sibSpImpact, aes(x=FamilyMembers, y=Count, fill=Survived))

p3 <- sibSpVsSurvivedGraph + geom_bar(stat = "identity")



ageImpact <- data.frame(Age=allData$Age, Survived=allData$Survived)

p4 <- ggplot(ageImpact, aes(Age,fill = factor(Survived))) + geom_histogram()



embarkedImpact <- data.frame(table(allData$Survived, allData$Embarked))

names(embarkedImpact) <- c("Survived","Embarked","Count")

embarkVsSurvivedGraph <- ggplot(embarkedImpact, aes(x=Embarked, y=Count, fill=Survived))

p5 <- embarkVsSurvivedGraph + geom_bar(stat = "identity")



cabinImpact <- data.frame(table(allData$Survived, substr(allData$Cabin,0,1)))

names(cabinImpact) <- c("Survived","Cabin","Count")

cabinVsSurvivedGraph <- ggplot(cabinImpact, aes(x=Cabin, y=Count, fill=Survived))

p6 <- cabinVsSurvivedGraph + geom_bar(stat="identity")



multiplot(p1, p2, p3, p4, p5, p6, cols=2)
# Fill Age Column

age <- allData$Age

n = length(age)

    # replace missing value with a random sample from raw data

set.seed(123)

for(i in 1:n){

  if(is.na(age[i])){

    age[i] = sample(na.omit(allData$Age),1)

  }

}
# response variable

f.survived = train$Survived

t.survived = test$Survived

# feature

# 1. age

f.age = age[1:891]    # for training

t.age = age[892:1309]  # for testing

# 2. cabin

f.cabin = substr(allData$Cabin,0,1)[1:891]

t.cabin = substr(allData$Cabin,0,1)[892:1309]

# 3. family

family <- allData$SibSp + allData$Parch

f.family = family[1:891]

t.family = family[892:1309]

# 4. plcass

f.pclass = train$Pclass

t.pclass = test$Pclass

# 5. sex

f.sex = train$Sex

t.sex = test$Sex

# 6. embarked

f.embarked = allData$Embarked[1:891]

t.embarked = allData$Embarked[892:1309]



# construct training data frame

new_train = data.frame(survived = f.survived, age = f.age, sex = f.sex, 

       embarked = f.embarked ,family = f.family ,cabin =  f.cabin, pclass= f.pclass)



# random forest

library('randomForest')



set.seed(123)

fit_rf <- randomForest(factor(survived) ~ age +  sex + embarked + family 

                 + cabin + pclass,data = new_train)



    # predicted result of regression

rf.fitted = predict(fit_rf)

ans_rf = rep(NA,891)

for(i in 1:891){

  ans_rf[i] = as.integer(rf.fitted[[i]]) - 1

}

    # check result

mean(ans_rf == train$Survived)

table(ans_rf)
# Random Forest

a = sum(ans_rf ==1 & f.survived == 1)

b = sum(ans_rf ==1 & f.survived == 0)

c = sum(ans_rf ==0 & f.survived == 1)

d = sum(ans_rf ==0 & f.survived == 0)

data.frame(a,b,c,d)
# Test dataFrame

test_data_set <- data.frame(survived = t.survived, age = t.age, sex = t.sex, embarked = t.embarked, 

                            family = t.family, cabin =  t.cabin, pclass = t.pclass)

# make prediction



levels(test_data_set$survived) <- levels(new_train$survived)



levels(test_data_set$age) <- levels(new_train$age)



levels(test_data_set$sex) <- levels(new_train$sex)



levels(test_data_set$embarked) <- levels(new_train$embarked)



levels(test_data_set$family) <- levels(new_train$family)



levels(test_data_set$cabin) <- levels(new_train$cabin)



levels(test_data_set$pclass) <- levels(new_train$pclass)



rf_predict = predict(fit_rf,newdata = test_data_set )

ans_rf_predict = rep(NA,418)

for(i in 1:418){

  ans_rf_predict[i] = as.integer(rf_predict[[i]]) - 1

}

table(ans_rf_predict)
# create .csv file with predictions

endResult<-data.frame(PassengerId = test$PassengerId, Survived = ans_rf_predict)

write.csv(endResult,file = "SurvivingTheTitanicResult.csv",row.names = F)