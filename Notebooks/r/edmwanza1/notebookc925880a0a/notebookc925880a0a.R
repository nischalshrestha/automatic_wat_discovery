

library(readr)

library(data.table)

library(ggplot2)

library(ggthemes)

library(scales)

library(Amelia)

library(tabplot)

library(dplyr)

library(mice)

library(randomForest)

library(VIM)

library(stringr)

library(caret)



train <- read.csv('../input/train.csv', stringsAsFactors = F)

test  <- read.csv('../input/test.csv', stringsAsFactors = F)
my_df <- bind_rows(train, test)#format and wrangle data together at once

str(my_df)

summary(my_df)

glimpse(my_df)

head(my_df, n = 10)
sapply(my_df, class)

#Convert "sex"from char->factor

my_df$Sex <- as.factor(my_df$Sex)

str(my_df)
#If a variable is "numeric" check for numerical special values otherwise just for NAs

is_special <- function(x){

  if(is.numeric(x)) !is.finite(x) else is.na(x)

}

sapply(my_df, is_special)
Missing_d <- function(x){sum(is.na(x))/length(x)*100} #USD to calculate % of missing data

apply(my_df, 2, Missing_d) #checking missing data variable-wise

md.pattern(my_df)
missmap(my_df, main = "Observed Vs Missing Data")

aggr_plot <- aggr(my_df, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 

                  labels=names(data), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))


which(is.na(my_df$Embarked))

#observations 62 and 830 are missing embarkment. Where did these voyagers come from?

#Find out how much they paid (fare)

embark_miss <- subset(my_df, PassengerId == 62 | PassengerId== 830)

#They both paid $80, belonged to class 1 Lets impute based on similar passengers

embark_fare <- my_df %>%

  filter(PassengerId != 62 & PassengerId != 830)

#As inspired by some Kagglers

ggplot(data = embark_fare, aes(x = Embarked, y= Fare, fill = factor(Pclass))) +

  geom_boxplot()+

  geom_hline(aes(yintercept = 80),

             colour = 'blue', linetype = 'dashed', lwd = 2)+

  scale_y_continuous(labels = dollar_format())+

  theme_few()

#As observed, $80-dollar dash line coincides with the median for class 1. We can infer that the 

#obervations 62 and 830 embarked from "C"

#CORRECTION:replace NA with "C"

my_df$Embarked[c(62, 830)] <- "C"

sum(is.na(my_df$Embarked))



which(is.na(my_df$Fare))

#voyager 1044 has missing fare

fare_miss <- subset(my_df, PassengerId == 1044)

#This passenger embarked the unsinkable at port "S" and was Class "3", use same approach as embark

fare_fix <- my_df%>%

  filter(PassengerId != 1044)

#whats the median fare for class "3" passengers who emabarked at "S"?

fare_fix%>%

  filter(Pclass == 3 & Embarked == 'S')%>%

  summarise(avg_fare = round(median(Fare),digits = 2))

#median amount paid is ~ 8.05::Its safe to CORRECT the NA value with the median value

my_df$Fare[1044] <- 8.05

sum(is.na(my_df$Fare))
mice_corr <- mice(my_df[c("Sex","Age")],m=5,maxit=50,meth='pmm',seed=500)

summary(mice_corr)

mice_corr$imp$Age #quick check of imputed data

mice_corr <- complete(mice_corr,1) #save the newly imputed data choosing the first of 

#five data sets (m=5)!!



#Check before & after distributions of the AGE variable

par(mfrow=c(1,2))

hist(my_df$Age, freq=F, main='Age: Original Data', 

     col='darkgreen', ylim=c(0,0.04))

hist(mice_corr$Age, freq=F, main='Age: MICE imputation', 

     col='lightgreen', ylim=c(0,0.04))

#Looks great!! Visual inspection shows NO change in distribution! 

#Replace original age values with imputed values
my_df$Age <- mice_corr$Age

sum(is.na(my_df$Age))
my_df$Cabin <- NULL #remove Cabin

#Before I decopose AGE, lets do some visual analyses related to the age variable

my_hist <- hist(my_df$Age, breaks = 100, plot = F )

my_col  <- ifelse(my_hist$breaks < 18, rgb(0.2,0.8,0.5,0.5), ifelse(my_hist$breaks >= 65, "purple"

                                                                     , rgb(0.2,0.2,0.2,0.2)))

plot(my_hist, col = my_col, border =F, xlab = "color by age group",main ="", xlim = c(0,100))
#I will define age group as child < 18, 18>adult<65 and elderly >= 65. As can be seen there was more #adults than the 2 age groups



posn_jitter <- position_jitter(0.5, 0)



ggplot(my_df[1:891,], aes(x = factor(Pclass), y = Age, col = factor(Sex))) + 

  geom_jitter(size = 2, alpha = 0.5, position = posn_jitter) + 

  facet_grid(. ~ Survived)

#It didnt help being a middle aged male in class 3!! On the contrary, majority females in class 1

#survived
#Decompose age 

my_df$Group[my_df$Age < 18] <- 'Child'

my_df$Group[my_df$Age >= 18 & my_df$Age < 65] <- 'Adult'

my_df$Group[my_df$Age >= 65] <- 'Elderly'



#Survival per age group

table(my_df$Group, my_df$Survived)
table(my_df$Survived, my_df$Pclass,my_df$Group)

my_df$Title <- stringr::str_replace_all(my_df$Name,'(.*,)|(\\..*)', '')

my_df$Name  <- stringr::str_replace_all(my_df$Name, "(.*),.*\\. (.*)", "\\2 \\1")
factors <- c('Embarked', 'Group', 'Title', 'Pclass')#left survived out...

my_df[factors] <- lapply(my_df[factors], as.factor)

sapply(my_df, class)
my_df$Ticket <- NULL

my_df$Name   <- NULL

train <- my_df[1:891,]

test  <- my_df[892:1309,]

#remove Survive from test set which is all NULL resulted from merging the 2 sets

test$Survived <- NULL

passenger_id <- test[,1]#pull passenger_id for submission

test <- test[,-1]#remove passenger_id from test set

train <- train[,-1]#remove passenger_id from train set its irrevalent

train$Survived <- ifelse(train$Survived == 1, 'Yes', 'No')

train$Survived <- as.factor(train$Survived)
set.seed(10)

shuffled <- sample(nrow(train))

train    <- train[shuffled,]
set.seed(34)

control <- trainControl(method="cv", number=10, repeats=3)

model <- train(Survived~., data = train,trControl=control)

importance <- varImp(model, scale=FALSE)

print(importance)

plot(importance, main = "feature importance")
set.seed(5048)

myControl <- trainControl(method = "repeatedcv", number = 10,repeats = 3, classProbs = T, 

                          verboseIter = F, summaryFunction = twoClassSummary)



metric <- 'ROC'
#1.RF

set.seed(5048)

RF_model <- train(Survived ~.,tuneLength = 3, data = train, method = "rf",

                  metric = metric,trControl = myControl)

print(RF_model)
#2.rpart

set.seed(5048)

RP_model <- train(Survived ~.,tuneLength = 3, data = train, method = "rpart",

                  metric = metric,trControl = myControl)

print(RP_model)
#3.logit

set.seed(5048)

Grid<- expand.grid(decay=c(0.0001,0.00001,0.000001))

LR_model <- train(Survived~., data = train, method = 'multinom',metric = metric,trControl=myControl,

                  tuneGrid=Grid, MaxNWts=2000)



print(LR_model)

#4.glmnet

set.seed(5048)

GN_model <- train(Survived~.,train,tuneGrid = expand.grid(alpha = 0:1,lambda = seq(0.0001, 1, length = 20)),

                  method = "glmnet", metric = metric,trControl = myControl)



print(GN_model)
#5.SVM

set.seed(5048)

SVM_model <- train(Survived~.,data = train, method = "svmRadial",metric = metric,

                   trControl = myControl)



print(SVM_model)

#6.naive bayes

set.seed(5048)

NB_model <- train(Survived~., data = train, method="nb", metric=metric, 

                  trControl= myControl)



print(NB_model)
#8.Shrinkage Discrimant Analysis

set.seed(5048)

SDA_model <- train(Survived~., data = train, method="sda", metric=metric, 

                   trControl=myControl)

print(SDA_model)
#9.glm

set.seed(5048)

GLM_model <- train(Survived~., data = train, method="glm", metric=metric, 

                   trControl=myControl)

print(GLM_model)
#Using ROC as measure

#create resample object assigning models to their descriptive names

resample_results <- resamples(list(RF=RF_model,CART=RP_model,GLMNET=GN_model,

                                   SDA=SDA_model,SVM=SVM_model, GLM = GLM_model, LR=LR_model,

                                   NaiveBayes=NB_model))

#Check summary on all metrics

summary(resample_results, metric = c('ROC','Sens','Spec'))

#Analyze MODEL performance 

bwplot(resample_results, metric = metric)

bwplot(resample_results, metric = c('ROC','Sens','Spec'))

densityplot(resample_results, metric = metric, auto.key = list(columns = 3))
prediction <- data.frame(passenger_id,predict(RF_model,test,type = "raw"))

#write.csv(prediction, "RF_submission.csv", row.names = FALSE)
