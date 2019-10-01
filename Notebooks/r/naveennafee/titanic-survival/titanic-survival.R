# This is the my first kernal using R, So any feedback are most welcome.

# Treating the missing value in the dataset,Extracting the new feature and Visualizing the data

# are done in this post.

# I'm going to use Random Forest and Decision Tree model to predict the Titanic Dataset.

# Overview of some of the features available in the dataset



#  1) Survival - 0 = No, 1 = Yes

#  2) PClass   - Passenger Class

#  3) embarked - Port of Embarkation

#  4) sibsp    - Sibling, Spouse

#  5) parch    - Parent, Child
# Loading the required library for our model

library('tidyverse')

library('ggthemes')

library('caret')

library('e1071')

library('rpart')

library('rpart.plot')

library('randomForest')
# Reading and Saving the Training and Testing CSV file

training<-read.csv('../input/train.csv',stringsAsFactors = FALSE)

testing<-read.csv('../input/test.csv',stringsAsFactors = FALSE)

# Inserting the Survived feature in the testing dataset with NA

testing$Survived<-NA

# Combining both the dataset into single dataset

full<-rbind(training,testing)

# Visualize the available features and data type of it.

str(full)
# Analysing the Missing Values in the dataset

colSums(is.na(full))

colSums(full=='')
# As Cabin is having most of the missing values, We are going to remove it.

full$Cabin<-NULL

str(full)
# Missing value treatment for Embarked

table(full$Embarked)

# As 'S' is the mostly used in the dataset, We are going to provide the same for the missing values

full[full$Embarked=="",]$Embarked<-"S"

table(full$Embarked)

# Data based on the Embarked and Survived

table(full$Embarked,full$Survived)
# There is one missing value for Fare in row 1044, As the passenger is from PClass 3, 

# we are going to provide the mean fare of PClass 3

full$Fare[1044]<-mean(full[full$Pclass=='3',]$Fare,na.rm=T)

# We are going to remove the Title from the Name field.

head(full$Name)

# We are using gsub to replace the following pattern into empty, So that we will get the title

full$title<-gsub('(.*,)|(\\..*)','',full$Name)

# List of availble title

unique(full$title)
# Now We are going the trim the whitespace available in the title.

full$title<-trimws(full$title)

# As there are many unknown title is available, we are going to combine all the unknown into Rare title

rare_title<-c('Dona','Lady','the Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer')

full[full$title=='Mlle',]$title<-"Miss"

full[full$title=='Ms',]$title<-"Miss"

full[full$title=='Mme',]$title<-"Mr"

full[full$title %in% rare_title,]$title<-"Rare"

# List of available title after the data treatment

table(full$title)
# Analysing the feature with the number of unique data

factor_check<-sapply(full,function(x) (length(unique(x))))

factor_check
# As Survived,Sex,Embarked,Pclass are the suitable feature to be converted into factor

factor_var<-c('Survived','Sex','Embarked','Pclass')

for (i in factor_var) {full[,i]<-as.factor(full[,i])}

# Available features with data type

str(full)
# Visualizing the chart for Embarked and Survived

ggplot(full[1:891,],aes(x=Embarked,fill=Survived))+geom_bar(position="dodge")+ggtitle("Port of Embarkation vs Survived")+ylab("Survived")
# Visualizing the chart for Title and Survived

ggplot(full[1:891,],aes(x=title,fill=Survived))+geom_bar()+ggtitle("Title of the Person vs Survived")+ylab("Survived")
# Visualizing the chart for Sex and Survived, From this we could see Women are most likely survived

ggplot(full[1:891,],aes(x=Sex,fill=Survived))+geom_bar(position="dodge")+ggtitle("Sex vs Survived")+ylab("Survived")
# Creating the new feature Family count with the help of SibSp and Parch

full$Family_count<-full$SibSp+full$Parch+1

# List of Family count in our titanic dataset

table(full$Family_count)
# Visualizing the data based on Family Count and Survival

ggplot(full[1:891,],aes(x=Family_count,fill=Survived))+geom_bar(position='dodge')+ggtitle("Family Count vs Survived")+ylab("Survived")+

  coord_flip()+scale_x_reverse(breaks=c(1:11))+theme_light()
# Creating the new field based on Family_Count Size

full$Family_size_ratio[full$Family_count<=2]<-"Small"

full$Family_size_ratio[full$Family_count>=3 & full$Family_count<=5]<-"Medium"

full$Family_size_ratio[full$Family_count>=6]<-"Big"

ggplot(full[1:891,],aes(x=Family_size_ratio,fill=Survived))+geom_bar(position='dodge')+ggtitle("Family Size Ratio vs Survived")+ylab("Survived")
# First 6 rows of data in our dataset

head(full)
# Imputing the mean value for the missing values in the Age feature with the help of MISC library Impute function

full$Age<-Hmisc::impute(full$Age,mean)

# Age vs Survived Box Plot

ggplot(full[1:891,],aes(x=Survived,y=Age))+geom_boxplot(color=c("blue","red"))+theme_few()+ggtitle("Age vs Survived")
# Bar Plot for PClass vs Survived

ggplot(full[1:891,],aes(x=Pclass,fill=Survived))+geom_bar(position="dodge")+ggtitle("PClass vs Survived")
# Structure of Full dataset

str(full)
# Converting the new features into factor as well

full$Family_size_ratio<-as.factor(full$Family_size_ratio)

full$title<-as.factor(full$title)

# Splitting the model with the features we require to train the dataset

train_model<-full[1:891,c("Survived","Age","Sex","Family_count","Family_size_ratio","Fare","title")]

test_model<-full[892:1309,c("Survived","Age","Sex","Family_count","Family_size_ratio","Fare","title")]  
# RANDOM FOREST Model Implementation

rf_model<-randomForest(Survived~.,data=train_model,importance=T)

rf_model
# Predicting the model

predicted<-predict(rf_model,train_model)

# Confusion Matrix to compare the 'Predicted value vs Actual Value' accuracy

confusionMatrix(predicted,train_model$Survived)
# Plotting the RF_MODEL

plot(rf_model)

legend('topright',colnames(rf_model$err.rate),col=1:3,fill=1:3)
# Finding which variable is important for model building

varImpPlot(rf_model, main = "RANDOM FOREST MODEL")
# Implementing based on Decision Tree, Method is 'class' for classification problem

rp_model<-rpart(Survived~.,train_model,method='class')

summary(rp_model)
# Decision tree of our model 

rpart.plot(rp_model,tweak=0.8)
rp_predicted<-predict(rp_model,train_model,type = 'class')

confusionMatrix(rp_predicted,train_model$Survived)

# As Random Forest is providing higher accuracy than Decision Tree, We are going to use that for Test Data
submission <- data.frame(PassengerId = testing$PassengerId)

submission$Survived <- predict(rf_model,test_model)

# Submitting the predicted value of the titanic data set

write.csv(submission, file = "random_forest_r_submission.csv", row.names=FALSE)