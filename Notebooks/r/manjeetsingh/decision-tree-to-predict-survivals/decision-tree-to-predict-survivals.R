# Load the important libraries

library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(dplyr)

library(rpart)

library("rpart")

library("rpart.plot")
# load the necessary files

train <- read.csv('../input/train.csv', stringsAsFactors = F)

test  <- read.csv('../input/test.csv', stringsAsFactors = F)

dim(train)

dim(test)
#Tets file have 11 columns while Train got 12. We need to merge them to get a complete dataset, so lest add a column named as Survival to Test



test$Survived <- NA

full_data <- rbind(train, test)

dim(full_data)

colSums(is.na(full_data))




# Any results you write to the current directory are saved as output.



table(full_data$Embarked)

full_data$Embarked[full_data$Embarked==""]="S"

table(full_data$Embarked)

full_data$Title <- gsub('(.*, )|(\\..*)', '', full_data$Name)

table(full_data$Sex, full_data$Title)

table(full_data$Sex, full_data$Title)

apply(full_data,2, function(x) length(unique(x)))

 

cols=c("Survived","Pclass","Sex","Embarked")

for (i in cols){

  full_data[,i]=as.factor(full_data[,i])

}



ggplot(full_data[1:891,],aes(x = Pclass,fill=factor(Survived))) +

  geom_bar() +

  ggtitle("Pclass v/s Survival Rate")+

  xlab("Pclass") +

  ylab("Total Count") +

  labs(fill = "Survived") 



full_data$Fsize <- full_data$SibSp + full_data$Parch + 1

full_data$Family <- paste(full_data$Surname, full_data$Fsize, sep='_')



ggplot(full_data[1:891,], aes(x = Fsize, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') +

  scale_x_continuous(breaks=c(1:11)) +

  labs(x = 'Family Size') 



full_data[c(62, 830), 'Embarked']



embark_fare <- full_data %>%

  filter(PassengerId != 62 & PassengerId != 830)



full_data$Embarked[c(62, 830)] <- 'C'



full_data$Fare[1044] <- median(full_data[full_data$Pclass == '3' & full_data$Embarked == 'S', ]$Fare, na.rm = TRUE)

sum(is.na(full_data$Age))

ggplot(full_data[1:891,], aes(x = Sex, fill = Survived)) +

  geom_bar() +

  facet_wrap(~Pclass) + 

  ggtitle("3D view of sex, pclass, and survival") +

  xlab("Sex") +

  ylab("Total Count") +

  labs(fill = "Survived")



head(full_data$Name)

full_data$FsizeD[full_data$Fsize == 1] <- 'singleton'

full_data$FsizeD[full_data$Fsize < 5 & full_data$Fsize > 1] <- 'small'

full_data$FsizeD[full_data$Fsize > 4] <- 'large'



factor_vars <- c('PassengerId','Pclass','Sex','Embarked',

                 'Title','Surname','Family','FsizeD')

      

full_data$Embarked[c(62,830)] = "S"

full_data$Embarked <- factor(full_data$Embarked)



full_data$Fare[1044] <- median(full_data$Fare, na.rm=TRUE)



full_data$family_size <- full_data$SibSp + full_data$Parch + 1





predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + family_size,

                       data=full_data[!is.na(full_data$Age),], method="anova")

full_data$Age[is.na(full_data$Age)] <- predict(predicted_age, full_data[is.na(full_data$Age),])

train_new <- full_data[1:891,]

test_new <- full_data[892:1309,]

test_new$Survived <- NULL



train_new$Cabin <- substr(train_new$Cabin,1,1)

train_new$Cabin[train_new$Cabin == ""] <- "H"

train_new$Cabin[train_new$Cabin == "T"] <- "H"



test_new$Cabin <- substr(test_new$Cabin,1,1)

test_new$Cabin[test_new$Cabin == ""] <- "H"



train_new$Cabin <- factor(train_new$Cabin)

test_new$Cabin <- factor(test_new$Cabin)



str(train_new)

str(test_new)

my_tree <- rpart(Survived ~ Age + Sex + Pclass  + family_size, data = train_new, method = "class", control=rpart.control(cp=0.0001))

summary(my_tree)

prp(my_tree, type = 4, extra = 100)

my_prediction <- predict(my_tree, test_new, type = "class")

head(my_prediction)

vector_passengerid <- test_new$PassengerId

my_solution <- data.frame(PassengerId = vector_passengerid, Survived = my_prediction)

head(my_solution)

write.csv(my_solution, file = "my_solution.csv",row.names=FALSE)






