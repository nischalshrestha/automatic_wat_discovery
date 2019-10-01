library(dplyr)

library(VIM)

library(mice)

library(funModeling)

library(caret)

library(lime)



train <- read.csv("../input/train.csv", stringsAsFactors = FALSE, na.strings=c("","NA"))

test <- read.csv("../input/test.csv", stringsAsFactors = FALSE, na.strings=c("","NA"))

df_combined <- bind_rows(train, test) #http://dplyr.tidyverse.org/reference/bind.html



str(df_combined)
aggr_plot <- aggr(df_combined, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 

                  labels=names(df_combined), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
miss_cols <- colnames(df_combined)[colSums(is.na(df_combined)) > 0]

miss_cols
subset(df_combined, is.na(Fare) == TRUE)
df_combined$Fare[1044] <- median(df_combined[df_combined$Pclass == '3' & df_combined$Embarked == 'S', ]$Fare, na.rm = TRUE)
nrow(subset(df_combined, is.na(Age) == TRUE))
set.seed(42)



mice_mod <- mice(df_combined[, !names(df_combined) %in% 

                      c('PassengerId','Name','Ticket','Cabin','Survived')], method='rf') 
mice_output <- complete(mice_mod)



df_combined$Age <- mice_output$Age
subset(df_combined, is.na(Embarked) == TRUE)
table(subset(df_combined, Fare >= 80, select = c("Embarked")))
df_combined$Embarked[c(62, 830)] <- 'C'
nrow(subset(df_combined, is.na(Cabin) == TRUE))
miss_cols <- colnames(df_combined)[colSums(is.na(df_combined)) > 0]

miss_cols
nums <- sapply(df_combined, is.numeric)

cats <- sapply(df_combined, is.character)

names(df_combined[,nums])

names(df_combined[,cats])
cols <- c("PassengerId", "Survived", "Pclass", "Name", "Sex", "Ticket", "Cabin", "Embarked")

df_combined[cols] <- lapply(df_combined[cols], factor)
lapply(df_combined[cols], nlevels)
names(df_combined[,-which(names(df_combined) %in% cols)])
var_list <- c("Pclass","Sex","Embarked","Age","SibSp","Parch","Fare")

plots <- function(var_list) {

    cross_plot(df_combined[1:891,], str_input=var_list, str_target="Survived")

}

lapply(var_list, plots)
df_combined$Title <- gsub('(.*, )|(\\..*)', '', df_combined$Name)

tab <- prop.table(table(df_combined$Title))

df <- data.frame(Title=names(tab), proportion=as.numeric(tab))

df <- df[order(-df$proportion),]

df
other_title <- c('Dr','Rev','Col','Major','Capt','Don','Dona','Jonkheer','Lady','Sir','the Countess')

df_combined$Title[df_combined$Title == 'Mlle'] <- 'Miss' #http://www.dictionary.com/browse/mademoiselle?s=t

df_combined$Title[df_combined$Title == 'Ms'] <- 'Miss' 

df_combined$Title[df_combined$Title == 'Mme'] <- 'Mrs' #http://www.dictionary.com/browse/madame

df_combined$Title[df_combined$Title %in% other_title] <- 'Other'
table(df_combined$Title)
df_combined$Family_Member_Count <- df_combined$SibSp + df_combined$Parch + 1
df_combined$Parent <- 'Not Parent'

df_combined$Parent[df_combined$Sex == 'female' & df_combined$Parch > 0 & df_combined$Age > 18 & df_combined$Title != 'Miss'] <- 'Mother'

df_combined$Parent[df_combined$Sex == 'male' & df_combined$Parch > 0 & df_combined$Age > 18] <- 'Father'
df_combined$Parent <- factor(df_combined$Parent)

df_combined$Title <- factor(df_combined$Title)
train <- df_combined[1:891,c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Title","Family_Member_Count","Parent")]

test <- df_combined[892:1309,c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Title","Family_Member_Count","Parent")]

target <- as.factor(df_combined[[2]][1:891])
levels(target) <- c("Not_Survive", "Survive")
myControl <- trainControl(

  method = "repeatedcv",

  number = 10,

  repeats = 3,

  classProbs = TRUE, # IMPORTANT!

  verboseIter = TRUE,    

)
model <- train(train, target, method = 'rf', metric = "Accuracy", ntree = 500, trControl = myControl)
model
preds <- predict(model, train, type = "raw")
head(preds)
confusionMatrix(target, preds)
explainer <- lime(train, model)
explanation <- explain(test[5:8,], explainer, n_labels = 1, n_features = length(names(test)))
plot_features(explanation)
my_prediction <- predict(model, newdata = test, type = "raw")



my_solution <- data.frame(PassengerId = df_combined[892:1309,1], Survived = my_prediction)



my_solution$Survived <- ifelse(my_solution$Survived == 'Not_Survive', 0, 1)



write.csv(my_solution, file = "my_solution.csv", row.names = FALSE)