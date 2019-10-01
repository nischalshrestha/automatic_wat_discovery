#loading libraries

library(mice)

library(ggplot2)

library(dplyr)

library(purrr)

library(randomForest)



#importing data

train <- read.csv("../input/train.csv", stringsAsFactors=FALSE)

test <- read.csv("../input/test.csv", stringsAsFactors=FALSE)



#combining data for full analysis

data <- bind_rows(train, test)
glimpse(data)
missing <- sapply(data, function(x) sum(is.na(x)))

print(missing)
#modifying categorical variables to factors

data$Pclass <- as.factor(data$Pclass)

levels(data$Pclass) <- c("Upper", "Middle", "Lower")



data$Embarked <- as.factor(data$Embarked)



data$Sex <- as.factor(data$Sex)

levels(data$Sex) <- c("Female", "Male")
head(data)
set.seed(765)

data_to_mice <- data %>% select(Pclass, Sex, Age, SibSp, Parch, Fare)

m_model <- mice(data_to_mice, method='rf')



imputes <- complete(m_model)



ggplot() +

    geom_density(data = data, aes(x=Age), fill='blue', alpha=0.5) +

    geom_density(data = imputes, aes(x=Age), fill='red', alpha=0.5)



data$Age <- imputes$Age
data %>% filter(Pclass=='Lower', Embarked=='S') %>% 

  ggplot(aes(Fare)) +

  geom_histogram(binwidth=1.2, fill='blue3', alpha=0.5) +

  geom_vline(aes(xintercept=mean(Fare, na.rm = TRUE)), col="red") +

  geom_vline(aes(xintercept=median(Fare, na.rm = TRUE)), col="green")
data$Fare[is.na(data$Fare)] <- median(data[data$Pclass == 'Lower' & data$Embarked=='S', ]$Fare, na.rm = TRUE)
data$FamilySize <- ifelse(data$SibSp + data$Parch + 1 == 1, "single", ifelse(data$SibSp + data$Parch + 1 < 4, "small", "large"))
ggplot(data, aes(FamilySize)) +

  geom_bar(position="dodge") + 

  scale_x_discrete(limits=c("single", "small", "large"))
data$Title <- gsub('(.*, )|(\\..*)', '', data$Name)

print(unique(data$Title))
#reassigning bizzare titles

data$Title[data$Title == 'Mlle']<- 'Miss' 

data$Title[data$Title == 'Ms']  <- 'Miss'

data$Title[data$Title == 'Mme'] <- 'Mrs' 

data$Title[data$Title == 'Dona']<- 'Mrs'

data$Title[data$Title == 'Don'] <- 'Mr'



#I will add VIP, Crew and Ohter for convinieance

vip <- c('Lady', 'the Countess', 'Sir', 'Jonkheer')

crew <- c('Capt', 'Col', 'Major')

other <- c('Dr', 'Rev')



data$Title[data$Title %in% vip] <- 'VIP'

data$Title[data$Title %in% crew] <- 'CREW'

data$Title[data$Title %in% other] <- 'OTHER'
data$FamilyName <- gsub(",.*","",data$Name)
data$Sector <- sapply(data$Cabin, substr, 1, 1)

data$Sector[data$Sector==""] <- NA

Family.Sector <- data %>% select(FamilyName, Sector) %>% filter(!is.na(Sector)) %>% unique()



data$Sector <- apply(data[, c("FamilyName", "Sector")], 1, function(x) {

  if(x["FamilyName"] %in% Family.Sector$FamilyName & is.na(x["Sector"])) {

    return(Family.Sector$Sector[Family.Sector$FamilyName == x["FamilyName"]][1])

  } else if(!is.na(x["Sector"])){

    return(x["Sector"])

  } else {

    return("M")

  }

})
head(data[c("FamilyName", "Sector")])
table(data$Sector)
data$FamilySize <- as.factor(data$FamilySize)

data$Title <- as.factor(data$Title)

data$Sector <- as.factor(data$Sector)
train <- data[1:891,]

test <- data[892:1309,]



titanic_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 

                                            Fare + Embarked + FamilySize + 

                                            Title + Sector,

                                            data = train)
plot(titanic_model)
#BY Megan L. Risdal

importance    <- importance(titanic_model)

varImportance <- data.frame(Variables = row.names(importance), 

                            Importance = round(importance[ ,'MeanDecreaseGini'],2))



# Create a rank variable based on importance

rankImportance <- varImportance %>%

  mutate(Rank = paste0('#',dense_rank(desc(Importance))))



# Use ggplot2 to visualize the relative importance of variables

ggplot(rankImportance, aes(x = reorder(Variables, Importance), 

    y = Importance, fill = Importance)) +

  geom_bar(stat='identity') + 

  geom_text(aes(x = Variables, y = 0.5, label = Rank),

    hjust=0, vjust=0.55, size = 4, colour = 'red') +

  labs(x = 'Variables') +

  coord_flip()
prediction <- predict(titanic_model, test)

solution <- data.frame(PassengerId = test$PassengerId, Survived = prediction)

write.csv(solution, file = 'rf_mod_Solution.csv', row.names = F)