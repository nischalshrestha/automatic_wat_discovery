#_____________________________Package + Data___________________________________

#______________________________________________________________________________



# Package

suppressMessages(library('ggplot2'))

suppressMessages(library('ggthemes')) 

suppressMessages(library('scales')) 

suppressMessages(library('dplyr'))

suppressMessages(library('randomForest'))



#Loading Data



train <- read.csv('../input/train.csv', stringsAsFactors = F)

test  <- read.csv('../input/test.csv', stringsAsFactors = F)



full  <- bind_rows(train, test) # test + train
str(full)
summary(full)
#______________________________________________________________________________

#_____________________________Exploratory Analysis_____________________________

#______________________________________________________________________________



# Age vs Survived

ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 

  geom_histogram(bins=30) + 

  theme_few() +

  xlab("Age") +

  scale_fill_discrete(name = "Survived") + 

  ggtitle("Age vs Survived")
# Sex vs Survived

ggplot(full[1:891,], aes(Sex, fill = factor(Survived))) + 

  geom_bar(stat = "count", position = 'dodge')+

  theme_few() +

  xlab("Sexe") +

  ylab("Count") +

  scale_fill_discrete(name = "Survived") + 

  ggtitle("Sex vs Survived")



tapply(full[1:891,]$Survived,full[1:891,]$Sex,mean)
#Sex vs Survived vs Age 

ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 

  geom_histogram(bins=30) + 

  theme_few() +

  xlab("Age") +

  ylab("Count") +

  facet_grid(.~Sex)+

  scale_fill_discrete(name = "Survived") + 

  theme_few()+

  ggtitle("Age vs Sex vs Survived")
# Pclass vs Survived



tapply(full[1:891,]$Survived,full[1:891,]$Pclass,mean)



ggplot(full[1:891,], aes(Pclass, fill = factor(Survived))) + 

  geom_bar(stat = "count")+

  theme_few() +

  xlab("Pclass") +

  facet_grid(.~Sex)+

  ylab("Count") +

  scale_fill_discrete(name = "Survived") + 

  ggtitle("Pclass vs Sexe vs Survived")
ggplot(full[1:891,], aes(x = Age, y = Sex)) + 

  geom_jitter(aes(colour = factor(Survived))) + 

  theme_few()+

  theme(legend.title = element_blank())+

  facet_wrap(~Pclass) + 

  labs(x = "Age", y = "Sex", title = "Pclass vs Sex vs Age vs Survived")+

  scale_fill_discrete(name = "Survived") + 

  scale_x_continuous(name="Age",limits=c(0, 81))
#Fare

ggplot(full[1:891,], aes(x = Fare, y = Pclass)) + 

  geom_jitter(aes(colour = factor(Survived))) + 

  theme_few()+

  theme(legend.title = element_blank())+

  labs(x = "Age", y = "Pclass", title = "Fare vs Pclass")+

  scale_fill_discrete(name = "Survived") + 

  scale_x_continuous(name="Fare", limits=c(0, 270), breaks=c(0, 40, 80, 120, 160, 200, 240, 280))
#______________________________________________________________________________

#______________________Data processing and ___________________________________

#_________________________exploratory analysis 2______________________________



#__________________________________Title_______________________________________

# Extract titles

full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)



# Titles by Sex

table(full$Sex, full$Title)
# Reassign rare titles

officer <- c('Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev')

royalty <- c('Dona', 'Lady', 'the Countess','Sir', 'Jonkheer')



# Reassign mlle, ms, and mme, and rare

full$Title[full$Title == 'Mlle']        <- 'Miss' 

full$Title[full$Title == 'Ms']          <- 'Miss'

full$Title[full$Title == 'Mme']         <- 'Mrs' 

full$Title[full$Title %in% royalty]  <- 'Royalty'

full$Title[full$Title %in% officer]  <- 'Officer'



full$Surname <- sapply(full$Name,  

                       function(x) strsplit(x, split = '[,.]')[[1]][1])
#graph title

ggplot(full[1:891,], aes(Title,fill = factor(Survived))) +

  geom_bar(stat = "count")+

  xlab('Title') +

  ylab("Count") +

  scale_fill_discrete(name = " Survived") + 

  ggtitle("Title vs Survived")+

  theme_few()



tapply(full[1:891,]$Survived,full[1:891,]$Title,mean)
#____________________________Family Size________________________________



#Family

# Family Size

full$Fsize <- full$SibSp + full$Parch + 1



ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') +

  scale_x_continuous(breaks=c(1:11)) +

  xlab('Family Size') +

  ylab("Count") +

  theme_few()+

  scale_fill_discrete(name = "Survived") + 

  ggtitle("Family Size vs Survived")



tapply(full[1:891,]$Survived,full[1:891,]$Fsize,mean)
# FsizeD

full$FsizeD[full$Fsize == 1] <- 'Alone'

full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'Small'

full$FsizeD[full$Fsize > 4] <- 'Big'



tapply(full[1:891,]$Survived,full[1:891,]$FsizeD,mean)



mosaicplot(table(full$FsizeD, full$Survived), main='Family Size vs Survived', ylab="Survived",xlab="Family Size",col = hcl(c(50, 120)),)
#________________________________Embarked______________________________________

# 2 missing datas : input median



tapply(full$Embarked, full$Pclass,median, na.rm=TRUE)

full[c(62, 830), 'Embarked']



full$Embarked[c(62, 830)] <- 'S'



ggplot(full[1:891,], aes(Pclass, fill = factor(Survived))) + 

  geom_bar(stat = "count")+

  theme_few() +

  xlab("Pclasse") +

  ylab("Count") +

  facet_wrap(~Embarked) + 

  scale_fill_discrete(name = "Survived") + 

  ggtitle("Embarked vs Pclass vs Survived")
#_________________________________Fare_________________________________________ 



full[1044, ]



ggplot(full[full$Pclass == '3', ], 

       aes(x = Fare)) +

  geom_density(fill = 'lightgrey', alpha=0.4) + 

  geom_vline(aes(xintercept=median(Fare, na.rm=T)),

             colour='darkred', linetype='dashed', lwd=1) +

  xlab('Fare') +

  ggtitle("Pclass = 3")+

  ylab("Density") +

  scale_x_continuous(labels=dollar_format()) +

  theme_few()



tapply(full$Fare, full$Pclass,median, na.rm=TRUE)



full$Fare[1044] <- median(full[full$Pclass == '3', ]$Fare, na.rm = TRUE)
#__________________________________Age__________________________________________



tapply(full$Age, full$Pclass,median, na.rm=TRUE)

tapply(full$Age, full$Title,median, na.rm=TRUE)
title.age <- aggregate(full$Age,by = list(full$Title), FUN = function(x) median(x, na.rm = T))



full[is.na(full$Age), "Age"] <- apply(full[is.na(full$Age), ] , 1, function(x) title.age[title.age[, 1]==x["Title"], 2])



#Na value count

sum(is.na(full$Age))
#__________________________________Child__________________________________________



full$Child[full$Age < 18] <- 'Child'

full$Child[full$Age >= 18] <- 'Adult'



ggplot(full[1:891,][full[1:891,]$Child == 'Child', ], aes(Sex, fill = factor(Survived))) + 

  geom_bar(stat = "count") + 

  xlab("Sex") +

  ylab("Count") +

  facet_wrap(~Pclass)+

  scale_fill_discrete(name = "Survie") +

  ggtitle("Child vs Sex vs Pclass vs Survived")+

  theme_few()



tapply(full[1:891,]$Survived,full[1:891,]$Child, mean)

table(full$Child, full$Survived)
#_________________________________Factor__________________________________________



full$Child  <- factor(full$Child)

full$Sex  <- factor(full$Sex)

full$Embarked  <- factor(full$Embarked)

full$Title  <- factor(full$Title)

full$Pclass  <- factor(full$Pclass)

full$FsizeD  <- factor(full$FsizeD)



#___________________________Data without Cabin & Ticket __________________________



full1 <- full[,-9]

full_mod <- full1[,-10]

#______________________________________________________________________________

#_____________________________Modeling + predict_______________________________

#______________________________________________________________________________



# Split full_mod

train <- full_mod[1:891,]

test <- full_mod[892:1309,]



# random forest

library('randomForest')



set.seed(123)

rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Fare + Embarked + Title + 

                           FsizeD + Child, data = train)
# prediction

rf.fitted = predict(rf_model)

ans_rf = rep(NA,891)

for(i in 1:891){

  ans_rf[i] = as.integer(rf.fitted[[i]]) - 1

}

# Résultat

mean(ans_rf == train$Survived)

table(ans_rf)



print(rf_model)

mean(ans_rf == train$Survived)

varImpPlot(rf_model)
prediction <- predict(rf_model, test)



# Solution 2 columns (prediction)

solution <- data.frame(Survived = prediction, PassengerID = test$PassengerId)



# .csv

write.csv(solution, file = 'rf_model_sol.csv', row.names = F)
# Error

plot(rf_model, ylim=c(0,0.36))

legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)
# Var importantes

importance    <- importance(rf_model)

varImportance <- data.frame(Variables = row.names(importance), 

                            Importance = round(importance[ ,'MeanDecreaseGini'],2))



# var imp

rankImportance <- varImportance %>%

  mutate(Rank = paste0('#',dense_rank(desc(Importance))))



# Graph var importantes

ggplot(rankImportance, aes(x = reorder(Variables, Importance), 

                           y = Importance, fill = Importance)) +

  geom_bar(stat='identity') + 

  geom_text(aes(x = Variables, y = 0.5, label = Rank),

            hjust=0, vjust=0.55, size = 4, colour = 'red') +

  labs(x = 'Variables') +

  coord_flip() + 

  theme_few()