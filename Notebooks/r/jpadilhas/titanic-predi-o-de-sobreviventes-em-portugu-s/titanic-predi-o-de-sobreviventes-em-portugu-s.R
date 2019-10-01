library(ggplot2) # Visualização de Dados

library(readr) # Leitura de arquivo

library('randomForest') #predição de sobreviventes
titanic.train <- read.csv("../input/train.csv", stringsAsFactor=F, header = T) #Arquivo de Treino

titanic.test <- read.csv("../input/test.csv", stringsAsFactor=F, header = T) #Arquivo de Teste
str(titanic.train)
table(is.na(titanic.train))

table(is.na(titanic.test))
ncol(titanic.train)

ncol(titanic.test)
titanic.test$Survived <- NA
titanic.train$TrainSet <- TRUE

titanic.test$TrainSet <- FALSE
titanic.full <- rbind(titanic.train, titanic.test)

table(titanic.full$TrainSet)

table(titanic.full$Survived, titanic.full$Sex)
boxplot(titanic.full$Fare)
boxplot.stats(titanic.full$Fare)
fare.wirsker <- boxplot.stats(titanic.full$Fare)$stats[5]



ggplot(data = titanic.train[titanic.train$Fare <= fare.wirsker,], aes(x = Fare, fill = factor(Survived))) +

  geom_histogram() +

  labs(x = 'Custo da Passagem', y = 'Total de Passageiros') +

  facet_grid(.~Sex) + 

  theme_minimal()