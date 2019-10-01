# This R enviro ere's several helpful packages to load in 



library(ggplot2) # Gráficos

library(dplyr) # data manipulation

library(randomForest)

library(ggthemes)

# Any results you write to the current directory are saved as output.
# Carregando os arquivos de Treino e Teste.

data_train <- read.csv("../input/train.csv")

data_test<- read.csv("../input/train.csv")



head(data_train)

colnames(data_train)
# Removendo colunas.

data_train$PassengerId <- NULL

data_train$SibSp <- NULL

data_train$Parch <- NULL

data_train$Ticket <- NULL

data_train$Cabin <- NULL

data_train$Name <- NULL
# Ordenando as colunas

data_train <- data_train %>%

select(Pclass, Age, Sex, Fare, Embarked, Survived)
head(data_train)
# Transformando as variáveis em facto.

data_train$Survived <- as.factor(data_train$Survived)

levels(data_train$Survived) <- c("Não", "Sim")



data_train$Sex <- as.factor(data_train$Sex)

levels(data_train$Sex) <- c("Feminino", "Masculino")



data_train$Pclass <- as.factor(data_train$Pclass)

levels(data_train$Pclass) <- c("Alta", "Média", "Baixa")



data_train$Age <- as.numeric(data_train$Age)

data_train$Age <- cut(data_train$Age, c(0, 30, 50, 100), labels = c("Jovem", "adulto", "Idoso"))



data_train$Embarked <- as.factor(data_train$Embarked)

levels(data_train$Embarked) <- c(0,"Cherbourg","Southampton", "Queenstown")
head(data_train)
#########################################################################

#Eclat Algotitmo utilizado para encontrar padrões nos data sets #########

#########################################################################

library(arules)

library(arulesViz)



data_train$Fare <- as.factor(data_train$Fare)

regras <- eclat(data_train, parameter = list(supp = 0.1, maxlen = 5))

inspect(regras)

#install.packages("aruleViz")

plot(regras, method="graph", control=list(type="items"))

# Verificando se a valores missing

sapply(data_train, function(x) sum(is.na(x)))

#missmap(data_train, main = "Valores Missing Observados")

data_train <- na.omit(data_train)

head(data_train)
# Esses 3 gráficos nos mostram que pessoas mais velhas, de classe social alta e do sexo 

# feminino pagarama mais caro nas cabines do navio.



data_train$Fare <- as.numeric(data_train$Fare)

# Gráfico média de preço da passagem por Idade

ggplot(data_train) + stat_summary(aes(x = data_train$Age, y = data_train$Fare),

                               fun.y = mean, geom = "bar",

                               fill = "lightgreen", col = "grey50")

#Gráfico média de preço da passagem por CLasse Social

ggplot(data_train) + stat_summary(aes(x = data_train$Pclass, y = data_train$Fare),

                               fun.y = mean, geom = "bar",

                               fill = "lightblue", col = "grey50")



#Gráfico média de preço da passagem por Sexo

ggplot(data_train) + stat_summary(aes(x = data_train$Sex, y = data_train$Fare),

                               fun.y = mean, geom = "bar",

                               fill = "lightblue", col = "grey50")
# Age vs Survived

ggplot(data_train[1:714,],aes(Age, fill = (Survived))) + 

  geom_bar(stat = "count") + 

  theme_few() +

  xlab("Age") +

  facet_grid(.~Sex)+

  ylab("count") +

  scale_fill_discrete(name = "Survived") + 

  ggtitle("Age vs Survived")



# Pclass vc Survived

ggplot(data_train[1:714,], aes(Pclass, fill = (Survived))) +

  geom_bar(stat = "count") +

  theme_few() +

  xlab("Pclass") +

  facet_grid(.~Sex) +

  ylab("count") +

  scale_fill_discrete(name = "Survided") +

  ggtitle("Pclass vs Survived")



# Embarked vs Survived

ggplot(data_train[1:714,], aes(Embarked, fill = (Survived))) +

  geom_bar(stat = "count") +

  xlab("Embarked") +

  facet_grid(.~Sex) +

  ylab("count") +

  scale_fill_discrete(name = "Survived") +

  ggtitle("Embarked vs Survived")
#########################################################################

################## Random Forest Classification Model ###################

#########################################################################



# Set the seed

set.seed(12345)



# Contruindo o modelo Random Forest

rf_model <- randomForest(Survived ~ ., data = data_train)

print("Modelo contruido")

rf_model
varImpPlot(rf_model)
########################################################################

############## Carregando dataset de Teste #############################

########################################################################



head(data_test)
data_test$PassengerId <- NULL

data_test$Name <- NULL

data_test$SibSp <- NULL

data_test$Parch <- NULL

data_test$Parch <- NULL

data_test$Ticket <- NULL

data_test$Cabin <- NULL
# Ordenando as colunas

data_test <- data_test %>%

  select(Pclass, Age, Sex, Fare, Embarked)
head(data_test)
# Transoformando as Variáveis em fator, numerico e character

data_test$Pclass <- as.factor(data_test$Pclass)

levels(data_test$Pclass) <- c("Alta", "Média", "Baixa")



data_test$Sex <- as.factor(data_test$Sex)

levels(data_test$Sex) <- c("Feminino", "Masculino")



data_test$Age <- as.numeric(data_test$Age)

data_test$Age <- cut(data_test$Age, c(0, 30, 50, 100), labels = c("Jovem", "adulto", "Idoso"))



data_test$Embarked <- as.factor(data_test$Embarked)

levels(data_test$Embarked) <- c(0, "Cherbourg", "Queenstown", "Southampton")
head(data_test)
# Verificando se a valores missing

sapply(data_test, function(x) sum(is.na(x)))

data_test <- na.omit(data_test)

head(data_test)
####################### Previsão Randon Forest #####################################

####################################################################################

predictionrf <- predict(rf_model, data_test)

predictionrf <- data.frame(predictionrf)

predictionrf
data_test['Previsão'] <- c(predictionrf)

data_test

table(predictionrf)
# Conferindo o erro do modelo 

plot(rf_model, ylim = c(0,0.36))

legend('topright', colnames(rf_model$err.rate), col = 1:3, fill = 1:3)



varImpPlot(rf_model)
# Obtendo as variaveis mais importantes

importance <- importance(rf_model)

varimportance <- data.frame(Variables = row.names(importance), importance = round(importance[,'MeanDecreaseGini'],2))



# Criando o rank de variaveis baseado na importância

rankImportance <- varimportance %>%

  mutate(Rank = paste0('#', dense_rank(desc(importance))))



# Usando o ggplot2 para visualizar a importância relativa das variaveis

ggplot(rankImportance, aes(x = reorder(Variables, importance), y = importance, fill = importance))+

  geom_bar(stat = 'identity')+

  geom_text(aes(x = Variables, y = 0.5, label = Rank), hjust = 0, vjust=0.55, size = 4, colour = 'red')+

  labs(x = 'Variables')+

  coord_flip()
