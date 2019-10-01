train<- read.csv("https://github.com/EasyD/IntroToDataScience/blob/master/train.csv")

str(train)

test<- read.csv("https://github.com/EasyD/IntroToDataScience/blob/master/test.csv")



test.survived <- data.frame(survived = rep("None", nrow(test)), test[,])

data.combined<- rbind.data.frame(train,test.survived)

                                

table(data.combined$Survived)

table(data.combined$Pclass)

 library(ggplot2)

train$Pclass<- as.factor(train$Pclass)

ggplot(train,aes(x=Pclass, fill=factor(survived)))+

  geom_histogram(width=0.5)+

  xlab("pclass")+

  ylab("total count")+

labs(fill="survived")
