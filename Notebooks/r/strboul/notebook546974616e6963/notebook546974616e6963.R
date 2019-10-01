library(tidyverse)

titanic <- read_csv("../input/train.csv")
str(titanic)
head(titanic)

tail(titanic)
ggplot(data = titanic) +

    geom_bar(mapping = aes(x = Pclass, color=Sur))
#Fare, class and frequency line graph

ggplot(data = titanic, mapping = aes(x = Age, color = Sex)) +

    geom_bar()

    