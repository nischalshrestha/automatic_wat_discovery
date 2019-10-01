# Libraries

library(ggplot2, quietly=TRUE)

library(readr, quietly=TRUE)

library(psych, quietly=TRUE)

library(dplyr, quietly=TRUE)

library(ggmosaic, quietly=TRUE)
data <- read_csv("../input/train.csv")

str(data)
head(data)
# Clean data

data.clean <- data[!is.na(data$Age),]



ggplot(data.clean, aes(x=Age)) +

  geom_histogram(binwidth = 2)



describe(data.clean$Age)
# Devide age into age groups

data.clean$Age.Group <- ifelse(data.clean$Age < 10, "0 ~ 9", 

                               ifelse(data.clean$Age < 20, "10 ~ 19",

                                      ifelse(data.clean$Age < 30, "20 ~ 29",

                                             ifelse(data.clean$Age < 40, "30 ~ 39",

                                                    ifelse(data.clean$Age < 50, "40 ~ 49",

                                                           ifelse(data.clean$Age < 0, "50 ~ 59", "60+"))))))



table(data.clean$Age.Group, data.clean$Survived)
mosaicplot(~ Age.Group + as.factor(Survived), data = data.clean, color = 6:7, las = 1)
ggplot(data.clean, aes(x=Age, color=Sex)) +

  geom_freqpoly(binwidth = 2)
mosaicplot(~ Sex + as.factor(Survived), data = data.clean, color = 6:7, las = 1)
# Create family size

data.clean$Family <- data.clean$SibSp + data.clean$Parch

data.clean$Family.Size <- ifelse(data.clean$Family == 0, "Single",

                               ifelse(data.clean$Family == 1, "Small", "Large"))
mosaicplot(~ Family.Size + as.factor(Survived), data = data.clean, color = 6:7, las = 1)
ggplot(data.clean, aes(x=Fare)) +

  geom_histogram(binwidth=5)
# cut_number is not giving satisfing results here.

data.clean$Fare.Group <- cut_number(data.clean$Fare, 8)

mosaicplot(~ Fare.Group + as.factor(Survived), data = data.clean, color = 6:7, las = 1)
mosaicplot(~ Fare.Group + as.factor(Sex), data = data.clean, color = 6:7, las = 1)
mosaicplot(~ Fare.Group + as.factor(Family.Size), data = data.clean, color=2:7, las = 1)