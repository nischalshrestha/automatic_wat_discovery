# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(dplyr)

library('stringr')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train <- read.csv('../input/train.csv', stringsAsFactors = F)



# Any results you write to the current directory are saved as output.
train$Ticket_Num <- sapply(train$Ticket, FUN=function(x) {ifelse(str_detect(x, " "),str_split(x, " ")[[1]][2], as.character(x))})
ggplot(train, aes(x =Ticket_Num, y =Fare)) + geom_point(aes(colour = Pclass,shape = factor(Survived)))
ggplot(train, aes(x =Ticket_Num, y =Fare)) + geom_text(aes(label=Ticket_Num))
train$Ticket_Num2[train$Ticket_Num <= 10000] <- 1

train$Ticket_Num2[train$Ticket_Num <= 20000] = 2

train$Ticket_Num2[train$Ticket_Num <= 30000] = 3

train$Ticket_Num2[train$Ticket_Num > 30000] = 4
ggplot(train, aes(x =Ticket_Num, y =Fare)) + geom_text(aes(label=Ticket_Num2,colour= factor(Survived)))
