#Loading packages

library(ggplot2) #visualization

library(ggfortify) #visualization

library(dplyr) #data manipulation

library(psych) #data description
#importing data

Data <- read.csv('../input/train.csv')

Test <- read.csv('../input/test.csv' )



#checking data

str(Data)
head(Data) #View the first rows of the Data

tail(Data) #View the last rows of the Data
describe(Data)