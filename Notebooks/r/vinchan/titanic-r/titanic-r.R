# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



train_df_ <- read.csv("../input/train.csv")

test_df_ <- read.csv("../input/test.csv")



# Any results you write to the current directory are saved as output.
train_df_$Ticket <- as.character(train_df_$Ticket)

train_df_$Name <- as.character(train_df_$Name)

train_df_$Cabin <- as.character(train_df_$Cabin)



# dim(train_df_)

# sum(train_df_$Cabin=="")



cherbourg_df_ <- train_df_[train_df_$Embarked=="C",]

aggregate(Survived ~ Pclass + Sex, train_df_, FUN=mean)

aggregate(Survived ~ Pclass + Sex + SibSp, train_df_, FUN=mean)



first_lm_ <- lm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Embarked, train_df_)

summary(first_lm_)

first_pred_ <- predict(first_lm_,test_df_)



write.csv(first_pred_, "../submission.csv")