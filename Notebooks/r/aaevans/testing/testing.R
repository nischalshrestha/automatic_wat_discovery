# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library('ggplot2') # visualization

library('ggthemes') # visualization

library('scales') # visualization

library('dplyr') # data manipulation

library('mice') # imputation

library('randomForest') # classification algorithm





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.



test_df <- read.csv("../input/test.csv", stringsAsFactors = FALSE)

train_df <- read.csv("../input/train.csv", stringsAsFactors = FALSE)



full_df <- bind_rows(test,train)
str(full_df)
summary(full_df)
hist(full_df$Age)
model <- glm(Survived ~ Pclass, family = "binomial", train)

p <- predict(model, test, type = "response")

str(p)
summary(p)
plot(p)