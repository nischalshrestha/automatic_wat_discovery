# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(caret) # machine learning and parameter tuning

library(randomForest) # Random forrest!

library(fields)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.

# The train and test data is stored in the ../input directory

train <- read.csv("../input/train.csv")

test  <- read.csv("../input/test.csv")



# We can inspect the train data. The results of this are printed in the log tab below

summary(train)

head(train)
bplot.xy(train$Survived, train$Age)

bplot.xy(train$Survived, train$Fare)

## Let's train a model

# Converting the Survived to Factor

train$Survived <- factor(train$Survived)

# SEED

set.seed(pi)

# Training the model

train.model <- train(Survived ~ Pclass + Sex + SibSp + Embarked + Parch + Fare,

                    data = train, 

                    method = "rf",

                    trControl = trainControl(method = "cv", number = 5)

                    )

# vealuating the train Model

train.model
## Let's Predict now!



test_prediction <- predict(train.model, data=test)

my_soluction_aliechoes <- data.frame(test_prediction)

write.csv(my_soluction_aliechoes, file = "my_soluction_aliechoes.csv", row.names = FALSE)
