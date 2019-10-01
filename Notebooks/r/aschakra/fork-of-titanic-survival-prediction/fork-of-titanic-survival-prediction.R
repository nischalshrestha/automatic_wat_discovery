# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
train = read.csv('../input/train.csv')

test = read.csv('../input/test.csv')
head(train)

head(test)
train1 = train[ -c(1, 4, 6, 9, 11, 12)]

head(train1, n=20)
library("randomForest")

m <- randomForest(Survived ~ ., data = train1)
test1 = test[ -c(1, 3, 5, 8, 10, 11)]

head(test1, n=20)
test_surv = predict(m, test1)

test_surv