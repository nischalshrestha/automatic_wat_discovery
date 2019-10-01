# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
# Reading training and test dataset

trainDS <- read.csv('../input/train.csv', stringsAsFactors = F)

testDS  <- read.csv('../input/test.csv', stringsAsFactors = F)

head(trainDS)
Features = trainDS[ , c("Pclass", "Age", "Sex")]

head(Features)
meanAge = mean(Features[, "Age"], na.rm=TRUE) #29.69

Features$Age[is.na(Features$Age)] <- meanAge
dat <- data.frame(X=Features[, "Pclass"],

                  Y=Features[, "Age"],

                  att1=trainDS[, "Survived"],

                  att2=trainDS[, "Sex"])

head(dat)

ggplot(dat) + geom_point(aes(x=X, y=Y, color=factor(att1),shape=att2), size=5) 
# 