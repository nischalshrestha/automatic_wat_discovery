# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(tidyverse)



cat("Files in input folder...\n")

list.files("../input")
# Import data

d <- read_csv("../input/train.csv")



d %>% head()
# Visulaising age

ggplot(d, aes(Age, Survived)) +

    geom_jitter()