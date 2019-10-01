# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
for (filename in list.files("../input", full.names=TRUE)) {

    cat(paste0(filename, ",\tmode: ", file.mode(filename), ",\tsize: ", file.size(filename)), "bytes.\n")

}
data <- read.csv("../input/train.csv", stringsAsFactors = FALSE)

test_data <- read.csv("../input/test.csv", stringsAsFactors = FALSE)

str(data)
library(dplyr)



get_title <- function(Name) {

    title <- gsub("(.*, )|(\\..*)", "", Name)

    

    # According to Wikipedia, Mlle is the French-language equivalent of "miss".

    title <- ifelse(title == "Mlle", "Miss", title)

    

    # Again according to Wikipedia, Mme is the French abbreviation for Madame.

    title <- ifelse(title == "Mme", "Mrs", title)

    

    title

}



data <- data %>% mutate(title = get_title(Name))

test_data <- test_data %>% mutate(title = get_title(Name))



data %>% 

    group_by(title) %>% summarise(n=n()) %>% 

    full_join(test_data %>% group_by(title) %>% summarise(n=n()), by="title") %>% print()
titles <- test_data %>% mutate(Survived=NA) %>% rbind(data) %>% group_by(title) %>% summarise(title_frequency = n())

print(titles)
replace_rare_titles <- function(d, titles) {

    d %>% left_join(titles, by="title") %>%

    mutate(title=ifelse(title_frequency < 10, "rare", title))

}



data <- replace_rare_titles(data, titles)

test_data <- replace_rare_titles(test_data, titles)



data %>% 

    group_by(title) %>% summarise(n=n()) %>% 

    full_join(test_data %>% group_by(title) %>% summarise(n=n()), by="title") %>% print()
data %>% group_by(title) %>% summarise(prob = mean(Survived)) %>% print()
data %>% group_by(Sex) %>% summarise(prob = mean(Survived)) %>% print()
data %>%

    group_by(Survived) %>% summarise(mean_fare = mean(Fare), ci = qnorm(0.975) * sd(Fare) / (n() - 1)) %>%

    print()
data %>% 

    group_by(Survived) %>% 

    summarise(mean_age = mean(Age, na.rm=TRUE), ci = qnorm(0.975) * sd(Age, na.rm=TRUE) / (n() - 1)) %>% print()
data$title <- as.factor(data$title)

data$Survived <- as.factor(data$Survived)

data$Sex <- as.factor(data$Sex)

str(data)
fm <- glm(Survived ~ title + Sex + Age + Fare, data=data, family="binomial")

summary(fm)
test_data$prediction <- predict(fm, newdata=test_data, type="response")

test_data$prediction[is.na(test_data$prediction)] <- as.integer(round(mean(as.numeric(data$Survived))))
submission <- data.frame(PassengerId = test_data$PassengerId, Survived = as.integer(test_data$prediction > 0.5))
print(head(submission, n=10))

print(nrow(submission))
write.csv(submission, file = './submission_01.csv', row.names = FALSE)