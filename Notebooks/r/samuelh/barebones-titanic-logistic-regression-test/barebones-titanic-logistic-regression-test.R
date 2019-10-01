# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.



library(tidyverse)

library(mice)



#Load data



df.train = read_csv("../input/train.csv")

df.test = read_csv("../input/test.csv")

df.example = read_csv("../input/gender_submission.csv")
df.train  %>%  head(100)

df.train  %>%  sapply(class)

summary(df.train)



df.example  %>%  head(10)

df.example  %>%  sapply(class)

summary(df.example)
df.train2 = df.train  %>% 

  mutate(Cabin = ifelse(Cabin == "", NA, Cabin)) %>% 

  mutate(Embarked = ifelse(Embarked == "", NA, Embarked)) %>%

  mutate(

    Pclass = factor(Pclass),

    Sex = factor(Sex),

    age_minor = Age < 18,

    cabin_code = factor(str_extract(Cabin, "[A-Z]+")),

    Embarked = factor(Embarked))  %>% 

  select(PassengerId, Survived, Pclass, Sex, age_minor, Fare, cabin_code, Embarked)



df.test2 = df.test  %>% 

  mutate(Cabin = ifelse(Cabin == "", NA, Cabin)) %>% 

  mutate(Embarked = ifelse(Embarked == "", NA, Embarked)) %>%

  mutate(

    Pclass = factor(Pclass),

    Sex = factor(Sex),

    age_minor = Age < 18,

    cabin_code = factor(str_extract(Cabin, "[A-Z]+")),

    Embarked = factor(Embarked))  %>% 

  select(PassengerId, Pclass, Sex, age_minor, Fare, cabin_code, Embarked)



#Impute missing values

df.train3 = mice(df.train2)  %>% 

complete  %>% 

mutate(age_minor = as.logical(age_minor))



df.test3 = mice(df.test2)  %>% 

complete %>% 

mutate(age_minor = as.logical(age_minor))

panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...)

{

    usr <- par("usr"); on.exit(par(usr))

    par(usr = c(0, 1, 0, 1))

    r <- abs(cor(x, y, use = "complete.obs"))

    txt <- format(c(r, 0.123456789), digits = digits)[1]

    txt <- paste0(prefix, txt)

    if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)

    text(0.5, 0.5, txt, cex = cex.cor * r)

}



df.train3  %>% select(-PassengerId)  %>% pairs(upper.panel = panel.cor)
model.logit = glm(

    Survived ~ Pclass + Sex + age_minor + Fare + cabin_code + Embarked,

    data = df.train3,

    family = binomial)



summary(model.logit)
vt.predicted <- predict.glm(object = model.logit,

                            newdata = df.test3,

                            type = 'response')
df.submission = data.frame(PassengerId = as.integer(df.test3$PassengerId),

                          Survived = as.numeric(vt.predicted > 0.5))



df.submission



write_csv(df.submission,

          'logistic_reg_sub_ex.csv')