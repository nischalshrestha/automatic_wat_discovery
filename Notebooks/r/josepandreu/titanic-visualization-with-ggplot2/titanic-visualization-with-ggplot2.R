if(!require(ggplot2)){

    install.packages('ggplot2', repos='http://cran.es.r-project.org') 

    require(ggplot2)

}

if(!require(reshape2)){

    install.packages('reshape2', repos='http://cran.es.r-project.org') 

    require(reshape2)

}
train_titanic <- read.csv("../input/train.csv", sep=",", header=T)

test_titanic <- read.csv("../input/test.csv", sep=",", header=T)
g1 <- data.frame(Age=train_titanic$Age, Cabin=train_titanic$Cabin, SibSp=train_titanic$Age, Parch=train_titanic$Parch, Sex=train_titanic$Sex, Pclass=train_titanic$Pclass, Survived=train_titanic$Survived)

g2 <- data.frame(Age=test_titanic$Age, Cabin=test_titanic$Cabin, SibSp=test_titanic$Age, Parch=test_titanic$Parch, Sex=test_titanic$Sex, Pclass=test_titanic$Pclass)



head(g1)

ggplot() +

    geom_jitter(data=g1, aes(Pclass, Age, colour = factor(Survived))) + 

    facet_grid(Sex ~ .) + 

    theme_light()





ggplot() +

    geom_jitter(data=g2, aes(Pclass, Age), size=0.1, colour="green") + 

    geom_jitter(data=g1, aes(Pclass, Age, colour = factor(Survived))) + 

    #scale_fill_brewer(palette = "Greens") +

    facet_grid(Sex ~ .) + 

    theme_dark()
