# -----------------------------------------------------------------------------------

# ---- Cross Validation Parameters -------------------------------------------------------------------

# -----------------------------------------------------------------------------------



# folds

K = 2



# replications per fold

B = 5



# -----------------------------------------------------------------------------------

# ---- Packages ---------------------------------------------------------------------

# -----------------------------------------------------------------------------------



{



# these are the packages needed to run this script file



# data handling

require(data.table)

require(stringr)

require(mice)



# plotting

require(ggplot2)

require(gridExtra)

require(GGally)

require(VIM)

require(lattice)

require(scales)

require(corrplot)



# modeling

require(MASS)

require(neuralnet)

require(randomForest)

require(e1071)

require(glmnet)

require(pROC)

require(caret)

require(crossval)

require(SuperLearner)

require(xgboost)



# parallel computing

require(foreach)

require(doParallel)



}



# -----------------------------------------------------------------------------------

# ---- Functions --------------------------------------------------------------------

# -----------------------------------------------------------------------------------



{



# ---- prints the data types of each column in a data frame -------------------------



types = function(dat)

{

	dat = data.frame(dat)

  

	column = sapply(1:ncol(dat), function(i) colnames(dat)[i])

	data.type = sapply(1:ncol(dat), function(i) class(dat[,i]))

	

	return(data.frame(cbind(column, data.type)))	

}



# ---- emulates the default ggplot2 color scheme ------------------------------------



ggcolor = function(n)

{

	hues = seq(15, 375, length = n + 1)

	hcl(h = hues, l = 65, c = 100)[1:n]

}



# ---- summarizes diagnostic errors of a crossval() object --------------------------



diag.cv = function(cv)

{

	require(data.table)

	

	# extract table of confusion matrix results

	dat = data.table(cv$stat.cv)

	

	# compute performance metrics for each iteration

	output = lapply(1:nrow(dat), function(i) 

				data.table(acc = (dat$TP[i] + dat$TN[i]) / (dat$FP[i] + dat$TN[i] + dat$TP[i] + dat$FN[i]),

							sens = dat$TP[i] / (dat$TP[i] + dat$FN[i]),

							spec = dat$TN[i] / (dat$FP[i] + dat$TN[i]),

							auc = dat$AUC,

							or = (dat$TP[i] * dat$TN[i]) / (dat$FN[i] * dat$FP[i])))

	

	# merge the list into one table

	output = rbindlist(output)

	

	# replace any NaN with NA

	output = as.matrix(output)

	output[is.nan(output)] = NA

	output = data.table(output)

	

	# compute summary stats

	output = output[, lapply(.SD, function(x) summary(na.omit(x)))]

	output[, stat := c("Min", "Q1", "Median", "Mean", "Q3", "Max")]

	

	# change data types

	output[, acc := as.numeric(acc)]

	output[, sens := as.numeric(sens)]

	output[, spec := as.numeric(spec)]

	output[, auc := as.numeric(auc)]

	output[, or := as.numeric(or)]

	output[, stat := factor(stat, levels = c("Min", "Q1", "Median", "Mean", "Q3", "Max"))]

	

	return(output)

}



}



# -----------------------------------------------------------------------------------

# ---- Prepare Data -----------------------------------------------------------------

# -----------------------------------------------------------------------------------



{



# set work directory



getwd()

# setwd("F:/Documents/Working/Kaggle/Titanic")



# import the data



train = data.table(read.csv("train.csv", na.strings = ""))

test = data.table(read.csv("test.csv", na.strings = ""))



# for column descriptions see: https://www.kaggle.com/c/titanic/data



# ---- preparing train --------------------------------------------------------------



{



# lets check out train and update column data types



train

types(train)



# update columns that should be treated as factors, not numbers



train[, Survived := factor(Survived)]

train[, Pclass := factor(Pclass)]



# lets look at the levels of our factor columns



levels(train$Survived)

levels(train$Pclass)

levels(train$Name)

levels(train$Sex)

levels(train$Ticket)

levels(train$Cabin)

levels(train$Embarked)



# every PassengerId has their own Cabin

# lets aggregating Cabin by its letter (ie. drop the number) so we can actually consider using it as a factor variable



train[, Cabin := factor(gsub("[0-9]", "", Cabin))]



# lets also keep just one letter for each passenger

	# some passengers have mutliple cabins assigned to them

	# we'll just use the first cabin assignment



train[, Cabin := factor(gsub(" ", "", Cabin))]

train[, Cabin := factor(substring(Cabin, 1, 1))]



# verify that the levels of Cabin are just the letters



levels(train$Cabin)



# the level T looks weird, lets see how many times it appeared

	# its weird because it breaks the cabin lettering pattern by jumping from G to T



train[Cabin == "T"]



# it only appears once, so perhaps its a mistake

# lets set it to NA and impute a value later



train[Cabin == "T", Cabin := NA]

train[, Cabin := droplevels(Cabin)]



# lets check out if there are any missing values (NA's) in train



aggr(train, numbers = TRUE, sortVars = TRUE, gap = 3, cex.axis = 0.8)



# Cabin, Age, and Embarked have missing values

# let impute the missing values, but before perform feature selection for Cabin, Age, and Embarked to find out which combination of variable are best for imputing missing values



# ---- Cabin ------------------------------------------------------------------------



{



# create a copy of train



mod.dat = data.table(train)



# lets not include Name and Ticket becuase we cannot afford that many degrees of freedom (these are factors with a large number of levels)

# lets not include PassengerId because that is an ID variable for each row, so it won't be helpful



mod.dat[, c("Name", "Ticket", "PassengerId") := NULL]



# lets only keep the observations without missing values



mod.dat = na.omit(mod.dat)



# let look at the summary of mod.dat



summary(mod.dat)



# ---- removing highly correlated variables ----



# extract all potential regressors

cor.dat = data.table(mod.dat[,!"Cabin"])



# reformat all columns to be numeric by creating dummy variables for factor columns

cor.dat = data.table(model.matrix(~., cor.dat)[,-1])



# compute correlations and plot them

cors = cor(cor.dat)

corrplot(cors, type = "upper", diag = FALSE, mar = c(2, 2, 2, 2), addshade = "all")



# find out which regressors are highly correlated (>= 0.75) and remove them

find.dat = findCorrelation(cors, cutoff = 0.75, names = TRUE)

find.dat



# no need to remove any columns from mod.dat according to find.dat



# ---- ranking variables by importance ----



# lets use cross validation to train the random forest model that will determine the importance of each variable



# set up the cross validation mechanism

cv = trainControl(method = "repeatedcv", number = K, repeats = B)



# train our random forest model

mod = train(Cabin ~ ., data = mod.dat, method = "rf", trControl = cv, importance = TRUE)



# compute variable importance on a scale of 0 to 100

imp = varImp(mod, scale = TRUE)$importance



# transform imp into long format

imp = data.table(regressor = rownames(imp), imp / 100)

imp = melt(imp, id.vars = "regressor")



# make regressor a factor for plotting purposes

imp[, regressor := factor(regressor, levels = unique(regressor))]



# plot a barplot of regressor importance



ggplot(imp, aes(x = regressor, y = value, fill = value)) +

geom_bar(stat = "identity", position = "dodge") +

labs(x = "Regressor", y = "Importance") +

scale_y_continuous(labels = percent) +

scale_fill_gradient(low = "yellow", high = "red") +

facet_wrap(~variable) +

theme_dark(15) +

theme(legend.position = "none", axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))



# plot a barplot of mean regressor importance



imp.avg = imp[, .(value = mean(value)), by = regressor]



ggplot(imp.avg, aes(x = regressor, y = value, fill = value)) +

geom_bar(stat = "identity", position = "dodge") +

labs(x = "Regressor", y = "Importance") +

scale_y_continuous(labels = percent) +

scale_fill_gradient(low = "yellow", high = "red") +

theme_dark(25) +

theme(legend.position = "none")



# there is no clear indication that any variables should be removed



# ---- variable selection by recursive feature elimination ----



# lets use cross validation to train the random forest model that will select our variables



# set up the cross validation mechanism

cv = rfeControl(functions = rfFuncs, method = "repeatedcv", number = K, repeats = B)



# run the RFE algorithm

vars = rfe(Cabin ~ ., data = mod.dat, sizes = 1:10, metric = "Accuracy", maximize = TRUE, rfeControl = cv)

vars

vars$optVariables



# heres our regressors for cabin



cabin.regressors = c("Fare")



# remove objects we no longer need



rm(mod.dat, cors, cor.dat, find.dat, cv, imp, imp.avg, mod, vars)



}



# ---- Age --------------------------------------------------------------------------



{



# create a copy of train



mod.dat = data.table(train)



# lets not include Name and Ticket becuase we cannot afford that many degrees of freedom (these are factors with a large number of levels)

# lets not include PassengerId because that is an ID variable for each row, so it won't be helpful

# lets not include Cabin becuase of the large amount of missing values



mod.dat[, c("Name", "Ticket", "Cabin", "PassengerId") := NULL]



# lets only keep the observations without missing values



mod.dat = na.omit(mod.dat)



# let look at the summary of mod.dat



summary(mod.dat)



# lets take the log of Age because it is a positive variable and the log function only accepts positive values

# so this means we can impute the log(Age) and then transform the imputations back into the original Age units with exp()

# this will ensure our final imputations are positive



mod.dat[, log.Age := log(Age)]

mod.dat[, Age := NULL]



# ---- removing highly correlated variables ----



# extract all potential regressors

cor.dat = data.table(mod.dat[,!"log.Age"])



# reformat all columns to be numeric by creating dummy variables for factor columns

cor.dat = data.table(model.matrix(~., cor.dat)[,-1])



# compute correlations and plot them

cors = cor(cor.dat)

corrplot(cors, type = "upper", diag = FALSE, mar = c(2, 2, 2, 2), addshade = "all")



# find out which regressors are highly correlated (>= 0.75) and remove them

find.dat = findCorrelation(cors, cutoff = 0.75, names = TRUE)

find.dat



# no need to remove any columns from mod.dat according to find.dat



# ---- ranking variables by importance ----



# lets use cross validation to train the random forest model that will determine the importance of each variable



# set up the cross validation mechanism

cv = trainControl(method = "repeatedcv", number = K, repeats = B)



# train our random forest model

mod = train(log.Age ~ ., data = mod.dat, method = "rf", trControl = cv, importance = TRUE)



# compute variable importance on a scale of 0 to 100

imp = varImp(mod, scale = TRUE)$importance



# transform imp into a table

imp = data.table(regressor = rownames(imp), imp / 100)



# make regressor a factor for plotting purposes

imp[, regressor := factor(regressor, levels = unique(regressor))]



# plot a barplot of regressor importance



ggplot(imp, aes(x = regressor, y = Overall, fill = Overall)) +

geom_bar(stat = "identity", position = "dodge") +

labs(x = "Regressor", y = "Importance") +

scale_y_continuous(labels = percent) +

scale_fill_gradient(low = "yellow", high = "red") +

theme_dark(25) +

theme(legend.position = "none")



# based on this plots we can see that Embarked is the regressor that does not have enough importance

# lets remove Embarked



mod.dat[, Embarked := NULL]



# ---- variable selection by recursive feature elimination ----



# lets use cross validation to train the random forest model that will select our variables



# set up the cross validation mechanism

cv = rfeControl(functions = rfFuncs, method = "repeatedcv", number = K, repeats = B)



# run the RFE algorithm

vars = rfe(log.Age ~ ., data = mod.dat, sizes = 1:7, metric = "RMSE", maximize = FALSE, rfeControl = cv)

vars

vars$optVariables



# heres our regressors for log.age



log.age.regressors = c("Parch", "SibSp", "Survived", "Pclass", "Fare", "Sex")



# remove objects we no longer need



rm(mod.dat, cors, cor.dat, find.dat, cv, imp, mod, vars)



}



# ---- Embarked ---------------------------------------------------------------------



{



# create a copy of train



mod.dat = data.table(train)



# lets not include Name and Ticket becuase we cannot afford that many degrees of freedom (these are factors with a large number of levels)

# lets not include PassengerId because that is an ID variable for each row, so it won't be helpful

# lets not include Cabin becuase of the large amount of missing values



mod.dat[, c("Name", "Ticket", "Cabin", "PassengerId") := NULL]



# lets only keep the observations without missing values



mod.dat = na.omit(mod.dat)



# let look at the summary of mod.dat



summary(mod.dat)



# ---- removing highly correlated variables ----



# extract all potential regressors

cor.dat = data.table(mod.dat[,!"Embarked"])



# reformat all columns to be numeric by creating dummy variables for factor columns

cor.dat = data.table(model.matrix(~., cor.dat)[,-1])



# compute correlations and plot them

cors = cor(cor.dat)

corrplot(cors, type = "upper", diag = FALSE, mar = c(2, 2, 2, 2), addshade = "all")



# find out which regressors are highly correlated (>= 0.75) and remove them

find.dat = findCorrelation(cors, cutoff = 0.75, names = TRUE)

find.dat



# no need to remove any columns from mod.dat according to find.dat



# ---- ranking variables by importance ----



# lets use cross validation to train the random forest model that will determine the importance of each variable



# set up the cross validation mechanism

cv = trainControl(method = "repeatedcv", number = K, repeats = B)



# train our random forest model

mod = train(Embarked ~ ., data = mod.dat, method = "rf", trControl = cv, importance = TRUE)



# compute variable importance on a scale of 0 to 100

imp = varImp(mod, scale = TRUE)$importance



# transform imp into long format

imp = data.table(regressor = rownames(imp), imp / 100)

imp = melt(imp, id.vars = "regressor")



# make regressor a factor for plotting purposes

imp[, regressor := factor(regressor, levels = unique(regressor))]



# plot a barplot of regressor importance



ggplot(imp, aes(x = regressor, y = value, fill = value)) +

geom_bar(stat = "identity", position = "dodge") +

labs(x = "Regressor", y = "Importance") +

scale_y_continuous(labels = percent) +

scale_fill_gradient(low = "yellow", high = "red") +

facet_wrap(~variable) +

theme_dark(15) +

theme(legend.position = "none", axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))



# plot a barplot of mean regressor importance



imp.avg = imp[, .(value = mean(value)), by = regressor]



ggplot(imp.avg, aes(x = regressor, y = value, fill = value)) +

geom_bar(stat = "identity", position = "dodge") +

labs(x = "Regressor", y = "Importance") +

scale_y_continuous(labels = percent) +

scale_fill_gradient(low = "yellow", high = "red") +

theme_dark(25) +

theme(legend.position = "none")



# there is no clear indication that any variables should be removed



# ---- variable selection by recursive feature elimination ----



# lets use cross validation to train the random forest model that will select our variables



# set up the cross validation mechanism

cv = rfeControl(functions = rfFuncs, method = "repeatedcv", number = K, repeats = B)



# run the RFE algorithm

vars = rfe(Embarked ~ ., data = mod.dat, sizes = 1:8, metric = "Accuracy", maximize = TRUE, rfeControl = cv)

vars

vars$optVariables



# heres our regressors for embarked



embark.regressors = c("Fare")



# remove objects we no longer need



rm(mod.dat, cors, cor.dat, find.dat, cv, imp, imp.avg, mod, vars)



}



# ---- Imputations ------------------------------------------------------------------



{



# lets create a copy of train



train.mice = data.table(train)



# create log.Age



train.mice[, log.Age := log(Age)]



# lets create our predictor matrix that indicates what regressors to use

# quickpred() sets up the default predictor matrix, then we can update it to what we want



my.pred = quickpred(train.mice)

my.pred



# make adjustments



my.pred["Cabin",] = as.numeric(colnames(my.pred) %in% cabin.regressors)

my.pred["Age",] = 0

my.pred["log.Age",] = as.numeric(colnames(my.pred) %in% log.age.regressors)

my.pred["Embarked",] = as.numeric(colnames(my.pred) %in% embark.regressors)



# verify adjustments



my.pred



# compute imputations

	# method = "fastpmm" uses predictive means matching to impute missing values (it's "fast" becuase it uses C++)



imputations = mice(data = train.mice, predictorMatrix = my.pred, method = "fastpmm", seed = 42)



# lets verify that the imputed cabin values follow the same relationships as the known cabin values

	# imputed values are orange

	# known values are blue



xyplot(x = imputations, 

			data = Cabin ~ Fare, 

			alpha = 1/3, pch = 20, cex = 1, col = c("cornflowerblue", "tomato"),

			jitter.x = TRUE, jitter.y = TRUE, factor = 1,

			scales = list(x = list(relation = "free")))



# lets verify that the imputed log.age values follow the same relationships as the known age values

	# imputed values are orange

	# known values are blue



xyplot(x = imputations, 

			data = log.Age ~ Parch + SibSp + Survived + Pclass + Fare + Sex, 

			alpha = 1/3, pch = 20, cex = 1, col = c("cornflowerblue", "tomato"),

			jitter.x = TRUE, jitter.y = TRUE, factor = 1,

			scales = list(x = list(relation = "free")))



# lets verify that the imputed embarked values follow the same relationships as the known embarked values

	# imputed values are orange

	# known values are blue



xyplot(x = imputations, 

			data = Embarked ~ Fare, 

			alpha = 1/3, pch = 20, cex = 1, col = c("cornflowerblue", "tomato"),

			jitter.x = TRUE, jitter.y = TRUE, factor = 2,

			scales = list(x = list(relation = "free")))

			

# the spread for emabarked imputed values seems to be centered right at the mean of Fare which makes sense given the pmm method

# normally this would be concerning but given that there are very few values of Embarked to impute, this relationship will suffice without much impact



# lets extract all of the imputed data sets



train.mice = data.table(complete(imputations, action = "long"))



# lets only keep Cabin, log.Age, Embarked, and PassengerId 

# we are keeping PassengerId so we can aggregate Cabin, log.Age and Embarked across all imputations



train.mice = train.mice[, .(PassengerId, Cabin, log.Age, Embarked)]



# lets transform log.Age back into its original units



train.mice[, Age := round(exp(log.Age), 2)]



# lets take the most frequent Cabin value across all imputations

# lets average Age across all imputations

# lets take the most frequent Embarked value across all imputations



train.mice = train.mice[, .(Cabin = names(sort(table(Cabin), decreasing = TRUE)[1]),

							Age = round(mean(Age), 2),

							Embarked = names(sort(table(Embarked), decreasing = TRUE)[1])), 

						by = PassengerId]



# lets convert Cabin and Embarked back into factor data types



train.mice[, Cabin := factor(Cabin)]

train.mice[, Embarked := factor(Embarked)]



# replace Cabin, Age, and Embarked in train with Cabin, Age, and Embarked in train.mice



train[, Cabin := train.mice$Cabin]

train[, Age := train.mice$Age]

train[, Embarked := train.mice$Embarked]



# lets remove objects we no longer need



rm(imputations, my.pred, cabin.regressors, log.age.regressors, embark.regressors, train.mice)



# lets verify that there are no missing values (NA's) in train



aggr(train, numbers = TRUE, sortVars = TRUE, gap = 3, cex.axis = 0.8)



}



}



# ---- preparing test ---------------------------------------------------------------



{



# lets check out test and update column data types



test

types(test)



# update columns that should be treated as factors, not numbers



test[, Pclass := factor(Pclass)]



# lets look at the levels of our factor columns



levels(test$Pclass)

levels(test$Name)

levels(test$Sex)

levels(test$Ticket)

levels(test$Cabin)

levels(test$Embarked)



# every PassengerId has their own Cabin

# lets see if Cabin is useful by aggregating it by on its letter (ie. drop the number)



test[, Cabin := factor(gsub("[0-9]", "", Cabin))]



# lets also keep just one letter for each passenger



test[, Cabin := factor(gsub(" ", "", Cabin))]

test[, Cabin := factor(substring(Cabin, 1, 1))]



# verify that the levels of Cabin are just the letters



levels(test$Cabin)



# lets check out if there are any missing values (NA's) in test



aggr(test, numbers = TRUE, sortVars = TRUE, gap = 3, cex.axis = 0.8)



# Cabin, Age, and Fare have missing values

# let impute the missing data, but before doing that lets find a good combination of regressors that explain Cabin, Age, and Fare



# ---- Cabin ------------------------------------------------------------------------



{



# create a copy of test



mod.dat = data.table(test)



# lets not include Name and Ticket becuase we cannot afford that many degrees of freedom (these are factors with a large number of levels)

# lets not include PassengerId because that is an ID variable for each row, so it won't be helpful



mod.dat[, c("Name", "Ticket", "PassengerId") := NULL]



# lets only keep the observations without missing values



mod.dat = na.omit(mod.dat)



# let look at the summary of mod.dat



summary(mod.dat)



# theres only 1 instance of Cabin == G and 1 instance of Embarked == Q

# given these infrequencies lets remove them as levels for the sake of choosing regressors



mod.dat = mod.dat[!(Cabin == "G" | Embarked == "Q")]

mod.dat[, Cabin := droplevels(Cabin)]

mod.dat[, Embarked := droplevels(Embarked)]



# ---- removing highly correlated variables ----



# extract all potential regressors

cor.dat = data.table(mod.dat[,!"Cabin"])



# reformat all columns to be numeric by creating dummy variables for factor columns

cor.dat = data.table(model.matrix(~., cor.dat)[,-1])



# compute correlations and plot them

cors = cor(cor.dat)

corrplot(cors, type = "upper", diag = FALSE, mar = c(2, 2, 2, 2), addshade = "all")



# find out which regressors are highly correlated (>= 0.75) and remove them

find.dat = findCorrelation(cors, cutoff = 0.75, names = TRUE)

find.dat



# no need to remove any columns from mod.dat according to find.dat



# ---- ranking variables by importance ----



# lets use cross validation to train the random forest model that will determine the importance of each variable



# set up the cross validation mechanism

cv = trainControl(method = "repeatedcv", number = K, repeats = B)



# train our random forest model

mod = train(Cabin ~ ., data = mod.dat, method = "rf", trControl = cv, importance = TRUE)



# compute variable importance on a scale of 0 to 100

imp = varImp(mod, scale = TRUE)$importance



# transform imp into long format

imp = data.table(regressor = rownames(imp), imp / 100)

imp = melt(imp, id.vars = "regressor")



# make regressor a factor for plotting purposes

imp[, regressor := factor(regressor, levels = unique(regressor))]



# plot a barplot of regressor importance



ggplot(imp, aes(x = regressor, y = value, fill = value)) +

geom_bar(stat = "identity", position = "dodge") +

labs(x = "Regressor", y = "Importance") +

scale_y_continuous(labels = percent) +

scale_fill_gradient(low = "yellow", high = "red") +

facet_wrap(~variable) +

theme_dark(15) +

theme(legend.position = "none", axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))



# plot a barplot of mean regressor importance



imp.avg = imp[, .(value = mean(value)), by = regressor]



ggplot(imp.avg, aes(x = regressor, y = value, fill = value)) +

geom_bar(stat = "identity", position = "dodge") +

labs(x = "Regressor", y = "Importance") +

scale_y_continuous(labels = percent) +

scale_fill_gradient(low = "yellow", high = "red") +

theme_dark(25) +

theme(legend.position = "none")



# based on these plots we can see that Sex doesn't seem important enough

# lets remove Sex



mod.dat[, Sex := NULL]



# ---- variable selection by recursive feature elimination ----



# lets use cross validation to train the random forest model that will select our variables



# set up the cross validation mechanism

cv = rfeControl(functions = rfFuncs, method = "repeatedcv", number = K, repeats = B)



# run the RFE algorithm

vars = rfe(Cabin ~ ., data = mod.dat, sizes = 1:7, metric = "Accuracy", maximize = TRUE, rfeControl = cv)

vars

vars$optVariables



# heres our regressors for cabin



cabin.regressors = c("Fare")



# remove objects we no longer need



rm(mod.dat, cors, cor.dat, find.dat, cv, imp, imp.avg, mod, vars)



}



# ---- Age --------------------------------------------------------------------------



{



# create a copy of test



mod.dat = data.table(test)



# lets not include Name and Ticket becuase we cannot afford that many degrees of freedom (these are factors with a large number of levels)

# lets not include PassengerId because that is an ID variable for each row, so it won't be helpful

# lets not include Cabin becuase of the large amount of missing values



mod.dat[, c("Name", "Ticket", "Cabin", "PassengerId") := NULL]



# lets only keep the observations without missing values



mod.dat = na.omit(mod.dat)



# let look at the summary of mod.dat



summary(mod.dat)



# lets take the log of Age because it is a positive variable and the log function only accepts positive values

# so this means we can impute the log(Age) and then transform the imputations back into the original Age units with exp()

# this will ensure our final imputations are positive



mod.dat[, log.Age := log(Age)]

mod.dat[, Age := NULL]



# ---- removing highly correlated variables ----



# extract all potential regressors

cor.dat = data.table(mod.dat[,!"log.Age"])



# reformat all columns to be numeric by creating dummy variables for factor columns

cor.dat = data.table(model.matrix(~., cor.dat)[,-1])



# compute correlations and plot them

cors = cor(cor.dat)

corrplot(cors, type = "upper", diag = FALSE, mar = c(2, 2, 2, 2), addshade = "all")



# find out which regressors are highly correlated (>= 0.75) and remove them

find.dat = findCorrelation(cors, cutoff = 0.75, names = TRUE)

find.dat



# no need to remove any columns from mod.dat according to find.dat



# ---- ranking variables by importance ----



# lets use cross validation to train the random forest model that will determine the importance of each variable



# set up the cross validation mechanism

cv = trainControl(method = "repeatedcv", number = K, repeats = B)



# train our random forest model

mod = train(log.Age ~ ., data = mod.dat, method = "rf", trControl = cv, importance = TRUE)



# compute variable importance on a scale of 0 to 100

imp = varImp(mod, scale = TRUE)$importance



# transform imp into a table

imp = data.table(regressor = rownames(imp), imp / 100)



# make regressor a factor for plotting purposes

imp[, regressor := factor(regressor, levels = unique(regressor))]



# plot a barplot of regressor importance



ggplot(imp, aes(x = regressor, y = Overall, fill = Overall)) +

geom_bar(stat = "identity", position = "dodge") +

labs(x = "Regressor", y = "Importance") +

scale_y_continuous(labels = percent) +

scale_fill_gradient(low = "yellow", high = "red") +

theme_dark(25) +

theme(legend.position = "none")



# based on this plots we can see that Sex and Embarked are the regressor that does not have enough importance

# SibSp doesn't look that good ethier but lets leave it up to recursive feature elimination to decide

# lets remove Embarked



mod.dat[, c("Sex", "Embarked") := NULL]



# ---- variable selection by recursive feature elimination ----



# lets use cross validation to train the random forest model that will select our variables



# set up the cross validation mechanism

cv = rfeControl(functions = rfFuncs, method = "repeatedcv", number = K, repeats = B)



# run the RFE algorithm

vars = rfe(log.Age ~ ., data = mod.dat, sizes = 1:5, metric = "RMSE", maximize = FALSE, rfeControl = cv)

vars

vars$optVariables



# heres our regressors for log.age



log.age.regressors = c("Pclass", "Parch", "Fare", "SibSp")



# remove objects we no longer need



rm(mod.dat, cors, cor.dat, find.dat, cv, imp, mod, vars)



}



# ---- Fare --------------------------------------------------------------------------



{



# create a copy of test



mod.dat = data.table(test)



# lets not include Name and Ticket becuase we cannot afford that many degrees of freedom (these are factors with a large number of levels)

# lets not include PassengerId because that is an ID variable for each row, so it won't be helpful

# lets not include Cabin becuase of the large amount of missing values



mod.dat[, c("Name", "Ticket", "Cabin", "PassengerId") := NULL]



# lets only keep the observations without missing values



mod.dat = na.omit(mod.dat)



# let look at the summary of mod.dat



summary(mod.dat)



# lets take the log of (Fare + 1) because it is a positive variable and the log function only accepts positive values

# so this means we can impute the log(Fare + 1) and then transform the imputations back into the original Fare units with exp()

# this will ensure our final imputations are positive

# the reason we are adding 1 to Fare is becuase it has a minimum value of 0, and log can only accept positive values



mod.dat[, log.Fare.1 := log(Fare + 1)]

mod.dat[, Fare := NULL]



# ---- removing highly correlated variables ----



# extract all potential regressors

cor.dat = data.table(mod.dat[,!"log.Fare.1"])



# reformat all columns to be numeric by creating dummy variables for factor columns

cor.dat = data.table(model.matrix(~., cor.dat)[,-1])



# compute correlations and plot them

cors = cor(cor.dat)

corrplot(cors, type = "upper", diag = FALSE, mar = c(2, 2, 2, 2), addshade = "all")



# find out which regressors are highly correlated (>= 0.75) and remove them

find.dat = findCorrelation(cors, cutoff = 0.75, names = TRUE)

find.dat



# no need to remove any columns from mod.dat according to find.dat



# ---- ranking variables by importance ----



# lets use cross validation to train the random forest model that will determine the importance of each variable



# set up the cross validation mechanism

cv = trainControl(method = "repeatedcv", number = K, repeats = B)



# train our random forest model

mod = train(log.Fare.1 ~ ., data = mod.dat, method = "rf", trControl = cv, importance = TRUE)



# compute variable importance on a scale of 0 to 100

imp = varImp(mod, scale = TRUE)$importance



# transform imp into a table

imp = data.table(regressor = rownames(imp), imp / 100)



# make regressor a factor for plotting purposes

imp[, regressor := factor(regressor, levels = unique(regressor))]



# plot a barplot of regressor importance



ggplot(imp, aes(x = regressor, y = Overall, fill = Overall)) +

geom_bar(stat = "identity", position = "dodge") +

labs(x = "Regressor", y = "Importance") +

scale_y_continuous(labels = percent) +

scale_fill_gradient(low = "yellow", high = "red") +

theme_dark(25) +

theme(legend.position = "none")



# based on this plots we can see that Sex, Age, and Embarked are the regressors that do not have enough importance

# lets remove Sex, Age, and Embarked



mod.dat[, c("Sex", "Age", "Embarked") := NULL]



# ---- variable selection by recursive feature elimination ----



# lets use cross validation to train the random forest model that will select our variables



# set up the cross validation mechanism

cv = rfeControl(functions = rfFuncs, method = "repeatedcv", number = K, repeats = B)



# run the RFE algorithm

vars = rfe(log.Fare.1 ~ ., data = mod.dat, sizes = 1:4, metric = "RMSE", maximize = FALSE, rfeControl = cv)

vars

vars$optVariables



# heres our regressors for log.fare.1



log.fare.1.regressors = c("Pclass", "SibSp", "Parch")



# remove objects we no longer need



rm(mod.dat, cors, cor.dat, find.dat, cv, imp, mod, vars)



}



# ---- Imputations ------------------------------------------------------------------



{



# lets create a copy of test



test.mice = data.table(test)



# we will need to do log transformations on Age, and Fare



test.mice[, log.Age := log(Age)]

test.mice[, log.Fare.1 := log(Fare + 1)]



# lets create our predictor matrix that indicates what regressors to use

# quickpred() sets up the default predictor matrix, then we can update it to what we want



my.pred = quickpred(test.mice)

my.pred



# make adjustments



my.pred["Cabin",] = as.numeric(colnames(my.pred) %in% cabin.regressors)

my.pred["log.Age",] = as.numeric(colnames(my.pred) %in% log.age.regressors)

my.pred["Age",] = 0

my.pred["log.Fare.1",] = as.numeric(colnames(my.pred) %in% log.fare.1.regressors)

my.pred["Fare",] = 0



# verify adjustments



my.pred



# compute imputations

	# m denotes the number of imputations run

	# method = "fastpmm" is the predictive mean matching method which works for any data type



imputations = mice(data = test.mice, predictorMatrix = my.pred, method = "fastpmm", seed = 42)



# lets verify that the imputed cabin values follow the same relationships as the known cabin values

	# imputed values are orange

	# known values are blue



xyplot(x = imputations, 

			data = Cabin ~ Fare, 

			alpha = 1/3, pch = 20, cex = 1, col = c("cornflowerblue", "tomato"),

			jitter.x = TRUE, jitter.y = TRUE, factor = 1,

			scales = list(x = list(relation = "free")))



# lets verify that the imputed age values follow the same relationships as the known age values

	# imputed values are orange

	# known values are blue



xyplot(x = imputations, 

			data = log.Age ~ Pclass + Parch + Fare + SibSp, 

			alpha = 1/3, pch = 20, cex = 1, col = c("cornflowerblue", "tomato"),

			jitter.x = TRUE, jitter.y = TRUE, factor = 1,

			scales = list(x = list(relation = "free")))



# lets verify that the imputed Fare values follow the same relationships as the known Fare values

	# imputed values are orange

	# known values are blue



xyplot(x = imputations, 

			data = log.Fare.1 ~ Pclass + SibSp + Parch, 

			alpha = 1/3, pch = 20, cex = 1, col = c("cornflowerblue", "tomato"),

			jitter.x = TRUE, jitter.y = TRUE, factor = 2,

			scales = list(x = list(relation = "free")))



# lets extract all of the imputed data sets



test.mice = data.table(complete(imputations, action = "long"))



# lets only keep Cabin, log.Age, log.Fare.1, and PassengerId 

# we are keeping PassengerId so we can aggregate Cabin, Age and Fare across all imputations



test.mice = test.mice[, .(PassengerId, Cabin, log.Age, log.Fare.1)]



# lets transform Age and Fare back into its original units



test.mice[, Age := round(exp(log.Age), 2)]

test.mice[, Fare := round(exp(log.Fare.1), 4) - 1]



# lets take the most frequent Cabin value across all imputations

# lets average Age across all imputations

# lets average Fare across all imputations



test.mice = test.mice[, .(Cabin = names(sort(table(Cabin), decreasing = TRUE)[1]),

							Age = round(mean(Age), 2),

							Fare = round(mean(Fare), 4)), 

						by = PassengerId]



# lets convert Cabin and Fare back into factor data types



test.mice[, Cabin := factor(Cabin)]



# replace Cabin, Age, and Fare in test with Cabin, Age, and Fare in test.mice



test[, Cabin := test.mice$Cabin]

test[, Age := test.mice$Age]

test[, Fare := test.mice$Fare]



# lets remove objects we no longer need



rm(imputations, my.pred, cabin.regressors, log.age.regressors, log.fare.1.regressors, test.mice)



# lets verify that there are no missing values (NA's) in test



aggr(test, numbers = TRUE, sortVars = TRUE, gap = 3, cex.axis = 0.8)



}



}



}



# -----------------------------------------------------------------------------------

# ---- Plot Data --------------------------------------------------------------------

# -----------------------------------------------------------------------------------



{



# ---- Transformations --------------------------------------------------------------



{



# lets produce boxplots of the numeric variables across the reponse variable Survived

# lets also run t.tests to see if there is a significant difference in means of the numeric variables across the reponse variable Survived

# we are looking to see if transformations on the numeric variables will create better seperation across Survived



# ---- Age --------------------------------------------------------------------------



p1 = ggplot(train, aes(x = Survived, y = Age, fill = Survived)) + 

		geom_boxplot(width = 0.5) +

		theme_bw(base_size = 30) +

		theme(legend.position = "none", plot.title = element_text(hjust = 0.5))



p2 = ggplot(train, aes(x = Survived, y = log(Age), fill = Survived)) + 

		geom_boxplot(width = 0.5) +

		theme_bw(base_size = 30) +

		theme(legend.position = "none", plot.title = element_text(hjust = 0.5))



# plot

grid.arrange(p1, p2, nrow = 1)



# test

t.test(Age ~ Survived, data = train)

t.test(log(Age) ~ Survived, data = train)



# Age without transformation already provides significant seperation



# ---- SibSp ------------------------------------------------------------------------



p1 = ggplot(train, aes(x = Survived, y = SibSp, fill = Survived)) + 

		geom_boxplot(width = 0.5) +

		theme_bw(base_size = 30) +

		theme(legend.position = "none", plot.title = element_text(hjust = 0.5))



p2 = ggplot(train, aes(x = Survived, y = SibSp^2, fill = Survived)) + 

		geom_boxplot(width = 0.5) +

		theme_bw(base_size = 30) +

		theme(legend.position = "none", plot.title = element_text(hjust = 0.5))



# plot

grid.arrange(p1, p2, nrow = 1)



# test

t.test(SibSp ~ Survived, data = train)

t.test(SibSp^2 ~ Survived, data = train)



# apply squared transformation

train[, SibSp.2 := SibSp^2]



# ---- Parch ------------------------------------------------------------------------



p1 = ggplot(train, aes(x = Survived, y = Parch, fill = Survived)) + 

		geom_boxplot(width = 0.5) +

		theme_bw(base_size = 30) +

		theme(legend.position = "none", plot.title = element_text(hjust = 0.5))



p2 = ggplot(train, aes(x = Survived, y = log(Parch + 1), fill = Survived)) + 

		geom_boxplot(width = 0.5) +

		theme_bw(base_size = 30) +

		theme(legend.position = "none", plot.title = element_text(hjust = 0.5))



# plot

grid.arrange(p1, p2, nrow = 1)



# test

t.test(Parch ~ Survived, data = train)

t.test(log(Parch + 1) ~ Survived, data = train)



# Parch without a transformation already provides significant seperation



# ---- Fare -------------------------------------------------------------------------



p1 = ggplot(train, aes(x = Survived, y = Fare, fill = Survived)) + 

		geom_boxplot(width = 0.5) +

		theme_bw(base_size = 30) +

		theme(legend.position = "none", plot.title = element_text(hjust = 0.5))



p2 = ggplot(train, aes(x = Survived, y = log(Fare + 1), fill = Survived)) + 

		geom_boxplot(width = 0.5) +

		theme_bw(base_size = 30) +

		theme(legend.position = "none", plot.title = element_text(hjust = 0.5))



# plot

grid.arrange(p1, p2, nrow = 1)



# test

t.test(Fare ~ Survived, data = train)

t.test(log(Fare + 1) ~ Survived, data = train)



# Fare without a transformation already provides significant seperation



# remove objects we no longer need



rm(p1, p2)



}



# ---- Interactions -----------------------------------------------------------------



{



# ---- Importance -------------------------------------------------------------------



{



# create a copy of train



mod.dat = data.table(train)



# lets not include Name and Ticket becuase we cannot afford that many degrees of freedom (these are factors with a large number of levels)

# lets not include PassengerId because that is an ID variable for each row, so it won't be helpful



mod.dat[, c("Name", "Ticket", "PassengerId") := NULL]



# ---- removing highly correlated variables ----



# extract all potential regressors

cor.dat = data.table(mod.dat[,!"Survived"])



# reformat all columns to be numeric by creating dummy variables for factor columns

cor.dat = data.table(model.matrix(~.^2, cor.dat)[,-c(1:17)])

# redifine mod.dat with just the two way interactions

mod.dat = cbind(Survived = mod.dat$Survived, cor.dat)



# compute correlations

cors = cor(cor.dat)

# make all NA correlations equal to 1

cors[is.na(cors)] = 1

# plot correlations

corrplot(cors, type = "upper", diag = FALSE, mar = c(2, 2, 2, 2), addshade = "all")



# find out which regressors are highly correlated (>= 0.75) and remove them

find.dat = findCorrelation(cors, cutoff = 0.75, names = TRUE)

find.dat



# remove columns from mod.dat according to find.dat

mod.dat = mod.dat[, !find.dat, with = FALSE]



# ---- ranking variables by importance ----



# lets use cross validation to train the random forest model that will determine the importance of each variable



# set up the cross validation mechanism

cv = trainControl(method = "repeatedcv", number = K, repeats = B)



# train our random forest model

mod = train(Survived ~ ., data = mod.dat, method = "rf", trControl = cv, importance = TRUE)



# compute variable importance on a scale of 0 to 100

imp = varImp(mod, scale = TRUE)$importance



# transform imp into long format

imp = data.table(regressor = rownames(imp), imp / 100)

imp = melt(imp, id.vars = "regressor")



# make regressor a factor for plotting purposes

imp[, regressor := factor(regressor, levels = unique(regressor))]



# plot a barplot of regressor importance



ggplot(imp, aes(x = regressor, y = value, fill = value)) +

geom_bar(stat = "identity", position = "dodge") +

labs(x = "Regressor", y = "Importance") +

scale_y_continuous(labels = percent) +

scale_fill_gradient(low = "yellow", high = "red") +

facet_wrap(~variable) +

theme_dark(15) +

theme(legend.position = "none", axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))



# plot a barplot of mean regressor importance



imp.avg = imp[, .(value = mean(value)), by = regressor]



ggplot(imp.avg, aes(x = regressor, y = value, fill = value)) +

geom_bar(stat = "identity", position = "dodge") +

labs(x = "Regressor", y = "Importance") +

scale_y_continuous(labels = percent) +

scale_fill_gradient(low = "yellow", high = "red") +

theme_dark(15) +

theme(legend.position = "none", axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))



# only keep variables with an importance greater than 40%



keep.dat = c("Survived", gsub("`", "", imp.avg[value >= 0.4, regressor]))

mod.dat = mod.dat[, keep.dat, with = FALSE]



# ---- variable selection by recursive feature elimination ----



# lets use cross validation to train the random forest model that will select our variables



# set up the cross validation mechanism

cv = rfeControl(functions = rfFuncs, method = "repeatedcv", number = K, repeats = B)



# run the RFE algorithm

vars = rfe(Survived ~ ., data = mod.dat, sizes = 1:(ncol(mod.dat) - 1), metric = "Accuracy", maximize = TRUE, rfeControl = cv)

vars

vars$optVariables



# lets go with the 8 variable model becuase it has significantly less regressors with similar performance



selectedVars = vars$variables

bestVar = vars$control$functions$selectVar(selectedVars, 8)

bestVar



# here's our interactions



survived.interactions = c("Sexmale:Age", "Pclass3:Age", "Sexmale:Fare", "Pclass3:Fare", "Age:Fare", "Pclass3:EmbarkedS", "Sexmale:EmbarkedS", "Fare:EmbarkedS")



# remove objects we no longer need



rm(mod.dat, cors, cor.dat, find.dat, cv, imp, imp.avg, mod, vars, bestVar, keep.dat, selectedVars)



}



# ---- Plots ------------------------------------------------------------------------



{



# lets produce jitter plots of interactions of interest



survived.interactions



# Sexmale:Age



ggplot(train, aes(x = Sex, y = cut(Age, breaks = hist(Age)$breaks), color = Survived)) + 

geom_jitter(alpha = 1/2, width = 1/4) +

facet_wrap(~Survived) +

theme_bw(base_size = 30) +

theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) +

ylab("Age")



train[, Sexmale.Age := model.matrix(~Sex:Age, data = train)[,3]]



# Pclass3:Age



ggplot(train, aes(x = Pclass, y = cut(Age, breaks = hist(Age)$breaks), color = Survived)) + 

geom_jitter(alpha = 1/2, width = 1/4) +

facet_wrap(~Survived) +

theme_bw(base_size = 30) +

theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) +

ylab("Age")



train[, Pclass3.Age := model.matrix(~Pclass:Age, data = train)[,4]]



# Sexmale:Fare



ggplot(train, aes(x = Sex, y = cut(Fare, breaks = c(-1e-10, hist(Fare)$breaks[-1])), color = Survived)) + 

geom_jitter(alpha = 1/2, width = 1/4) +

facet_wrap(~Survived) +

theme_bw(base_size = 30) +

theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) +

ylab("Fare")



train[, Sexmale.Fare := model.matrix(~Sex:Fare, data = train)[,3]]



# Pclass3:Fare



ggplot(train, aes(x = Pclass, y = cut(Fare, breaks = c(-1e-10, hist(Fare)$breaks[-1])), color = Survived)) + 

geom_jitter(alpha = 1/2, width = 1/4) +

facet_wrap(~Survived) +

theme_bw(base_size = 30) +

theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) +

ylab("Fare")



train[, Pclass3.Fare := model.matrix(~Pclass:Fare, data = train)[,4]]



# Age:Fare



ggplot(train, aes(x = cut(Age, breaks = hist(Age)$breaks), y = cut(Fare, breaks = c(-1e-10, hist(Fare)$breaks[-1])), color = Survived)) + 

geom_jitter(alpha = 1/2, width = 1/4) +

facet_wrap(~Survived) +

theme_bw(base_size = 20) +

theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) +

labs(x = "Age", y = "Fare")



# this doesn't appear as helpful

# we won't include this interaction



# Pclass3:EmbarkedS



ggplot(train, aes(x = Pclass, y = Embarked, color = Survived)) + 

geom_jitter(alpha = 1/2, width = 1/4) +

facet_wrap(~Survived) +

theme_bw(base_size = 30) +

theme(legend.position = "none", plot.title = element_text(hjust = 0.5))



train[, Pclass3.EmbarkedS := model.matrix(~Pclass:Embarked, data = train)[,10]]



# Sexmale:EmbarkedS



ggplot(train, aes(x = Sex, y = Embarked, color = Survived)) + 

geom_jitter(alpha = 1/2, width = 1/4) +

facet_wrap(~Survived) +

theme_bw(base_size = 30) +

theme(legend.position = "none", plot.title = element_text(hjust = 0.5))



train[, Sexmale.EmbarkedS := model.matrix(~Sex:Embarked, data = train)[,7]]



# Fare:EmbarkedS



ggplot(train, aes(x = cut(Fare, breaks = c(-1e-10, hist(Fare)$breaks[-1])), y = Embarked, color = Survived)) + 

geom_jitter(alpha = 1/2, width = 1/4) +

facet_wrap(~Survived) +

theme_bw(base_size = 20) +

theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) +

xlab("Fare")



# this doesn't appear as helpful

# we won't include this interaction



# remove interaction object



rm(survived.interactions)



}



}



}



# -----------------------------------------------------------------------------------

# ---- Variable Selection -----------------------------------------------------------

# -----------------------------------------------------------------------------------



{



# build a copy of train for modeling



mod.dat = data.table(train)



# lets not include Name and Ticket becuase we cannot afford that many degrees of freedom (these are factors with a large number of levels)

# lets not include PassengerId because that is an ID variable for each row, so it won't be helpful



mod.dat[, c("Name", "Ticket", "PassengerId") := NULL]



# ---- removing highly correlated variables ----



# extract all potential regressors

cor.dat = data.table(mod.dat[,!"Survived"])



# reformat all columns to be numeric by creating dummy variables for factor columns

cor.dat = data.table(model.matrix(~., cor.dat)[,-1])



# compute correlations and plot them

cors = cor(cor.dat)

corrplot(cors, type = "upper", diag = FALSE, mar = c(2, 2, 2, 2), addshade = "all")



# find out which regressors are highly correlated (>= 0.75) and remove them

find.dat = findCorrelation(cors, cutoff = 0.75, names = TRUE)

find.dat



# remove columns from mod.dat according to find.dat



mod.dat = cbind(Survived = mod.dat$Survived, cor.dat)

mod.dat[, c("Pclass3", "Sexmale.Age", "SibSp") := NULL]



# ---- ranking variables by importance ----



# lets use cross validation to train the random forest model that will determine the importance of each variable



# set up the cross validation mechanism

cv = trainControl(method = "repeatedcv", number = K, repeats = B)



# train our random forest model

mod = train(Survived ~ ., data = mod.dat, method = "rf", trControl = cv, importance = TRUE)



# compute variable importance on a scale of 0 to 100

imp = varImp(mod, scale = TRUE)$importance



# transform imp into long format

imp = data.table(regressor = rownames(imp), imp / 100)

imp = melt(imp, id.vars = "regressor")



# make regressor a factor for plotting purposes

imp[, regressor := factor(regressor, levels = unique(regressor))]



# plot a barplot of regressor importance



ggplot(imp, aes(x = regressor, y = value, fill = value)) +

geom_bar(stat = "identity", position = "dodge") +

labs(x = "Regressor", y = "Importance") +

scale_y_continuous(labels = percent) +

scale_fill_gradient(low = "yellow", high = "red") +

facet_wrap(~variable) +

theme_dark(15) +

theme(legend.position = "none", axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))



# plot a barplot of mean regressor importance



imp.avg = imp[, .(value = mean(value)), by = regressor]



ggplot(imp.avg, aes(x = regressor, y = value, fill = value)) +

geom_bar(stat = "identity", position = "dodge") +

labs(x = "Regressor", y = "Importance") +

scale_y_continuous(labels = percent) +

scale_fill_gradient(low = "yellow", high = "red") +

theme_dark(20) +

theme(legend.position = "none", axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))



# only keep variables with an importance greater than 40%



keep.dat = c("Survived", gsub("`", "", imp.avg[value >= 0.4, regressor]))

mod.dat = mod.dat[, keep.dat, with = FALSE]



# ---- variable selection by recursive feature elimination ----



# lets use cross validation to train the random forest model that will select our variables



# set up the cross validation mechanism

cv = rfeControl(functions = rfFuncs, method = "repeatedcv", number = K, repeats = B)



# run the RFE algorithm

vars = rfe(Survived ~ ., data = mod.dat, sizes = 1:(ncol(mod.dat) - 1), metric = "Accuracy", maximize = TRUE, rfeControl = cv)

vars

vars$optVariables



# heres our regressors for Survived



survived.regressors = c("Sexmale", "Sexmale.Fare", "Pclass3.Age", "Age", "Fare", "Pclass3.Fare", "Sexmale.EmbarkedS", "Pclass3.EmbarkedS", "SibSp.2")



# lets remove all columns from train not in survived.regressors



train = mod.dat[, c("Survived", survived.regressors), with = FALSE]



# lets update test so it has the same regressors as train



mod.dat = data.table(test)



# lets not include Name and Ticket becuase we cannot afford that many degrees of freedom (these are factors with a large number of levels)

# lets not include PassengerId because that is an ID variable for each row, so it won't be helpful



mod.dat[, c("Name", "Ticket", "PassengerId") := NULL]



# create all terms up to two-way interactions



mod.dat = data.table(model.matrix(~ .^2, mod.dat)[,-1])



# rename column names to have "." instead of ":"



name = gsub(":", ".", names(mod.dat))

setnames(mod.dat, name)



# add transformation term to test



mod.dat[, SibSp.2 := SibSp^2]



# lets remove all columns from test not in survived.regressors



test = mod.dat[, survived.regressors, with = FALSE]



# remove objects we no longer need



rm(keep.dat, name, mod.dat, cors, cor.dat, find.dat, cv, imp, imp.avg, mod, vars, survived.regressors)



}



# -----------------------------------------------------------------------------------

# ---- Logistic Regression Model ----------------------------------------------------

# -----------------------------------------------------------------------------------



{



# ---- Set Up -----------------------------------------------------------------------



# lets build a logistic regression model with all terms

# we are trying to see if we can remove any non-significant regressors



log.mod.all = glm(Survived ~ ., data = train, 

				family = binomial(link = "logit"), 

				control = list(maxit = 100))



summary(log.mod.all)



# remove Sexmale.EmbarkedS



log.mod.trim = glm(Survived ~ . - Sexmale.EmbarkedS, data = train, 

					family = binomial(link = "logit"), 

					control = list(maxit = 100))



summary(log.mod.trim)



# remove Pclass3.Fare



log.mod.trim = glm(Survived ~ . - Sexmale.EmbarkedS - Pclass3.Fare, data = train, 

					family = binomial(link = "logit"), 

					control = list(maxit = 100))



summary(log.mod.trim)



# ---- ROC --------------------------------------------------------------------------



# create names for each model

mod.name = c("all", "trim")



# lets compare the ROC curves of log.mod.all and log.mod.trim to determine the right cutoff point for each model

	# here are some useful definitions:

		# threshold - the point at which to choose 0 or 1 for the prediction (ie. cutoff point)

			# say threshold = .2, then any prediction >= 0.2 is accepted as 1 otherwise the prediction is accepted as 0

		# sensitivities - the proportion of observations that were correctly labeled as 1 (ie. True Positive Rate)

		# specificities - the proportion of observations that were correctly labeled as 0 (ie. True Negative Rate)

		# cases - the total amount of 1's in the data

		# controls - the total amount of 0's in the data

		# AUC - area under the curve - average value of sensitivity for all possible values of specificity, and vis versa



# extract actual values

actuals = as.numeric(as.character(train$Survived))



# compute fitted values of each model

fits = list(fitted(log.mod.all), fitted(log.mod.trim))



# compute an roc object for each model

rocs = lapply(1:length(fits), function(i) roc(actuals ~ fits[[i]]))



# compute cutoffs to evaluate

cutoffs = c(-Inf, seq(0.01, 0.99, 0.01), Inf)



# compute roc curves for each model

rocs.cutoffs = lapply(1:length(fits), function(i) data.table(t(coords(rocs[[i]], x = cutoffs, input = "threshold", ret = c("threshold", "specificity", "sensitivity", "accuracy")))))



# aggregate results into one table

DT = rbindlist(lapply(1:length(fits), function(i) data.table(mod = mod.name[i], 

																cutoff = rocs.cutoffs[[i]]$threshold, 

																TPR = rocs.cutoffs[[i]]$sensitivity, 

																FPR = 1 - rocs.cutoffs[[i]]$specificity, 

																ACC = rocs.cutoffs[[i]]$accuracy)))



# make mod into a factor data type to facilitate plotting

DT[, mod := factor(mod)]



# compute cut-offs points that maximize accuracy

	# accuracy is the total number of correct predictions

# these cuts are based on two weights (ie. two entries in the vector 'best.weights')

	# entry 1: the relative cost of of a false negative classification (as compared with a false positive classification)

		# default value = 1 (ie. both types of false classifications are equally bad)

	# entry 2: the prevalence, or the proportion of cases in the population (n.cases/(n.controls + n.cases))

		# default value = 0.5 (ie. there are expeted to be an equal number of 0's and 1's)

		

cuts = as.numeric(sapply(1:length(fits), function(i) 

							coords(rocs[[i]], 

									x = "best", 

									best.weights = c(1, length(rocs[[i]]$cases) / length(rocs[[i]]$response)))[1]))



# plot ROC curves



roc.plot = ggplot(DT, aes(x = FPR, y = TPR, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_abline(slope = 1, size = 1) +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "False Positive", y = "True Positive", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



roc.plot



# plot the accuracy curves

	# the dashed lines indicate the cut-off for each model that maximizes accuracy

	

acc.plot = ggplot(DT, aes(x = cutoff, y = ACC, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_vline(xintercept = cuts, color = ggcolor(length(cuts)), size = 1, linetype = "dashed") +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "Cut-Off", y = "Accuracy", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



acc.plot



# ---- CV ---------------------------------------------------------------------------



# lets use cross validation to choose which model is best



# two definitions:

	# positive ~ a passenger survives

	# negative ~ a passenger doesn't survive



# for each iteration in the K-fold validation we will compute the following metrics

	# acc ~ percent of correct predictions

	# sens ~ percent of correct positive predictions out of all observations that are truely positive

	# spec ~ percent of correct negative predictions out of all observations that are truely negative

	# ppv ~ percent of correct positive predictions out of all observations that were predicted to be positive

	# npv ~ percent of correct negative predictions out of all observations that were predicted to be negative

	# or ~ likelihood of being correct over the likelihood of being incorrect

		# (ie. we are 'or' times more likely to be correct than incorrect)



# build a function that will report prediction results of our models



log.pred = function(Xtrain, Ytrain, Xtest, Ytest, negative, cutoff)

{

	# build the table for training the model

	dat = data.table(Xtrain)

	dat[, y := Ytrain]

	

	# build the training model

	mod = glm(y ~ ., data = dat, 

				family = binomial(link = "logit"), 

				control = list(maxit = 100))

	

	# make predictions with the training model using the test set

	ynew = predict(mod, data.table(Xtest), type = "response")

	ynew = as.numeric(ynew >= cutoff)

	Ytest = as.numeric(as.character(Ytest))

	

	# build a confusion matrix to summarize the performance of our training model

	output = confusionMatrix(Ytest, ynew, negative = negative)

	output = c(output, "AUC" = as.numeric(auc(roc(Ytest ~ ynew))))

	

	# the confusion matrix is the output

	return(output)

}



# compute diagnostic errors for each model



log.diag = foreach(i = 1:2) %do%

{

	# extract our predictors (X)

	

	if(i == 1)

	{

		X = data.table(train)

		

	} else

	{

		X = data.table(train[,!c("Sexmale.EmbarkedS", "Pclass3.Fare")])

	} 

	

	# extract our response (Y)

	

	Y = X$Survived

	X[, Survived := NULL]

	

	# perform cross validation

	

	log.cv = crossval(predfun = log.pred, X = X, Y = Y, K = K, B = B, 

						verbose = FALSE, cutoff = cuts[i], negative = 0)



	# compute diagnostic errors of cross validation



	output = diag.cv(log.cv)

	output[, mod := mod.name[i]]

	

	return(output)

}



# combine the list of tables into one table



log.diag = rbindlist(log.diag)



# convert log.diag into long format for plotting purposes



DT = data.table(melt(log.diag, id.vars = c("stat", "mod")))



# convert mod into a factor for plotting purposes



DT[, mod := factor(mod)]



# remove Inf values as these don't help



DT = data.table(DT[value < Inf])



# plot barplots of each diagnostic metric



ggplot(DT[stat == "Q1" | stat == "Median" | stat == "Q3"], aes(x = stat, y = value, group = mod, fill = mod)) +

geom_bar(stat = "identity", position = "dodge") +

labs(x = "Summary Statistic", y = "Value", fill = "Model") + 

facet_wrap(~variable, scales = "free_y") +

theme_bw(base_size = 15) +

theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



# ---- Results ----------------------------------------------------------------------



# in the roc and accuracy plots both models are very similar

# in the diagnostics barplot, log.mod.trim is slightly higher across most metrics



names(log.mod.trim$coefficients)



log.mod = glm(formula = Survived ~ Sexmale + Sexmale.Fare + Pclass3.Age + Age + Fare + Pclass3.EmbarkedS + SibSp.2, 

				data = train, family = binomial(link = "logit"), control = list(maxit = 100))



log.cutoff = cuts[2]



# only keep mod == trim and change it to log as this is our chosen logistic regression model



log.diag = log.diag[mod == "trim"]

log.diag[, mod := rep("log", nrow(log.diag))]



# store model diagnostic results



mods.diag = data.table(log.diag)



# store the model



log.list = list("mod" = log.mod, "cutoff" = log.cutoff)

mods.list = list("log" = log.list)



# remove objects we no longer need



rm(actuals, cuts, DT, fits, i, roc.plot, acc.plot,

	log.cutoff, log.cv, log.diag, log.list, log.mod, log.mod.all, log.mod.trim, 

	log.pred, mod.name, output, rocs, X, Y, rocs.cutoffs, cutoffs)



}



# -----------------------------------------------------------------------------------

# ---- Penalty Regression Model -----------------------------------------------------

# -----------------------------------------------------------------------------------



{



# Ridge Regression differs from least squares regression by penalizing the objective function (Min: RSS) by adding the product of a constant (lambda2) and the sum of the squared coefficients (the L-2 norm)

# so the objective function for Ridge Regression is (Min: RSS + lambda2 * L-2)

# this penalty factor is used to discourage overfitting by incentivizing the model to:

	# choose coefficients that aren't too large

	# shrink coefficients that would be small, close to zero

# glmnet offers ridge regression by setting the parameter alpha = 0



# Lasso Regression differs from least squares regression by penalizing the objective function (Min: RSS) by adding the product of a constant (lambda1) and the sum of the absolute coefficients (the L-1 norm)

# so the objective function for Ridge Regression is (Min: RSS + lambda1 * L-1)

# this penalty factor is used to discourage overfitting by incentivizing the model to:

	# choose coefficients that aren't too large

	# shrink coefficients that would be small, to zero

# glmnet offers lasso regression by setting the parameter alpha = 1



# Elastic Net Regression differs from least squares regression by penalizing the objective function (Min: RSS) by adding the Ridge Regression and Lasso Regression penalty factors

# so the objective function for Ridge Regression is (Min: RSS + lambda2 * L-2 + lambda1 * L-1)

# the glmnet package uses the parameters alpha and lambda, where:

	# lambda = lambda1 + lambda2

	# alpha = lambda1 / lambda

# the glmnet package offers cross validation to optimize lambda for a given alpha

# alpha takes on values between 0 and 1 so we will try various alpha values and use cross validation to see which alpha yeilds the best model



# ---- Set Up -----------------------------------------------------------------------



# extract predictors (X) and response (Y)

# lets use the regressors from the log.mod.all and log.mod.trim models and see which set of variables performs better



X.all = data.table(train[,!"Survived"])

X.trim = data.table(X.all[, !c("Sexmale.EmbarkedS", "Pclass3.Fare")])

X.all = as.matrix(X.all)

X.trim = as.matrix(X.trim)

Y = train$Survived



# build a sequence of alpha values to test



doe = data.table(expand.grid(name = c("all", "trim"), alpha = seq(0, 1, 0.1)))



# build each model in doe



mods = foreach(i = 1:nrow(doe)) %do%

{

	# extract our predictors (X)

	

	if(doe$name[i] == "all")

	{

		X = X.all

		

	} else

	{

		X = X.trim

	}

	

	output = cv.glmnet(x = X, y = Y, family = "binomial", alpha = doe$alpha[i])

	return(output)

}



# ---- ROC --------------------------------------------------------------------------



# create names for each model

mod.name = 1:nrow(doe)



# extract actual values

actuals = as.numeric(as.character(Y))



# compute fitted values of each model

fits = foreach(i = 1:length(mods)) %do% 

{

	# extract our predictors (X)

	

	if(doe$name[i] == "all")

	{

		X = X.all

		

	} else

	{

		X = X.trim

	}

	

	output = as.numeric(predict(mods[[i]], s = mods[[i]]$lambda.min, X, type = "response"))

	return(output)

}



# compute an roc object for each model

rocs = lapply(1:length(fits), function(i) roc(actuals ~ fits[[i]]))



# compute cutoffs to evaluate

cutoffs = c(-Inf, seq(0.01, 0.99, 0.01), Inf)



# compute roc curves for each model

rocs.cutoffs = lapply(1:length(fits), function(i) data.table(t(coords(rocs[[i]], x = cutoffs, input = "threshold", ret = c("threshold", "specificity", "sensitivity", "accuracy")))))



# aggregate results into one table

DT = rbindlist(lapply(1:length(fits), function(i) data.table(mod = mod.name[i], 

																cutoff = rocs.cutoffs[[i]]$threshold, 

																TPR = rocs.cutoffs[[i]]$sensitivity, 

																FPR = 1 - rocs.cutoffs[[i]]$specificity, 

																ACC = rocs.cutoffs[[i]]$accuracy)))



# make mod into a factor data type to facilitate plotting

DT[, mod := factor(mod)]



# compute cut-offs points that maximize accuracy

cuts = as.numeric(sapply(1:length(fits), function(i) 

							coords(rocs[[i]], 

									x = "best", 

									best.weights = c(1, length(rocs[[i]]$cases) / length(rocs[[i]]$response)))[1]))



# plot ROC curves



roc.plot = ggplot(DT, aes(x = FPR, y = TPR, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_abline(slope = 1, size = 1) +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "False Positive", y = "True Positive", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 2))



roc.plot



# plot the accuracy curves

	# the dashed lines indicate the cut-off for each model that maximizes accuracy

	

acc.plot = ggplot(DT, aes(x = cutoff, y = ACC, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_vline(xintercept = cuts, color = ggcolor(length(cuts)), size = 1, linetype = "dashed") +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "Cut-Off", y = "Accuracy", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 2))



acc.plot



# ---- CV ---------------------------------------------------------------------------



# build a function that will report prediction results of our models



pen.pred = function(Xtrain, Ytrain, Xtest, Ytest, negative, cutoff, alpha)

{

	# build the training model

	mod = cv.glmnet(x = Xtrain, y = Ytrain, family = "binomial", alpha = alpha)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(predict(mod, s = mod$lambda.min, Xtest, type = "response"))

	ynew = as.numeric(ynew >= cutoff)

	Ytest = as.numeric(as.character(Ytest))

	

	# build a confusion matrix to summarize the performance of our training model

	output = confusionMatrix(Ytest, ynew, negative = negative)

	output = c(output, "AUC" = as.numeric(auc(roc(Ytest ~ ynew))))

	

	# the confusion matrix is the output

	return(output)

}



# compute diagonistic errors for each of the models in doe



pen.diag = foreach(i = 1:nrow(doe)) %do%

{

	# extract our predictors (X)

	

	if(doe$name[i] == "all")

	{

		X = X.all

		

	} else

	{

		X = X.trim

	}

	

	# perform cross validation

	

	pen.cv = crossval(predfun = pen.pred, X = X, Y = Y, K = K, B = B, 

						verbose = FALSE, negative = 0, alpha = doe$alpha[i], cutoff = cuts[i])



	# compute diagnostic errors of cross validation



	output = diag.cv(pen.cv)



	# add columns of parameter values that define model i



	output[, name := doe$name[i]]

	output[, alpha := doe$alpha[i]]

	output[, mod := i]

	

	return(output)

}



# combine the list of tables into one table

pen.diag = rbindlist(pen.diag)



# convert pen.diag into long format for plotting purposes

DT = data.table(melt(pen.diag, id.vars = c("stat", "name", "alpha", "mod")))



# convert mod into a factor for plotting purposes

DT[, mod := factor(mod)]



# remove Inf values as these don't help

DT = data.table(DT[value < Inf])



# plot barplots of each diagnostic metric



diag.plot = ggplot(DT[stat == "Q1" | stat == "Median" | stat == "Q3"], aes(x = stat, y = value, group = mod, fill = mod)) +

			geom_bar(stat = "identity", position = "dodge", color = "white") +

			labs(x = "Summary Statistic", y = "Value", fill = "Model") + 

			facet_wrap(~variable, scales = "free_y") +

			theme_bw(base_size = 15) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



diag.plot



# ---- Results ----------------------------------------------------------------------



# the roc and accuracy plots look very similar

# the odds ratio in the diagnostics plot stands out across models

# lets filter out models



pen.diag[stat == "Median" & or >= 20 & sens >= 0.60 & name == "trim"]



# model 14 looks good in this group

# build our model using all terms and let glmnet's cross validation determine which terms to keep



pen.mod = cv.glmnet(x = X.trim, y = Y, family = "binomial", alpha = 0.6)



# extract coefficients of the chosen terms for the lambda that minimizes mean cross-validated error



pen.coef = coef(pen.mod, s = "lambda.min")

pen.coef = data.table(term = rownames(pen.coef), coefficient = as.numeric(pen.coef))



# store model diagnostic results



pen.diag = pen.diag[mod == 14]

pen.diag[, c("name", "alpha") := NULL]



pen.diag[, mod := rep("pen", nrow(pen.diag))]

mods.diag = rbind(mods.diag, pen.diag)



# store the model



pen.list = list("mod" = pen.mod, "coef" = pen.coef, "cutoff" = cuts[14])

mods.list$pen = pen.list



# remove objects we no longer need



rm(DT, pen.pred, pen.cv, X, Y, pen.diag, pen.list, pen.mod, pen.coef, mods, doe, i, diag.plot,

	output, acc.plot, actuals, cutoffs, cuts, fits, mod.name, roc.plot, rocs, rocs.cutoffs,

	X.all, X.trim)



}



# -----------------------------------------------------------------------------------

# ---- Linear Discriminant Analysis -------------------------------------------------

# -----------------------------------------------------------------------------------



{



# ---- Set Up -----------------------------------------------------------------------



# extract predictors (X) and response (Y)

# lets use the regressors from the log.mod.all and log.mod.trim models and see which set of variables performs better



X.all = data.table(train[,!"Survived"])

X.trim = data.table(X.all[, !c("Sexmale.EmbarkedS", "Pclass3.Fare")])

Y = train$Survived



# build a sequence of alpha values to test



doe = data.table(name = c("all", "trim"))



# build each model in doe



mods = foreach(i = 1:nrow(doe)) %do%

{

	# extract our predictors (X)

	

	if(doe$name[i] == "all")

	{

		X = X.all

		

	} else

	{

		X = X.trim

	}

	

	# build table for modeling

	

	dat = cbind(Y, X)

	

	# build model

	

	output = lda(Y ~ ., data = dat)

	return(output)

}



# ---- ROC --------------------------------------------------------------------------



# create names for each model

mod.name = 1:nrow(doe)



# extract actual values

actuals = as.numeric(as.character(Y))



# compute fitted values of each model

fits = foreach(i = 1:length(mods)) %do% 

{

	# extract our predictors (X)

	

	if(doe$name[i] == "all")

	{

		X = X.all

		

	} else

	{

		X = X.trim

	}

	

	output = as.numeric(predict(mods[[i]], newdata = X)$posterior[,2])

	return(output)

}



# compute an roc object for each model

rocs = lapply(1:length(fits), function(i) roc(actuals ~ fits[[i]]))



# compute cutoffs to evaluate

cutoffs = c(-Inf, seq(0.01, 0.99, 0.01), Inf)



# compute roc curves for each model

rocs.cutoffs = lapply(1:length(fits), function(i) data.table(t(coords(rocs[[i]], x = cutoffs, input = "threshold", ret = c("threshold", "specificity", "sensitivity", "accuracy")))))



# aggregate results into one table

DT = rbindlist(lapply(1:length(fits), function(i) data.table(mod = mod.name[i], 

																cutoff = rocs.cutoffs[[i]]$threshold, 

																TPR = rocs.cutoffs[[i]]$sensitivity, 

																FPR = 1 - rocs.cutoffs[[i]]$specificity, 

																ACC = rocs.cutoffs[[i]]$accuracy)))



# make mod into a factor data type to facilitate plotting

DT[, mod := factor(mod)]



# compute cut-offs points that maximize accuracy

cuts = as.numeric(sapply(1:length(fits), function(i) 

							coords(rocs[[i]], 

									x = "best", 

									best.weights = c(1, length(rocs[[i]]$cases) / length(rocs[[i]]$response)))[1]))



# plot ROC curves



roc.plot = ggplot(DT, aes(x = FPR, y = TPR, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_abline(slope = 1, size = 1) +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "False Positive", y = "True Positive", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



roc.plot



# plot the accuracy curves

	# the dashed lines indicate the cut-off for each model that maximizes accuracy

	

acc.plot = ggplot(DT, aes(x = cutoff, y = ACC, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_vline(xintercept = cuts, color = ggcolor(length(cuts)), size = 1, linetype = "dashed") +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "Cut-Off", y = "Accuracy", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



acc.plot



# ---- CV ---------------------------------------------------------------------------



# build a function that will report prediction results of our models



lda.pred = function(Xtrain, Ytrain, Xtest, Ytest, negative, cutoff)

{

	# build table for modeling

	

	dat = cbind("Y" = Ytrain, Xtrain)

	

	# build the training model

	mod = lda(Y ~ ., data = dat)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(predict(mod, newdata = Xtest)$posterior[,2])

	ynew = as.numeric(ynew >= cutoff)

	Ytest = as.numeric(as.character(Ytest))

	

	# build a confusion matrix to summarize the performance of our training model

	output = confusionMatrix(Ytest, ynew, negative = negative)

	output = c(output, "AUC" = as.numeric(auc(roc(Ytest ~ ynew))))

	

	# the confusion matrix is the output

	return(output)

}



# compute diagonistic errors for each of the models in doe



lda.diag = foreach(i = 1:nrow(doe)) %do%

{

	# extract our predictors (X)

	

	if(doe$name[i] == "all")

	{

		X = X.all

		

	} else

	{

		X = X.trim

	}

	

	# perform cross validation

	

	lda.cv = crossval(predfun = lda.pred, X = X, Y = Y, K = K, B = B, 

						verbose = FALSE, negative = 0, cutoff = cuts[i])



	# compute diagnostic errors of cross validation



	output = diag.cv(lda.cv)



	# add columns of parameter values that define model i



	output[, name := doe$name[i]]

	output[, mod := i]

	

	return(output)

}



# combine the list of tables into one table

lda.diag = rbindlist(lda.diag)



# convert lda.diag into long format for plotting purposes

DT = data.table(melt(lda.diag, id.vars = c("stat", "name", "mod")))



# convert mod into a factor for plotting purposes

DT[, mod := factor(mod)]



# remove Inf values as these don't help

DT = data.table(DT[value < Inf])



# plot barplots of each diagnostic metric



diag.plot = ggplot(DT[stat == "Q1" | stat == "Median" | stat == "Q3"], aes(x = stat, y = value, group = mod, fill = mod)) +

			geom_bar(stat = "identity", position = "dodge", color = "white") +

			labs(x = "Summary Statistic", y = "Value", fill = "Model") + 

			facet_wrap(~variable, scales = "free_y") +

			theme_bw(base_size = 15) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



diag.plot



# ---- Results ----------------------------------------------------------------------



# the roc, accuracy, and diagnostic plots look very similar

# lets go with the trim model becuase it requires less terms fo similar performance



lda.mod = lda(Survived ~ Sexmale + Sexmale.Fare + Pclass3.Age + Age + Fare + Pclass3.EmbarkedS + SibSp.2, data = train)



# store model diagnostic results



lda.diag = lda.diag[mod == 2]

lda.diag[, c("name") := NULL]



lda.diag[, mod := rep("lda", nrow(lda.diag))]

mods.diag = rbind(mods.diag, lda.diag)



# store the model



lda.list = list("mod" = lda.mod, "cutoff" = cuts[2])

mods.list$lda = lda.list



# remove objects we no longer need



rm(DT, dat, lda.pred, lda.cv, X, Y, lda.diag, lda.list, lda.mod, mods, doe, i, diag.plot,

	output, acc.plot, actuals, cutoffs, cuts, fits, mod.name, roc.plot, rocs, rocs.cutoffs,

	X.all, X.trim)



}



# -----------------------------------------------------------------------------------

# ---- Quadratic Discriminant Analysis ----------------------------------------------

# -----------------------------------------------------------------------------------



{



# ---- Set Up -----------------------------------------------------------------------



# extract predictors (X) and response (Y)

# lets use the regressors from the log.mod.all and log.mod.trim models and see which set of variables performs better



X.all = data.table(train[,!"Survived"])

X.trim = data.table(X.all[, !c("Sexmale.EmbarkedS", "Pclass3.Fare")])

Y = train$Survived



# build a sequence of alpha values to test



doe = data.table(name = c("all", "trim"))



# build each model in doe



mods = foreach(i = 1:nrow(doe)) %do%

{

	# extract our predictors (X)

	

	if(doe$name[i] == "all")

	{

		X = X.all

		

	} else

	{

		X = X.trim

	}

	

	# build table for modeling

	

	dat = cbind(Y, X)

	

	# build model

	

	output = qda(Y ~ ., data = dat)

	return(output)

}



# ---- ROC --------------------------------------------------------------------------



# create names for each model

mod.name = 1:nrow(doe)



# extract actual values

actuals = as.numeric(as.character(Y))



# compute fitted values of each model

fits = foreach(i = 1:length(mods)) %do% 

{

	# extract our predictors (X)

	

	if(doe$name[i] == "all")

	{

		X = X.all

		

	} else

	{

		X = X.trim

	}

	

	output = as.numeric(predict(mods[[i]], newdata = X)$posterior[,2])

	return(output)

}



# compute an roc object for each model

rocs = lapply(1:length(fits), function(i) roc(actuals ~ fits[[i]]))



# compute cutoffs to evaluate

cutoffs = c(-Inf, seq(0.01, 0.99, 0.01), Inf)



# compute roc curves for each model

rocs.cutoffs = lapply(1:length(fits), function(i) data.table(t(coords(rocs[[i]], x = cutoffs, input = "threshold", ret = c("threshold", "specificity", "sensitivity", "accuracy")))))



# aggregate results into one table

DT = rbindlist(lapply(1:length(fits), function(i) data.table(mod = mod.name[i], 

																cutoff = rocs.cutoffs[[i]]$threshold, 

																TPR = rocs.cutoffs[[i]]$sensitivity, 

																FPR = 1 - rocs.cutoffs[[i]]$specificity, 

																ACC = rocs.cutoffs[[i]]$accuracy)))



# make mod into a factor data type to facilitate plotting

DT[, mod := factor(mod)]



# compute cut-offs points that maximize accuracy

cuts = as.numeric(sapply(1:length(fits), function(i) 

							coords(rocs[[i]], 

									x = "best", 

									best.weights = c(1, length(rocs[[i]]$cases) / length(rocs[[i]]$response)))[1]))



# plot ROC curves



roc.plot = ggplot(DT, aes(x = FPR, y = TPR, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_abline(slope = 1, size = 1) +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "False Positive", y = "True Positive", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



roc.plot



# plot the accuracy curves

	# the dashed lines indicate the cut-off for each model that maximizes accuracy

	

acc.plot = ggplot(DT, aes(x = cutoff, y = ACC, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_vline(xintercept = cuts, color = ggcolor(length(cuts)), size = 1, linetype = "dashed") +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "Cut-Off", y = "Accuracy", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



acc.plot



# ---- CV ---------------------------------------------------------------------------



# build a function that will report prediction results of our models



qda.pred = function(Xtrain, Ytrain, Xtest, Ytest, negative, cutoff)

{

	# build table for modeling

	

	dat = cbind("Y" = Ytrain, Xtrain)

	

	# build the training model

	mod = qda(Y ~ ., data = dat)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(predict(mod, newdata = Xtest)$posterior[,2])

	ynew = as.numeric(ynew >= cutoff)

	Ytest = as.numeric(as.character(Ytest))

	

	# build a confusion matrix to summarize the performance of our training model

	output = confusionMatrix(Ytest, ynew, negative = negative)

	output = c(output, "AUC" = as.numeric(auc(roc(Ytest ~ ynew))))

	

	# the confusion matrix is the output

	return(output)

}



# compute diagonistic errors for each of the models in doe



qda.diag = foreach(i = 1:nrow(doe)) %do%

{

	# extract our predictors (X)

	

	if(doe$name[i] == "all")

	{

		X = X.all

		

	} else

	{

		X = X.trim

	}

	

	# perform cross validation

	

	qda.cv = crossval(predfun = qda.pred, X = X, Y = Y, K = K, B = B, 

						verbose = FALSE, negative = 0, cutoff = cuts[i])



	# compute diagnostic errors of cross validation



	output = diag.cv(qda.cv)



	# add columns of parameter values that define model i



	output[, name := doe$name[i]]

	output[, mod := i]

	

	return(output)

}



# combine the list of tables into one table

qda.diag = rbindlist(qda.diag)



# convert qda.diag into long format for plotting purposes

DT = data.table(melt(qda.diag, id.vars = c("stat", "name", "mod")))



# convert mod into a factor for plotting purposes

DT[, mod := factor(mod)]



# remove Inf values as these don't help

DT = data.table(DT[value < Inf])



# plot barplots of each diagnostic metric



diag.plot = ggplot(DT[stat == "Q1" | stat == "Median" | stat == "Q3"], aes(x = stat, y = value, group = mod, fill = mod)) +

			geom_bar(stat = "identity", position = "dodge", color = "white") +

			labs(x = "Summary Statistic", y = "Value", fill = "Model") + 

			facet_wrap(~variable, scales = "free_y") +

			theme_bw(base_size = 15) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



diag.plot



# ---- Results ----------------------------------------------------------------------



# the roc and accuracy plots look very similar

# the diagnostic plot shows the all model to perform better



qda.mod = qda(Survived ~ ., data = train)



# store model diagnostic results



qda.diag = qda.diag[mod == 1]

qda.diag[, c("name") := NULL]



qda.diag[, mod := rep("qda", nrow(qda.diag))]

mods.diag = rbind(mods.diag, qda.diag)



# store the model



qda.list = list("mod" = qda.mod, "cutoff" = cuts[1])

mods.list$qda = qda.list



# remove objects we no longer need



rm(DT, dat, qda.pred, qda.cv, X, Y, qda.diag, qda.list, qda.mod, mods, doe, i, diag.plot,

	output, acc.plot, actuals, cutoffs, cuts, fits, mod.name, roc.plot, rocs, rocs.cutoffs,

	X.all, X.trim)



}



# -----------------------------------------------------------------------------------

# ---- Gradient Boosting Model ------------------------------------------------------

# -----------------------------------------------------------------------------------



{



# ---- Set Up -----------------------------------------------------------------------



# we have 5 hyperparameters of interest:

	# eta  ~ the learning rate

		# default value: 0.3

	# max_depth  ~ maximum depth of a tree

		# default value: 6

	# min_child_weight ~ minimum sum of instance weight needed in a child

		# default value: 1

	# nrounds ~ max number of iterations

	# gamma ~ minimum loss reduction required to make a further partition on a leaf node

		

# check out this link for help on tuning:

# # http://machinelearningmastery.com/configure-gradient-boosting-algorithm/

	

# extract our predictors (X) and response (Y)



X = data.table(train)

Y = as.numeric(as.character(X$Survived))

X[, Survived := NULL]

X = as.matrix(X)



# compute the proportion of events in our response (ie. Survived == 1)



event.rate = nrow(train[Survived == 1]) / nrow(train)



# create parameter combinations to test



doe = data.table(expand.grid(nrounds = 500,

								eta = 0.1,

								max_depth = c(4, 7, 10), 

								min_child_weight = seq(1/sqrt(event.rate), 3/event.rate, length.out = 3),

								gamma = 0))



# build models

mods = lapply(1:nrow(doe), function(i) 

				xgboost(label = Y, data = X,

						objective = "binary:logistic", eval_metric = "auc",

						eta = doe$eta[i],

						max_depth = doe$max_depth[i],

						nrounds = doe$nrounds[i],

						min_child_weight = doe$min_child_weight[i],

						gamma = doe$gamma[i], verbose = 0))



# ---- ROC --------------------------------------------------------------------------



# create names for each model

mod.name = 1:nrow(doe)



# extract actual values

actuals = Y



# compute fitted values of each model

fits = lapply(1:length(mods), function(i) as.numeric(predict(mods[[i]], newdata = X)))



# compute an roc object for each model

rocs = lapply(1:length(fits), function(i) roc(actuals ~ fits[[i]]))



# compute cutoffs to evaluate

cutoffs = c(-Inf, seq(0.01, 0.99, 0.01), Inf)



# compute roc curves for each model

rocs.cutoffs = lapply(1:length(fits), function(i) data.table(t(coords(rocs[[i]], x = cutoffs, input = "threshold", ret = c("threshold", "specificity", "sensitivity", "accuracy")))))



# aggregate results into one table

DT = rbindlist(lapply(1:length(fits), function(i) data.table(mod = mod.name[i], 

																cutoff = rocs.cutoffs[[i]]$threshold, 

																TPR = rocs.cutoffs[[i]]$sensitivity, 

																FPR = 1 - rocs.cutoffs[[i]]$specificity, 

																ACC = rocs.cutoffs[[i]]$accuracy)))



# make mod into a factor data type to facilitate plotting

DT[, mod := factor(mod)]



# compute cut-offs points that maximize accuracy

cuts = as.numeric(sapply(1:length(fits), function(i) 

							coords(rocs[[i]], 

									x = "best", 

									best.weights = c(1, length(rocs[[i]]$cases) / length(rocs[[i]]$response)))[1]))



# plot ROC curves



roc.plot = ggplot(DT, aes(x = FPR, y = TPR, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_abline(slope = 1, size = 1) +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "False Positive", y = "True Positive", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



roc.plot



# plot the accuracy curves

	# the dashed lines indicate the cut-off for each model that maximizes accuracy

	

acc.plot = ggplot(DT, aes(x = cutoff, y = ACC, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_vline(xintercept = cuts, color = ggcolor(length(cuts)), size = 1, linetype = "dashed") +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "Cut-Off", y = "Accuracy", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



acc.plot



# ---- CV ---------------------------------------------------------------------------



# build a function that will report prediction results of our model



gbm.pred = function(Xtrain, Ytrain, Xtest, Ytest, negative, cutoff, objective, eval_metric, eta, max_depth, nrounds, min_child_weight, gamma)

{

	# build the training model

	mod = xgboost(label = Ytrain, data = Xtrain,

					objective = objective, eval_metric = eval_metric,

					eta = eta,

					max_depth = max_depth,

					nrounds = nrounds,

					min_child_weight = min_child_weight,

					gamma = gamma, verbose = 0)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(predict(mod, newdata = Xtest))

	ynew = as.numeric(ynew >= cutoff)

	

	# build a confusion matrix to summarize the performance of our training model

	output = confusionMatrix(Ytest, ynew, negative = negative)

	output = c(output, "AUC" = as.numeric(auc(roc(Ytest ~ ynew))))

	

	# the confusion matrix is the output

	return(output)

}



# compute diagonistic errors for each of the models in doe



gbm.diag = foreach(i = 1:nrow(doe)) %do%

{

	# perform cross validation

	

	gbm.cv = crossval(predfun = gbm.pred, X = X, Y = Y, K = K, B = B, 

						verbose = FALSE, negative = 0, cutoff = cuts[i],

						objective = "binary:logistic", eval_metric = "auc",

						eta = doe$eta[i], max_depth = doe$max_depth[i],

						nrounds = doe$nrounds[i], min_child_weight = doe$min_child_weight[i],

						gamma = doe$gamma[i])

	

	# compute diagnostic errors of cross validation

	

	output = diag.cv(gbm.cv)

	

	# add columns of parameter values that define model i

	

	output[, eta := doe$eta[i]]

	output[, max_depth := doe$max_depth[i]]

	output[, nrounds := doe$nrounds[i]]

	output[, min_child_weight := doe$min_child_weight[i]]

	output[, gamma := doe$gamma[i]]

	output[, mod := i]

	

	return(output)

}



# combine the list of tables into one table

gbm.diag = rbindlist(gbm.diag)



# convert gbm.diag into long format for plotting purposes

DT = data.table(melt(gbm.diag, id.vars = c("stat", "eta", "max_depth", "nrounds", "min_child_weight", "gamma", "mod")))



# convert mod into a factor for plotting purposes

DT[, mod := factor(mod)]



# remove Inf values as these don't help

DT = data.table(DT[value < Inf])



# plot barplots of each diagnostic metric



diag.plot = ggplot(DT[stat == "Q1" | stat == "Median" | stat == "Q3"], aes(x = stat, y = value, group = mod, fill = mod)) +

			geom_bar(stat = "identity", position = "dodge", color = "white") +

			labs(x = "Summary Statistic", y = "Value", fill = "Model") + 

			facet_wrap(~variable, scales = "free_y") +

			theme_bw(base_size = 15) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



diag.plot



# ---- Results ----------------------------------------------------------------------



# the roc and accuracy plots show similar performance across models

# in the diagnostics plot model 4 looks to be better than most models across each metric

# lets choose model 4



gbm.diag = gbm.diag[mod == 4]



# rename model to gbm as this is our chosen random forest model



gbm.diag[, mod := rep("gbm", nrow(gbm.diag))]



# build our model



gbm.mod = xgboost(label = Y, data = X,

					objective = "binary:logistic", eval_metric = "auc",

					eta = 0.1, max_depth = 4,

					nrounds = 500, min_child_weight = 4.714936,

					gamma = 0, verbose = 0)



# store model diagnostic results



gbm.diag[, c("eta", "max_depth", "nrounds", "min_child_weight", "gamma") := NULL]

mods.diag = rbind(mods.diag, gbm.diag)



# store the model



gbm.list = list("mod" = gbm.mod, "cutoff" = cuts[4])

mods.list$gbm = gbm.list



# remove objects we no longer need



rm(doe, DT, i, output, gbm.pred, gbm.cv, X, Y, gbm.diag, gbm.list, gbm.mod, event.rate,

	acc.plot, actuals, cutoffs, cuts, diag.plot, fits, mods, mod.name, roc.plot, rocs, rocs.cutoffs)



}



# -----------------------------------------------------------------------------------

# ---- Random Forest Model ----------------------------------------------------------

# -----------------------------------------------------------------------------------



{



# ---- Set Up -----------------------------------------------------------------------



# we have 3 hyperparameters of interest:

	# ntree ~ number of decision trees to create

		# default value: 500

	# mtry ~ the number of regressors to randomly choose at each parent node of each tree, from which the best is chosen (based on minimal OOB error) to split the parent node into two child nodes

		# default value: floor(sqrt(p)) (where p is number of regressors)

	# nodesize ~ minimum size of terminal nodes (ie. the minimum number of data points that can be grouped together in any node of a tree)

		# default value: 1



# default value for mtry in our case:

floor(sqrt(ncol(train) - 1))



# extract our predictors (X) and response (Y)



X = data.table(train)

Y = X$Survived

X[, Survived := NULL]



# create parameter combinations to test



doe = data.table(expand.grid(ntree = 500,

								mtry = c(3, 5, 7), 

								nodesize = c(1, 5, 9)))



# build models

mods = lapply(1:nrow(doe), function(i) 

				randomForest(Survived ~ .,

								data = train,

								ntree = doe$ntree[i],

								mtry = doe$mtry[i],

								nodesize = doe$nodesize[i]))



# ---- ROC --------------------------------------------------------------------------



# create names for each model

mod.name = 1:nrow(doe)



# extract actual values

actuals = as.numeric(as.character(train$Survived))



# compute fitted values of each model

fits = lapply(1:length(mods), function(i) as.numeric(predict(mods[[i]], type = "prob")[,2]))



# compute an roc object for each model

rocs = lapply(1:length(fits), function(i) roc(actuals ~ fits[[i]]))



# compute cutoffs to evaluate

cutoffs = c(-Inf, seq(0.01, 0.99, 0.01), Inf)



# compute roc curves for each model

rocs.cutoffs = lapply(1:length(fits), function(i) data.table(t(coords(rocs[[i]], x = cutoffs, input = "threshold", ret = c("threshold", "specificity", "sensitivity", "accuracy")))))



# aggregate results into one table

DT = rbindlist(lapply(1:length(fits), function(i) data.table(mod = mod.name[i], 

																cutoff = rocs.cutoffs[[i]]$threshold, 

																TPR = rocs.cutoffs[[i]]$sensitivity, 

																FPR = 1 - rocs.cutoffs[[i]]$specificity, 

																ACC = rocs.cutoffs[[i]]$accuracy)))



# make mod into a factor data type to facilitate plotting

DT[, mod := factor(mod)]



# compute cut-offs points that maximize accuracy

cuts = as.numeric(sapply(1:length(fits), function(i) 

							coords(rocs[[i]], 

									x = "best", 

									best.weights = c(1, length(rocs[[i]]$cases) / length(rocs[[i]]$response)))[1]))



# plot ROC curves



roc.plot = ggplot(DT, aes(x = FPR, y = TPR, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_abline(slope = 1, size = 1) +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "False Positive", y = "True Positive", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



roc.plot



# plot the accuracy curves

	# the dashed lines indicate the cut-off for each model that maximizes accuracy

	

acc.plot = ggplot(DT, aes(x = cutoff, y = ACC, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_vline(xintercept = cuts, color = ggcolor(length(cuts)), size = 1, linetype = "dashed") +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "Cut-Off", y = "Accuracy", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



acc.plot



# ---- CV ---------------------------------------------------------------------------



# build a function that will report prediction results of our model



rf.pred = function(Xtrain, Ytrain, Xtest, Ytest, negative, ntree, mtry, nodesize, cutoff)

{

	# build the table for training the model

	dat = data.table(Xtrain)

	dat[, y := Ytrain]

	

	# build the training model

	mod = randomForest(y ~ .,

						data = dat,

						ntree = ntree,

						mtry = mtry,

						nodesize = nodesize)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(predict(mod, newdata = data.table(Xtest), type = "prob")[,2])

	ynew = as.numeric(ynew >= cutoff)

	Ytest = as.numeric(as.character(Ytest))

	

	# build a confusion matrix to summarize the performance of our training model

	output = confusionMatrix(Ytest, ynew, negative = negative)

	output = c(output, "AUC" = as.numeric(auc(roc(Ytest ~ ynew))))

	

	# the confusion matrix is the output

	return(output)

}



# compute diagonistic errors for each of the models in doe



rf.diag = foreach(i = 1:nrow(doe)) %do%

{

	# perform cross validation

	

	rf.cv = crossval(predfun = rf.pred, X = X, Y = Y, K = K, B = B, 

						verbose = FALSE, negative = 0, 

						ntree = doe$ntree[i], mtry = doe$mtry[i], nodesize = doe$nodesize[i], cutoff = cuts[i])



	# compute diagnostic errors of cross validation



	output = diag.cv(rf.cv)



	# add columns of parameter values that define model i



	output[, ntree := doe$ntree[i]] 

	output[, mtry := doe$mtry[i]]

	output[, nodesize := doe$nodesize[i]]

	output[, mod := i]

	

	return(output)

}



# combine the list of tables into one table

rf.diag = rbindlist(rf.diag)



# convert rf.diag into long format for plotting purposes

DT = data.table(melt(rf.diag, id.vars = c("stat", "ntree", "mtry", "nodesize", "mod")))



# convert mod into a factor for plotting purposes

DT[, mod := factor(mod)]



# remove Inf values as these don't help

DT = data.table(DT[value < Inf])



# plot barplots of each diagnostic metric



diag.plot = ggplot(DT[stat == "Q1" | stat == "Median" | stat == "Q3"], aes(x = stat, y = value, group = mod, fill = mod)) +

			geom_bar(stat = "identity", position = "dodge", color = "white") +

			labs(x = "Summary Statistic", y = "Value", fill = "Model") + 

			facet_wrap(~variable, scales = "free_y") +

			theme_bw(base_size = 15) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



diag.plot



# ---- Results ----------------------------------------------------------------------



# the roc and accuracy plots show similar performance across models

# in the diagnostics plot, model 4 seems to do better than most models across all metrics

# lets go with model 4



rf.diag = rf.diag[mod == 4]



# rename model to rf as this is our chosen random forest model



rf.diag[, mod := rep("rf", nrow(rf.diag))]



# build our random forest model



rf.mod = randomForest(Survived ~ ., data = train,

						ntree = 500, mtry = 3, nodesize = 5)



# store model diagnostic results



rf.diag[, c("ntree", "mtry", "nodesize") := NULL]

mods.diag = rbind(mods.diag, rf.diag)



# store the model



rf.list = list("mod" = rf.mod, "cutoff" = cuts[4])

mods.list$rf = rf.list



# remove objects we no longer need



rm(doe, DT, i, output, rf.pred, rf.cv, X, Y, rf.diag, rf.list, rf.mod,

	acc.plot, actuals, cutoffs, cuts, diag.plot, fits, mods, mod.name, roc.plot, rocs, rocs.cutoffs)



}



# -----------------------------------------------------------------------------------

# ---- Nueral Network Model ---------------------------------------------------------

# -----------------------------------------------------------------------------------



{



# ---- Set Up -----------------------------------------------------------------------



# adjust the structure of train such that each regressor is numeric



# create a copy of train

train.nn = data.table(train)

# make Survived a numeric value

train.nn[, Survived := as.numeric(as.character(Survived))]



# check out the min and max values of each column in train.nn

min.max = melt(train.nn[, !"Survived"], measure.vars = 1:(ncol(train.nn) - 1))

min.max = min.max[, .(min = min(value), max = max(value)), by = variable]

min.max = min.max[min < 0 | max > 1]

min.max



# rescale columns that have a max value above 1 and/or a min value below 0

	# a 0 to 1 scaling is required for neural networks

train.nn[, rescale.Age := rescale(Age, to = c(0, 1))]

train.nn[, rescale.Fare := rescale(Fare, to = c(0, 1))]

train.nn[, rescale.SibSp.2 := rescale(SibSp.2, to = c(0, 1))]

train.nn[, rescale.Sexmale.Fare := rescale(Sexmale.Fare, to = c(0, 1))]

train.nn[, rescale.Pclass3.Age := rescale(Pclass3.Age, to = c(0, 1))]

train.nn[, rescale.Pclass3.Fare := rescale(Pclass3.Fare, to = c(0, 1))]



# remove columns that violate the scale

train.nn[, as.character(min.max$variable) := NULL]



# repeat this adjustment for test incase we end up using the chosen neuralnet on test



# create a copy of train

test.nn = data.table(test)



# check out the min and max values of each column in test.nn

min.max = melt(test.nn, measure.vars = 1:ncol(test.nn))

min.max = min.max[, .(min = min(value), max = max(value)), by = variable]

min.max = min.max[min < 0 | max > 1]

min.max



# rescale columns that have a max value above 1 and/or a min value below 0

	# a 0 to 1 scaling is required for neural networks

test.nn[, rescale.Age := rescale(Age, to = c(0, 1), from = range(train$Age))]

test.nn[, rescale.Fare := rescale(Fare, to = c(0, 1), from = range(train$Fare))]

test.nn[, rescale.SibSp.2 := rescale(SibSp.2, to = c(0, 1), from = range(train$SibSp.2))]

test.nn[, rescale.Sexmale.Fare := rescale(Sexmale.Fare, to = c(0, 1), from = range(train$Sexmale.Fare))]

test.nn[, rescale.Pclass3.Age := rescale(Pclass3.Age, to = c(0, 1), from = range(train$Pclass3.Age))]

test.nn[, rescale.Pclass3.Fare := rescale(Pclass3.Fare, to = c(0, 1), from = range(train$Pclass3.Fare))]



# remove columns that violate the scale

test.nn[, as.character(min.max$variable) := NULL]



# extract our predictors (X) and response (Y)

X = data.table(train.nn)

Y = X$Survived

X[, Survived := NULL]



# we have 3 hyperparameters of interest:

	# hidden ~ a vector of integers specifying the number of hidden neurons (vertices) in each layer

	# learningrate ~ a numeric value specifying the threshold for the partial derivatives of the error function as stopping criteria

	# threshold ~ a numeric value specifying the learning rate used by traditional backpropagation. Used only for traditional backpropagation.

	

# lets use cross validation to tune our model



# we are going to use rprop+ as our algorithm so learningrate does not need to be specified and we will stick to the defualt values for learningrate.factor

# we are going to fix threshold to a value larger than the default to speed up the computation process

# we are going to vary the structure of our hidden layers as this is what defines the nueral network structure



# one rule of thumb for choosing the number of hidden nodes is 2/3 the size of the input layer plus the size of the output layer



input.size = ncol(X)

output.size = length(levels(train$Survived)) - 1



hidden.size = round(((2/3) * input.size) + output.size, 0)

hidden.size



rm(input.size, output.size, hidden.size)



# the hidden layer size seems small so we will try higher values as well and let cross validation distinuish if the higher node sizes yeild overfitting



# create parameter combinations to test

	# layers ~ indicate how many layers we will try

		# we will only try 1 and 2 becuase 1 is expected to work for the vast majority of practical applications and 2 allows us to potentially capture some nonlinearlity in the data

	# nodes ~ indicates the total number of hidden layer nodes in our network

	# prop ~ indicates what proportion of nodes to give to the first hidden layer when we are building a 2 hidden layer network



doe = data.table(expand.grid(layers = c(1, 2),

							nodes = c(10, 30, 50),

							prop = c(1/3, 1/2, 2/3)))



# make prop = 1 when layers = 1, and then remove duplicated scenarios



doe = doe[layers == 1, prop := 1]			

doe = doe[!duplicated(doe)]



# build the table for training the model

dat = data.table(X)

dat[, y := Y]



# build a formula string for the training model

rhs = paste(names(X), collapse = " + ")

form = paste("y ~", rhs)



# build models

mods = foreach(i = 1:nrow(doe)) %do%

{

	# compute the hidden layer structure for model i

	

	hidden = vector(length = doe$layers[i])

	hidden[1] = round(doe$nodes[i] * doe$prop[i], 0)

	if(doe$layers[i] == 2) hidden[2] = doe$nodes[i] - hidden[1]

	

	# build model i

	

	output = neuralnet(form, data = dat, 

						hidden = hidden, threshold = 0.25, 

						linear.output = FALSE)

	

	return(output)

}



# ---- ROC --------------------------------------------------------------------------



# create names for each model

mod.name = 1:nrow(doe)



# extract actual values

actuals = dat$y



# compute fitted values of each model

fits = lapply(1:length(mods), function(i) as.numeric(mods[[i]]$net.result[[1]]))



# compute an roc object for each model

rocs = lapply(1:length(fits), function(i) roc(actuals ~ fits[[i]]))



# compute cutoffs to evaluate

cutoffs = c(-Inf, seq(0.01, 0.99, 0.01), Inf)



# compute roc curves for each model

rocs.cutoffs = lapply(1:length(fits), function(i) data.table(t(coords(rocs[[i]], x = cutoffs, input = "threshold", ret = c("threshold", "specificity", "sensitivity", "accuracy")))))



# aggregate results into one table

DT = rbindlist(lapply(1:length(fits), function(i) data.table(mod = mod.name[i], 

																cutoff = rocs.cutoffs[[i]]$threshold, 

																TPR = rocs.cutoffs[[i]]$sensitivity, 

																FPR = 1 - rocs.cutoffs[[i]]$specificity, 

																ACC = rocs.cutoffs[[i]]$accuracy)))



# make mod into a factor data type to facilitate plotting

DT[, mod := factor(mod)]



# compute cut-offs points that maximize accuracy

cuts = as.numeric(sapply(1:length(fits), function(i) 

							coords(rocs[[i]], 

									x = "best", 

									best.weights = c(1, length(rocs[[i]]$cases) / length(rocs[[i]]$response)))[1]))



# plot ROC curves



roc.plot = ggplot(DT, aes(x = FPR, y = TPR, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_abline(slope = 1, size = 1) +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "False Positive", y = "True Positive", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



roc.plot



# plot the accuracy curves

	# the dashed lines indicate the cut-off for each model that maximizes accuracy

	

acc.plot = ggplot(DT, aes(x = cutoff, y = ACC, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_vline(xintercept = cuts, color = ggcolor(length(cuts)), size = 1, linetype = "dashed") +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "Cut-Off", y = "Accuracy", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



acc.plot



# ---- CV ---------------------------------------------------------------------------



# build a function that will report prediction results of our model



nn.pred = function(Xtrain, Ytrain, Xtest, Ytest, negative, hidden, threshold, linear.output, cutoff)

{

	# build the table for training the model

	dat = data.table(Xtrain)

	dat[, y := Ytrain]

	

	# build a formula string for the training model

	rhs = paste(names(Xtrain), collapse = " + ")

	form = paste("y ~", rhs)

	

	# build the training model

	mod = neuralnet(form,

					data = dat,

					hidden = hidden,  

					threshold = threshold, 

					linear.output = linear.output)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(compute(mod, data.table(Xtest))$net.result)

	ynew = as.numeric(ynew >= cutoff)	



	# build a confusion matrix to summarize the performance of our training model

	output = confusionMatrix(Ytest, ynew, negative = negative)

	output = c(output, "AUC" = as.numeric(auc(roc(Ytest ~ ynew))))

	

	# the confusion matrix is the output

	return(output)

}



# compute diagonistic errors for each of the models in doe



nn.diag = foreach(i = 1:nrow(doe)) %do%

{

	# compute the hidden layer structure for model i

	

	hidden = vector(length = doe$layers[i])

	hidden[1] = round(doe$nodes[i] * doe$prop[i], 0)

	if(doe$layers[i] == 2) hidden[2] = doe$nodes[i] - hidden[1]

	

	# perform cross validation

	

	nn.cv = crossval(predfun = nn.pred, X = X, Y = Y, K = K, B = B, verbose = FALSE, negative = 0, 

						hidden = hidden, threshold = 0.25, linear.output = FALSE, cutoff = cuts[i])



	# compute diagnostic errors of cross validation



	output = diag.cv(nn.cv)



	# add columns of parameter values that define model i



	output[, layers := doe$layers[i]] 

	output[, nodes := doe$nodes[i]]

	output[, prop := doe$prop[i]]

	output[, mod := i]

	

	return(output)

}



# combine the list of tables into one table

nn.diag = rbindlist(nn.diag)



# convert nn.diag into long format for plotting purposes

DT = data.table(melt(nn.diag, id.vars = c("stat", "layers", "nodes", "prop", "mod")))



# convert mod into a factor for plotting purposes

DT[, mod := factor(mod)]



# remove Inf values as these don't help

DT = data.table(DT[value < Inf])



# plot barplots of each diagnostic metric



diag.plot = ggplot(DT[stat == "Q1" | stat == "Median" | stat == "Q3"], aes(x = stat, y = value, group = mod, fill = mod)) +

			geom_bar(stat = "identity", position = "dodge", color = "white") +

			labs(x = "Summary Statistic", y = "Value", fill = "Model") + 

			facet_wrap(~variable, scales = "free_y") +

			theme_bw(base_size = 15) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



diag.plot



# ---- Results ----------------------------------------------------------------------



# the roc and accuracy plots show similar performance 

# the diagnostics plot shows that model 8 has better odds ratio while being better tha nmost models across the other metrics



# lets pick model 8



nn.diag = nn.diag[mod == 8]



# rename the model to nn as this is our chosen model



nn.diag[, mod := rep("nn", nrow(nn.diag))]



# get the cutoff value and auc



nn.cutoff = cuts[8]



# build our model



nn.mod = neuralnet(Survived ~ Sexmale + Sexmale.EmbarkedS + Pclass3.EmbarkedS + rescale.Age + rescale.Fare + rescale.SibSp.2 + rescale.Sexmale.Fare + rescale.Pclass3.Age + rescale.Pclass3.Fare, 

					data = train.nn, hidden = c(15, 15), threshold = 0.25, linear.output = FALSE)



# store model diagnostic results



nn.diag[, c("layers", "nodes", "prop") := NULL]

mods.diag = rbind(mods.diag, nn.diag)



# store the model



nn.list = list("mod" = nn.mod, "cutoff" = nn.cutoff, "train" = train.nn, "test" = test.nn)

mods.list$nn = nn.list



# remove objects we no longer need



rm(doe, DT, i, output, nn.pred, nn.cv, X, Y, nn.diag, nn.list, nn.mod,

	acc.plot, actuals, cutoffs, cuts, diag.plot, fits, mods, mod.name, roc.plot, rocs, rocs.cutoffs,

	dat, form, hidden, min.max, nn.cutoff, rhs, train.nn, test.nn)



}



# -----------------------------------------------------------------------------------

# ---- Support Vector Machine Model -------------------------------------------------

# -----------------------------------------------------------------------------------



{



# ---- Set Up -----------------------------------------------------------------------



# we will focus on nu-classification

# this is becuase the parameter 'nu' takes on values within the interval [0, 1] so it makes tuning simplier



# we will focus on the radial basis as our kernel function for projecting our data into a higher demensional feature space becuase:

	# this seems to be the popular method

	# this only requires one parameter: 'gamma'

	# this may end up being less computationally intensive 

		# becuase we are estimating how the data would behave in higher demsional space without directly computing this projection as other kernels do



# hyperparameters of interest:

	# nu:

		# It is used to determine the proportion of support vectors you desire to keep in your solution with respect to the total number of samples in the dataset.

		# If this takes on a large value, then all of your data points could become support vectors, which is overfitting

	# gamma:

		# If gamma is too large, the radius of the area of influence of the support vectors only includes the support vector itself and no amount of regularization with 'cost' will be able to prevent overfitting.

		# When gamma is very small, the model is too constrained and cannot capture the complexity or “shape” of the data. The region of influence of any selected support vector would include the whole training set.

		# grid.py tool from the libSVM package checks values of gamma from 2^-15 to 2^3 

		

# the default values of our parameters are:

	# nu = 0.5

	# gamma = 1 / (number of regressors)



# gamma in our case

1 / (ncol(train) - 1)



# extract our predictors (X) and response (Y)



X = data.table(train)

Y = X$Survived

X[, Survived := NULL]



# lets use cross validation to tune our model



# create parameter combinations to test



doe = expand.grid(gamma = seq(from = 2^(-15), to = 2^3, length.out = 7),

				  nu = seq(from = 0.3, to = 0.7, by = 0.1))



doe = rbind(c(1 / ncol(X), 0.5), doe)

doe = doe[!duplicated(doe),]

doe = data.table(doe)



# build models

mods = lapply(1:nrow(doe), function(i) 

				svm(Survived ~ ., data = train, type = "nu-classification",

					gamma = doe$gamma[i], nu = doe$nu[i], probability = TRUE))



# ---- ROC --------------------------------------------------------------------------



# create names for each model

mod.name = 1:nrow(doe)



# extract actual values

actuals = as.numeric(as.character(train$Survived))



# compute fitted values of each model

fits = lapply(1:length(mods), function(i) as.numeric(attr(predict(mods[[i]], newdata = X, probability = TRUE), "probabilities")[,2]))



# compute an roc object for each model

rocs = lapply(1:length(fits), function(i) roc(actuals ~ fits[[i]]))



# compute cutoffs to evaluate

cutoffs = c(-Inf, seq(0.01, 0.99, 0.01), Inf)



# compute roc curves for each model

rocs.cutoffs = lapply(1:length(fits), function(i) data.table(t(coords(rocs[[i]], x = cutoffs, input = "threshold", ret = c("threshold", "specificity", "sensitivity", "accuracy")))))



# aggregate results into one table

DT = rbindlist(lapply(1:length(fits), function(i) data.table(mod = mod.name[i], 

																cutoff = rocs.cutoffs[[i]]$threshold, 

																TPR = rocs.cutoffs[[i]]$sensitivity, 

																FPR = 1 - rocs.cutoffs[[i]]$specificity, 

																ACC = rocs.cutoffs[[i]]$accuracy)))



# make mod into a factor data type to facilitate plotting

DT[, mod := factor(mod)]



# compute cut-offs points that maximize accuracy

cuts = as.numeric(sapply(1:length(fits), function(i) 

							coords(rocs[[i]], 

									x = "best", 

									best.weights = c(1, length(rocs[[i]]$cases) / length(rocs[[i]]$response)))[1]))



# plot ROC curves



roc.plot = ggplot(DT, aes(x = FPR, y = TPR, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_abline(slope = 1, size = 1) +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "False Positive", y = "True Positive", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 2))



roc.plot



# plot the accuracy curves

	# the dashed lines indicate the cut-off for each model that maximizes accuracy

	

acc.plot = ggplot(DT, aes(x = cutoff, y = ACC, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_vline(xintercept = cuts, color = ggcolor(length(cuts)), size = 1, linetype = "dashed") +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "Cut-Off", y = "Accuracy", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 2))



acc.plot



# ---- CV ---------------------------------------------------------------------------



# build a function that will report prediction results of our model



svm.pred = function(Xtrain, Ytrain, Xtest, Ytest, negative, gamma, nu, cutoff)

{

	# build the table for training the model

	dat = data.table(Xtrain)

	dat[, y := Ytrain]



	# build the training model

	mod = svm(y ~ .,

				data = dat,

				type = "nu-classification",

				gamma = gamma,

				nu = nu,

				probability = TRUE)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(attr(predict(mod, newdata = data.table(Xtest), probability = TRUE), "probabilities")[,2])

	ynew = as.numeric(ynew >= cutoff)

	Ytest = as.numeric(as.character(Ytest))

	

	# build a confusion matrix to summarize the performance of our training model

	output = confusionMatrix(Ytest, ynew, negative = negative)

	output = c(output, "AUC" = as.numeric(auc(roc(Ytest ~ ynew))))

	

	# the confusion matrix is the output

	return(output)

}



# compute diagonistic errors for each of the models in doe



svm.diag = foreach(i = 1:nrow(doe)) %do%

{

	# perform cross validation

	

	svm.cv = crossval(predfun = svm.pred, X = X, Y = Y, K = K, B = B, 

						verbose = FALSE, negative = 0, gamma = doe$gamma[i], nu = doe$nu[i], cutoff = cuts[i])

	

	# compute diagnostic errors of cross validation

	

	output = diag.cv(svm.cv)

	

	# add columns of parameter values that define model i

	

	output[, gamma := doe$gamma[i]] 

	output[, nu := doe$nu[i]]

	output[, mod := i]

	

	return(output)

}



# combine the list of tables into one table

svm.diag = rbindlist(svm.diag)



# convert svm.diag into long format for plotting purposes

DT = data.table(melt(svm.diag, id.vars = c("stat", "gamma", "nu", "mod")))



# convert mod into a factor for plotting purposes

DT[, mod := factor(mod)]



# remove Inf values as these don't help

DT = data.table(DT[value < Inf])



# plot barplots of each diagnostic metric



diag.plot = ggplot(DT[stat == "Q1" | stat == "Median" | stat == "Q3"], aes(x = stat, y = value, group = mod, fill = mod)) +

			geom_bar(stat = "identity", position = "dodge", color = "white") +

			labs(x = "Summary Statistic", y = "Value", fill = "Model") + 

			facet_wrap(~variable, scales = "free_y") +

			theme_bw(base_size = 15) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 2))



diag.plot



# some models have poor sensitivity relative to other models

# and some models have significantly better odds ratio



# ---- Results ----------------------------------------------------------------------



# lets filter out models based on sensitivity and odds ratio



svm.diag[stat == "Median" & or >= 10 & acc >= 0.7 & auc >= 0.77]



# lets go with model 32



svm.diag = svm.diag[mod == 32]



# rename model to svm as this is our chosen model



svm.diag[, mod := rep("svm", nrow(svm.diag))]



# build our model



svm.mod = svm(Survived ~ ., data = train, type = "nu-classification",

				gamma = 2.66668701171875, nu = 0.7, probability = TRUE)



# store model diagnostic results



svm.diag[, c("gamma", "nu") := NULL]

mods.diag = rbind(mods.diag, svm.diag)



# store the model



svm.list = list("mod" = svm.mod, "cutoff" = cuts[32])

mods.list$svm = svm.list



# remove objects we no longer need



rm(doe, DT, i, output, svm.pred, svm.cv, X, Y, svm.diag, svm.list, svm.mod,

	acc.plot, actuals, cutoffs, cuts, diag.plot, fits, mods, mod.name, roc.plot, rocs, rocs.cutoffs)



}



# -----------------------------------------------------------------------------------

# ---- Super Learner Model ----------------------------------------------------------

# -----------------------------------------------------------------------------------



{



# ---- Set Up -----------------------------------------------------------------------



# extract our predictors (X) and response (Y)



X = data.table(train)

Y = as.numeric(as.character(X$Survived))

X[, Survived := NULL]

X = as.matrix(X)



# create Super Learner wrappers for our models of interest



# logistic regression wrapper

my.log = function(Y, X, newX, ...)

{

	# build the table for training the model

	dat = data.table(X)

	dat[, y := factor(Y, levels = 0:1)]

	

	# build the training model

	mod = glm(y ~ ., data = dat,

				family = binomial(link = "logit"), 

				control = list(maxit = 100))

	

	# make predictions with the training model using the test set

	ynew = predict(mod, data.table(newX), type = "response")

	ynew = as.numeric(ynew >= mods.list$log$cutoff)

	

	# return a list of the model (must label as fit) and predictions (must label as pred)

	output = list(pred = ynew, fit = mod)

	return(output)

}



# linear discriminant analysis wrapper

my.lda = function(Y, X, newX, ...)

{

	# build the table for training the model

	dat = data.table(X)

	dat[, y := factor(Y, levels = 0:1)]

	

	# build the training model

	mod = lda(y ~ ., data = dat)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(predict(mod, newdata = data.table(newX))$posterior[,2])

	ynew = as.numeric(ynew >= mods.list$lda$cutoff)

	

	# return a list of the model (must label as fit) and predictions (must label as pred)

	output = list(pred = ynew, fit = mod)

	return(output)

}



# quadratic discriminant analysis wrapper

my.qda = function(Y, X, newX, ...)

{

	# build the table for training the model

	dat = data.table(X)

	dat[, y := factor(Y, levels = 0:1)]

	

	# build the training model

	mod = qda(y ~ ., data = dat)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(predict(mod, newdata = data.table(newX))$posterior[,2])

	ynew = as.numeric(ynew >= mods.list$qda$cutoff)

	

	# return a list of the model (must label as fit) and predictions (must label as pred)

	output = list(pred = ynew, fit = mod)

	return(output)

}



# penalty regression wrapper

my.pen = function(Y, X, newX, ...)

{

	# make Y into a factor data type

	Y = factor(Y, levels = 0:1)

	

	# build the training model

	mod = cv.glmnet(x = X, y = Y, family = "binomial", alpha = 0.1)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(predict(mod, s = mod$lambda.min, newX, type = "response"))

	ynew = as.numeric(ynew >= mods.list$pen$cutoff)

	

	# return a list of the model (must label as fit) and predictions (must label as pred)

	output = list(pred = ynew, fit = mod)

	return(output)

}



# gradient boosting wrapper

my.gbm = function(Y, X, newX, ...)

{

	# build the training model

	mod = xgboost(label = Y, data = X,

					objective = "binary:logistic", eval_metric = "auc",

					eta = 0.1, max_depth = 4,

					nrounds = 500, min_child_weight = 4.714936,

					gamma = 0, verbose = 0)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(predict(mod, newdata = newX))

	ynew = as.numeric(ynew >= mods.list$gbm$cutoff)

	

	# return a list of the model (must label as fit) and predictions (must label as pred)

	output = list(pred = ynew, fit = mod)

	return(output)

}



# random forest wrapper

my.rf = function(Y, X, newX, ...)

{

	# build the table for training the model

	dat = data.table(X)

	dat[, y := factor(Y, levels = 0:1)]

	

	# build the training model

	mod = randomForest(y ~ .,

						data = dat,

						ntree = 500,

						mtry = 3,

						nodesize = 5)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(predict(mod, newdata = data.table(newX), type = "prob")[,2])

	ynew = as.numeric(ynew >= mods.list$rf$cutoff)

	

	# return a list of the model (must label as fit) and predictions (must label as pred)

	output = list(pred = ynew, fit = mod)

	return(output)

}



# neural network wrapper

my.nn = function(Y, X, newX, ...)

{

	# build the table for training the model

	dat = data.table(X)

	dat[, y := Y]

	

	# build the table for the predictions

	new.dat = data.table(newX)

	

	# rescale columns in dat that have a max value above 1 and/or a min value below 0

		# a 0 to 1 scaling is required for neural networks

	dat[, rescale.Age := rescale(Age, to = c(0, 1), from = range(train$Age))]

	dat[, rescale.Fare := rescale(Fare, to = c(0, 1), from = range(train$Fare))]

	dat[, rescale.SibSp.2 := rescale(SibSp.2, to = c(0, 1), from = range(train$SibSp.2))]

	dat[, rescale.Sexmale.Fare := rescale(Sexmale.Fare, to = c(0, 1), from = range(train$Sexmale.Fare))]

	dat[, rescale.Pclass3.Age := rescale(Pclass3.Age, to = c(0, 1), from = range(train$Pclass3.Age))]

	dat[, rescale.Pclass3.Fare := rescale(Pclass3.Fare, to = c(0, 1), from = range(train$Pclass3.Fare))]

	

	# remove columns that violate the scale

	dat[, c("Age", "Fare", "SibSp.2", "Sexmale.Fare", "Pclass3.Age", "Pclass3.Fare") := NULL]

	

	# rescale columns in new.dat that have a max value above 1 and/or a min value below 0

		# a 0 to 1 scaling is required for neural networks

	new.dat[, rescale.Age := rescale(Age, to = c(0, 1), from = range(train$Age))]

	new.dat[, rescale.Fare := rescale(Fare, to = c(0, 1), from = range(train$Fare))]

	new.dat[, rescale.SibSp.2 := rescale(SibSp.2, to = c(0, 1), from = range(train$SibSp.2))]

	new.dat[, rescale.Sexmale.Fare := rescale(Sexmale.Fare, to = c(0, 1), from = range(train$Sexmale.Fare))]

	new.dat[, rescale.Pclass3.Age := rescale(Pclass3.Age, to = c(0, 1), from = range(train$Pclass3.Age))]

	new.dat[, rescale.Pclass3.Fare := rescale(Pclass3.Fare, to = c(0, 1), from = range(train$Pclass3.Fare))]

	

	# remove columns that violate the scale

	new.dat[, c("Age", "Fare", "SibSp.2", "Sexmale.Fare", "Pclass3.Age", "Pclass3.Fare") := NULL]

	

	# build a formula string for the training model

	rhs = paste(names(new.dat), collapse = " + ")

	form = paste("y ~", rhs)

	

	# build the training model

	mod = neuralnet(form,

					data = dat,

					hidden = c(15, 15), 

					threshold = 0.25, 

					linear.output = FALSE)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(compute(mod, new.dat)$net.result)

	ynew = as.numeric(ynew >= mods.list$nn$cutoff)	

	

	# return a list of the model (must label as fit) and predictions (must label as pred)

	output = list(pred = ynew, fit = mod)

	return(output)

}



# support vector machine wrapper

my.svm = function(Y, X, newX, ...)

{

	# build the table for training the model

	dat = data.table(X)

	dat[, y := factor(Y, levels = 0:1)]

	

	# build the training model

	mod = svm(y ~ .,

				data = dat,

				type = "nu-classification",

				gamma = 2.66668701171875, 

				nu = 0.7,

				probability = TRUE)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(attr(predict(mod, newdata = data.table(newX), probability = TRUE), "probabilities")[,2])

	ynew = as.numeric(ynew >= mods.list$svm$cutoff)

	

	# return a list of the model (must label as fit) and predictions (must label as pred)

	output = list(pred = ynew, fit = mod)

	return(output)

}



# create a library of the above wrappers

my.library = list("my.log", "my.pen", "my.lda", "my.qda", "my.gbm", "my.rf", "my.nn", "my.svm")



# build the super learner model

sl.mod = SuperLearner(Y = Y, X = X, family = binomial(), 

						SL.library = my.library, method = "method.AUC")



# ---- ROC --------------------------------------------------------------------------



# create names for each model

mod.name = c("Super Learner")



# extract actual values

actuals = Y



# compute fitted values of each model

fits = list(as.numeric(sl.mod$SL.predict))



# compute an roc object for each model

rocs = lapply(1:length(fits), function(i) roc(actuals ~ fits[[i]]))



# compute cutoffs to evaluate

cutoffs = c(-Inf, seq(0.01, 0.99, 0.01), Inf)



# compute roc curves for each model

rocs.cutoffs = lapply(1:length(fits), function(i) data.table(t(coords(rocs[[i]], x = cutoffs, input = "threshold", ret = c("threshold", "specificity", "sensitivity", "accuracy")))))



# aggregate results into one table

DT = rbindlist(lapply(1:length(fits), function(i) data.table(mod = mod.name[i], 

																cutoff = rocs.cutoffs[[i]]$threshold, 

																TPR = rocs.cutoffs[[i]]$sensitivity, 

																FPR = 1 - rocs.cutoffs[[i]]$specificity, 

																ACC = rocs.cutoffs[[i]]$accuracy)))



# make mod into a factor data type to facilitate plotting

DT[, mod := factor(mod)]



# compute cut-offs points that maximize accuracy

cuts = as.numeric(sapply(1:length(fits), function(i) 

							coords(rocs[[i]], 

									x = "best", 

									best.weights = c(1, length(rocs[[i]]$cases) / length(rocs[[i]]$response)))[1]))



# plot ROC curves



roc.plot = ggplot(DT, aes(x = FPR, y = TPR, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_abline(slope = 1, size = 1) +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "False Positive", y = "True Positive", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



roc.plot



# plot the accuracy curves

	# the dashed lines indicate the cut-off for each model that maximizes accuracy

	

acc.plot = ggplot(DT, aes(x = cutoff, y = ACC, group = mod, color = mod)) +

			geom_line(size = 1) +

			geom_vline(xintercept = cuts, color = ggcolor(length(cuts)), size = 1, linetype = "dashed") +

			scale_y_continuous(labels = percent) +

			scale_x_continuous(labels = percent) +

			labs(x = "Cut-Off", y = "Accuracy", color = "Model") + 

			theme_bw(base_size = 25) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(color = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 1))



acc.plot



# ---- CV ---------------------------------------------------------------------------



# build a function that will report prediction results of our models



sl.pred = function(Xtrain, Ytrain, Xtest, Ytest, negative, cutoff)

{

	# build the training model

	mod = SuperLearner(Y = Ytrain, X = Xtrain, newX = Xtest, family = binomial(), 

						SL.library = my.library, method = "method.AUC")

	

	# make predictions with the training model using the test set

	ynew = as.numeric(mod$SL.predict)

	ynew = as.numeric(ynew >= cutoff)

	

	# build a confusion matrix to summarize the performance of our training model

	output = confusionMatrix(Ytest, ynew, negative = negative)

	output = c(output, "AUC" = as.numeric(auc(roc(Ytest ~ ynew))))

	

	# the confusion matrix is the output

	return(output)

}



# perform cross validation



sl.cv = crossval(predfun = sl.pred, X = X, Y = Y, K = K, B = B, 

					verbose = FALSE, cutoff = cuts, negative = 0)



# compute diagnostic errors of cross validation



sl.diag = diag.cv(sl.cv)



# ---- Results ----------------------------------------------------------------------



# store model diagnostic results



sl.diag[, mod := rep("sl", nrow(sl.diag))]

mods.diag = rbind(mods.diag, sl.diag)



# store the model



sl.list = list("mod" = sl.mod, "cutoff" = cuts)

mods.list$sl = sl.list



# remove objects we no longer need



rm(DT, sl.pred, sl.cv, X, Y, sl.diag, sl.list, sl.mod, my.gbm, my.lda, my.qda, my.log,

	acc.plot, actuals, cutoffs, cuts, fits, mod.name, roc.plot, rocs, rocs.cutoffs,

	my.pen, my.rf, my.nn, my.svm, my.library)



}



# -----------------------------------------------------------------------------------

# ---- Model Selection --------------------------------------------------------------

# -----------------------------------------------------------------------------------



{



# ---- Diagnostics ------------------------------------------------------------------



# convert mods.diag into long format for plotting purposes

DT = data.table(melt(mods.diag, id.vars = c("stat", "mod")))



# convert mod into a factor for plotting purposes

DT[, mod := factor(mod)]



# remove Inf values as these don't help

DT = data.table(DT[value < Inf])



# plot barplots of each diagnostic metric



diag.plot = ggplot(DT[stat == "Q1" | stat == "Median" | stat == "Q3"], aes(x = stat, y = value, group = mod, fill = mod)) +

			geom_bar(stat = "identity", position = "dodge", color = "white") +

			labs(x = "Summary Statistic", y = "Value", fill = "Model") + 

			facet_wrap(~variable, scales = "free_y") +

			theme_bw(base_size = 15) +

			theme(legend.position = "top", legend.key.size = unit(.25, "in"), plot.title = element_text(hjust = 0.5)) +

			guides(fill = guide_legend(override.aes = list(size = 10, linetype = 1), nrow = 2))



diag.plot



# lets go with the super learner



# ---- predict test -----------------------------------------------------------------



# extract our predictors (X) and response (Y)



X = data.table(train)

Y = as.numeric(as.character(X$Survived))

X[, Survived := NULL]

X = as.matrix(X)

newX = as.matrix(test)



# create Super Learner wrappers



# logistic regression wrapper

my.log = function(Y, X, newX, ...)

{

	# build the table for training the model

	dat = data.table(X)

	dat[, y := factor(Y, levels = 0:1)]

	

	# build the training model

	mod = glm(y ~ ., data = dat,

				family = binomial(link = "logit"), 

				control = list(maxit = 100))

	

	# make predictions with the training model using the test set

	ynew = predict(mod, data.table(newX), type = "response")

	ynew = as.numeric(ynew >= mods.list$log$cutoff)

	

	# return a list of the model (must label as fit) and predictions (must label as pred)

	output = list(pred = ynew, fit = mod)

	return(output)

}



# linear discriminant analysis wrapper

my.lda = function(Y, X, newX, ...)

{

	# build the table for training the model

	dat = data.table(X)

	dat[, y := factor(Y, levels = 0:1)]

	

	# build the training model

	mod = lda(y ~ ., data = dat)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(predict(mod, newdata = data.table(newX))$posterior[,2])

	ynew = as.numeric(ynew >= mods.list$lda$cutoff)

	

	# return a list of the model (must label as fit) and predictions (must label as pred)

	output = list(pred = ynew, fit = mod)

	return(output)

}



# quadratic discriminant analysis wrapper

my.qda = function(Y, X, newX, ...)

{

	# build the table for training the model

	dat = data.table(X)

	dat[, y := factor(Y, levels = 0:1)]

	

	# build the training model

	mod = qda(y ~ ., data = dat)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(predict(mod, newdata = data.table(newX))$posterior[,2])

	ynew = as.numeric(ynew >= mods.list$qda$cutoff)

	

	# return a list of the model (must label as fit) and predictions (must label as pred)

	output = list(pred = ynew, fit = mod)

	return(output)

}



# penalty regression wrapper

my.pen = function(Y, X, newX, ...)

{

	# make Y into a factor data type

	Y = factor(Y, levels = 0:1)

	

	# build the training model

	mod = cv.glmnet(x = X, y = Y, family = "binomial", alpha = 0.1)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(predict(mod, s = mod$lambda.min, newX, type = "response"))

	ynew = as.numeric(ynew >= mods.list$pen$cutoff)

	

	# return a list of the model (must label as fit) and predictions (must label as pred)

	output = list(pred = ynew, fit = mod)

	return(output)

}



# gradient boosting wrapper

my.gbm = function(Y, X, newX, ...)

{

	# build the training model

	mod = xgboost(label = Y, data = X,

					objective = "binary:logistic", eval_metric = "auc",

					eta = 0.1, max_depth = 4,

					nrounds = 500, min_child_weight = 4.714936,

					gamma = 0, verbose = 0)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(predict(mod, newdata = newX))

	ynew = as.numeric(ynew >= mods.list$gbm$cutoff)

	

	# return a list of the model (must label as fit) and predictions (must label as pred)

	output = list(pred = ynew, fit = mod)

	return(output)

}



# random forest wrapper

my.rf = function(Y, X, newX, ...)

{

	# build the table for training the model

	dat = data.table(X)

	dat[, y := factor(Y, levels = 0:1)]

	

	# build the training model

	mod = randomForest(y ~ .,

						data = dat,

						ntree = 500,

						mtry = 3,

						nodesize = 5)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(predict(mod, newdata = data.table(newX), type = "prob")[,2])

	ynew = as.numeric(ynew >= mods.list$rf$cutoff)

	

	# return a list of the model (must label as fit) and predictions (must label as pred)

	output = list(pred = ynew, fit = mod)

	return(output)

}



# neural network wrapper

my.nn = function(Y, X, newX, ...)

{

	# build the table for training the model

	dat = data.table(X)

	dat[, y := Y]

	

	# build the table for the predictions

	new.dat = data.table(newX)

	

	# rescale columns in dat that have a max value above 1 and/or a min value below 0

		# a 0 to 1 scaling is required for neural networks

	dat[, rescale.Age := rescale(Age, to = c(0, 1), from = range(train$Age))]

	dat[, rescale.Fare := rescale(Fare, to = c(0, 1), from = range(train$Fare))]

	dat[, rescale.SibSp.2 := rescale(SibSp.2, to = c(0, 1), from = range(train$SibSp.2))]

	dat[, rescale.Sexmale.Fare := rescale(Sexmale.Fare, to = c(0, 1), from = range(train$Sexmale.Fare))]

	dat[, rescale.Pclass3.Age := rescale(Pclass3.Age, to = c(0, 1), from = range(train$Pclass3.Age))]

	dat[, rescale.Pclass3.Fare := rescale(Pclass3.Fare, to = c(0, 1), from = range(train$Pclass3.Fare))]

	

	# remove columns that violate the scale

	dat[, c("Age", "Fare", "SibSp.2", "Sexmale.Fare", "Pclass3.Age", "Pclass3.Fare") := NULL]

	

	# rescale columns in new.dat that have a max value above 1 and/or a min value below 0

		# a 0 to 1 scaling is required for neural networks

	new.dat[, rescale.Age := rescale(Age, to = c(0, 1), from = range(train$Age))]

	new.dat[, rescale.Fare := rescale(Fare, to = c(0, 1), from = range(train$Fare))]

	new.dat[, rescale.SibSp.2 := rescale(SibSp.2, to = c(0, 1), from = range(train$SibSp.2))]

	new.dat[, rescale.Sexmale.Fare := rescale(Sexmale.Fare, to = c(0, 1), from = range(train$Sexmale.Fare))]

	new.dat[, rescale.Pclass3.Age := rescale(Pclass3.Age, to = c(0, 1), from = range(train$Pclass3.Age))]

	new.dat[, rescale.Pclass3.Fare := rescale(Pclass3.Fare, to = c(0, 1), from = range(train$Pclass3.Fare))]

	

	# remove columns that violate the scale

	new.dat[, c("Age", "Fare", "SibSp.2", "Sexmale.Fare", "Pclass3.Age", "Pclass3.Fare") := NULL]

	

	# build a formula string for the training model

	rhs = paste(names(new.dat), collapse = " + ")

	form = paste("y ~", rhs)

	

	# build the training model

	mod = neuralnet(form,

					data = dat,

					hidden = c(15, 15), 

					threshold = 0.25, 

					linear.output = FALSE)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(compute(mod, new.dat)$net.result)

	ynew = as.numeric(ynew >= mods.list$nn$cutoff)	

	

	# return a list of the model (must label as fit) and predictions (must label as pred)

	output = list(pred = ynew, fit = mod)

	return(output)

}



# support vector machine wrapper

my.svm = function(Y, X, newX, ...)

{

	# build the table for training the model

	dat = data.table(X)

	dat[, y := factor(Y, levels = 0:1)]

	

	# build the training model

	mod = svm(y ~ .,

				data = dat,

				type = "nu-classification",

				gamma = 2.66668701171875, 

				nu = 0.7,

				probability = TRUE)

	

	# make predictions with the training model using the test set

	ynew = as.numeric(attr(predict(mod, newdata = data.table(newX), probability = TRUE), "probabilities")[,2])

	ynew = as.numeric(ynew >= mods.list$svm$cutoff)

	

	# return a list of the model (must label as fit) and predictions (must label as pred)

	output = list(pred = ynew, fit = mod)

	return(output)

}



# create a library of the above wrappers

my.library = list("my.log", "my.pen", "my.lda", "my.qda", "my.gbm", "my.rf", "my.nn", "my.svm")



# build the model

my.mod = SuperLearner(Y = Y, X = X, newX = newX, family = binomial(), 

						SL.library = my.library, method = "method.AUC")



my.mod



# extract predictions

my.pred = my.mod$SL.predict

my.pred = as.numeric(my.pred >= mods.list$sl$cutoff)



# build submission

my.pred = data.table(PassengerId = (1:nrow(test)) + nrow(train),

						Survived = my.pred)



write.csv(my.pred, file = "submission-nick-morris.csv", row.names = FALSE)



}                        