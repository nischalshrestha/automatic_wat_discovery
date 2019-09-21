library(lobstr)
library(rlang)
library(dplyr)

# TODO: Could ignore tidyverse for now but need to write more test cases for subset expressions

# source <- "full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'"
# source <- "full_data$Title[full_data[['Title']] == 'Mlle']"
# source <- "full_data$Title[full_data[['Title']] == 'Mlle' & full_data[0:10,]$Fare > 200, ]$Title"
# source <- "mtcars[mtcars[['cyl']] == 6 & mtcars$disp > 160, ][['cyl']]"
# source <- "full_data[full_data$Fare > 200, ]"
sf <- srcfile(source)
parse(text = source, srcfile=sf)
df <- getParseData(sf)
df <- df[df$text != '',]
# df

# TODO: wrap these into functions
# Figure out the locations of $, [, and [[ which operate on dfs
dbrackets <- which(df$text == "[[")
brackets <- which(df$text == "[")
dollars <- which(df$text == "$")
# replace variables to left of $ if not a ]
dfDollars <- df[dollars - 1, ]
dfDollars <- dfDollars[dfDollars$text != "]", ]
if (nrow(dfDollars) > 0) {
  df[rownames(dfDollars), ]$text <- "df"
} else {
  print("could not replace variable in front of $")
}

# replace variables to left of [
dfBrackets <- df[brackets - 1, ]
twoTokensBack <- brackets - 2
# find the cases where we don't have a [ on a column and replace those vars
dfBracketsNoDollar <- dfBrackets[df[twoTokensBack, ]$text != "$", ]
if (nrow(dfBracketsNoDollar) == 0 && twoTokensBack > 0) { # skip if [ is for a column
  # print("all [ operated on a column")
} else { # otherwise, rename for those that operated directly on a dataframe
  # print("some [ operated on a column")
  if (twoTokensBack > 0) {
    df[rownames(dfBracketsNoDollar), ]$text <- "df"
  } else {
    df[rownames(dfBrackets), ]$text <- "df"
  }
}
# [[ could have a ] next to it when reference a column of a dataframe result
# from a subset operation using [
dfDbrackets <- df[dbrackets - 1, ]
if (nrow(dfDbrackets) > 0) { 
  df[rownames(dfDbrackets[dfDbrackets$text != "]",]), ]$text <- "df"
}

# concatenate all text into one expression
renamed <- paste(df$text, collapse='')
renamed
# can test this later for execution
# eval(parse(text=paste("df<-read.csv(\"../train.csv\");", renamed, collapse='')))
