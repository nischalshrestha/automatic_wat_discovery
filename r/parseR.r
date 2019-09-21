library(lobstr)
library(rlang)

# lobstr::ast(full_data$Title[full_data[['Title']] == 'Mlle'])
# subset_enexpr(full_data$Title[full_data[['Title']] == 'Mlle'])

# TODO: Could ignore tidyverse for now but need to write more test cases for subset expressions

# source <- "full_data$Title[full_dat[['Title']] == 'Mlle' & full_data[0:10,]$Fare > 200, ]$Title"
source <- "mtcars[mtcars[['cyl']] == 6 & mtcars$disp > 160, ][['cyl']]"
# source <- "full_data[full_data$Fare > 200, ]"
sf <- srcfile(source)
parse(text = source, srcfile=sf)
df <- getParseData(sf)
df <- df[df$text != '',]
# df

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
# replace variables to left of [ unless it's working off of a column
dfBrackets <- df[brackets - 1, ]
dfBrackets <- dfBrackets[df[brackets - 2, ]$text == "$", ]
if (nrow(dfBrackets) > 0) { # skip if [ is for a column
  print("could not replace variable in front of [")
} else { # otherwise, rename
  df[brackets - 1, ]$text <- "df"
}
# [[ could have a ] next to it when reference a column of a dataframe result
# from a subset operation using [
dfDbrackets <- df[dbrackets - 1, ]
df[rownames(dfDbrackets[dfDbrackets$text != "]",]), ]$text <- "df"

# test that resulting renamed expression works
renamed <- paste(df$text, collapse='')
eval(parse(text=paste("df<-mtcars;", renamed, collapse='')))
