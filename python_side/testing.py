import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
# from rpy2.robjects.lib import dplyr

from rpy2.robjects.vectors import DataFrame, Vector, FloatVector, IntVector, StrVector, ListVector, FactorVector, BoolVector

######## Packages

# # import R's "base" package
# base = importr('base')
# # import R's "utils" package
# utils = importr('utils')
# tidyverse = importr('tidyverse')
# rpy2_obj = robjects.r['pi']
# print(type(rpy2_obj))

######## Environments
# can create raw R expressions
robjects.r('''
        # create a function `f`
        f <- function(r, verbose=FALSE) {
            if (verbose) {
                cat("I am calling f().\n")
            }
            2 * pi * r
        }
        # call the function `f` with argument value 3
        f(3)
        df <- data.frame(a = c(1,2,3), b = c(4,5,6))
        df2 <- df[1:2,]
        #train <- read.csv('../input/train.csv', stringsAsFactors = F)
        ''')

# r_f = robjects.globalenv['f']
# print(r_f.r_repr())
# print(type(r_f(3)))

# ls from R can be used to poke around in globalenv
# print(robjects.r.ls(robjects.globalenv))

######## Data structures
# PrimitiveType[T]
# ComplexType[dict]
# print(IntVector([1,2,3]))
# print(FloatVector([1.0,2.0,3.0]))
# print(BoolVector([True, False, True]))
# print(StrVector(["True", "False", "True"]))
# print(ListVector({'a': "True", 'b':1, 'c':False}))
# print(DataFrame({'a': IntVector([1,2,3])}))

######## Dataframes

# Comparing dataframes btw Pandas and R
# rdf = robjects.DataFrame({'a': IntVector([1,2,3])})
# df = pd.DataFrame({'a':[1,2,3,4]})
# rdf2 = pandas2ri.rpy2py_dataframe(rdf)
# # print(type(df))
# df_diff = pd.concat([df,rdf2]).drop_duplicates(keep=False)
# print(df_diff.empty)
# if not df_diff.empty:
#     print('diff:\n', df_diff)
# else:
#     print('no diff')

# dplyr 
# mtcars = robjects.r['mtcars']
# dataf = (dplyr.DataFrame(mtcars).
#             filter('gear>3').
#             mutate(powertoweight='hp*36/wt').
#             group_by('gear').
#             summarize(mean_ptw='mean(powertoweight)'))
# print(dataf)

