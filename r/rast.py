"""
This module is used to parse R code using rpy2
"""
import os, sys
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import DataFrame

# this is to silence R errors/warnings
sys.stderr = open(os.devnull, 'w') 

SYMBOL = "SYMBOL"
NUM_CONST = "NUM_CONST"
LEFT_ASSIGN = "LEFT_ASSIGN"
LBB = "LBB"
LB = "\'[\'"
SYMBOL_FUNCTION_CALL = "SYMBOL_FUNCTION_CALL"
DOLLA = "\'$\'"
SPECIAL = "SPECIAL"
EQ_ASSIGN = "EQ_ASSIGN"
CALLS = ["c", "read.csv", "dim", "head", "slice", "filter", "select", "distinct", "arrange", "summary", "summarise"]

def parse_r(parsed_df):
    """
    Parses for following grammar:

    symbol 
    symbol left_assign symbol
    symbol '['
    symbol lbb 
    symbol $ symbol
    symbol '[' symbol_function_call
    symbol special symbol_function_call
    symbol left_assign (symbol_function_call | symbol lb | symbol lbb)
    symbol_function_call

    where symbol_function_call is one of:
    "c", "read.csv", "dim", "head", "slice", "filter", "select", "distinct", "arrange", "summary", "summarise"
    """
    valid = False
    terminals = parsed_df[parsed_df.terminal == 1]
    # print(terminals.token)
    # if sum(terminals.token == LEFT_ASSIGN) > 0 or sum(terminals.token == SPECIAL) > 0 or sum(terminals.token == EQ_ASSIGN) > 0:
    #     valid = False
    # if len(terminals) == 1 and terminals.token[0] == SYMBOL:
    #     # print('variable')
    #     valid = False
    # el
    if len(terminals) == 3:
        # print('column reference')
        if terminals.token[0] == SYMBOL and terminals.token[1] == DOLLA and terminals.token[2] == SYMBOL:
            # print(terminals.text)
            valid = True
    elif len(terminals) > 3:
        if terminals.token[0] == SYMBOL and (terminals.token[1] == LBB or terminals.token[1] == DOLLA):
            # print('column reference')
            # print(terminals.text)
            if terminals.token[3] == LEFT_ASSIGN:
                # print('assignment')
                if terminals.token[4] == SYMBOL_FUNCTION_CALL:
                    # Handle calls
                    valid = is_valid_call(terminals)
                elif not terminals.token[4] == NUM_CONST:
                    valid = True
            else:
                # print(terminals.text)
                valid = True
        elif terminals.token[0] == SYMBOL and terminals.token[1] == LB:
            # print('subsetting')
            valid = True
        elif terminals.token[0] == SYMBOL and terminals.token[1] == LEFT_ASSIGN:
            # print('assignment')
            if terminals.token[2] == SYMBOL_FUNCTION_CALL:
        #         # print('call')
        #         # Handle calls
                valid = is_valid_call(terminals)
            elif len(terminals) >= 4:  # Handle other types of rhs exprs
                if terminals.token[2] == SYMBOL and (terminals.token[3] == LB or terminals.token[3] == LBB):
                    if not terminals.token[4] == NUM_CONST:
                       valid = True
        elif terminals.token[0] == SYMBOL and terminals.token[1] == SPECIAL and terminals.token[2] == SYMBOL_FUNCTION_CALL:
        #     # print('pipe')
        #     # Validate all calls 
            valid = is_valid_call(terminals)
        elif terminals.token[0] == SYMBOL_FUNCTION_CALL:
            # print('call')
            # print(terminals.text)
            valid = is_valid_call(terminals)
    return valid

def is_valid_call(terminals):
    all_symbol_funcs = terminals[terminals.token == SYMBOL_FUNCTION_CALL].text
    if all(x in CALLS for x in all_symbol_funcs.values):
        # print('valid')
        return True
    return False

def check_r(source):
    """
    This function parses R code using rpy2 and returns whether or not it has
    the simple grammar we want to accept.
    """
    # Load R functions
    srcfile = robjects.r['srcfile']
    rparse = robjects.r['parse']
    get_parse_data = robjects.r['getParseData']
    try:
        # Filter out some things like block expressions
        if 'if' in source or 'for' in source or 'while' in source: return False
        if 'ggplot' in source or 'geom' in source or 'facet' in source: return False
        # Call R functions
        sf = srcfile(source)
        rparse(text = source, srcfile = sf)
        # Grab df with parse info
        parse_df = get_parse_data(sf)
        r_df = pandas2ri.rpy2py_dataframe(parse_df)
        # print(r_df)
        return parse_r(r_df)
    except:
        pass
    return False

if __name__ == "__main__":
    test_strings = """df <- data.frame(a = c(1,2,3))
    df$a
    df$a[[10]]
    df$a <- filter(df, a > 1)
    df[['a']]
    df$Title[df$Title == 'Mlle'] <- 'Miss'
    df[1]
    df[1:3]
    df[1:3, 2:4]
    df[c(1,2,3)]
    df <- read.csv(\"test.csv\")
    df <- select(df, 'a')
    df <- df[1]
    df <- df[[1]]
    select(a)
    df %>% select(a) %>% filter(a > 0)
    select(df, a) %>% filter(a > 0)
    select(df, a) %>% filter(a > 0)
    df = df[1]
    select(df, col_one = col1)""".splitlines()
    failed = 0
    for t in test_strings:
        if not check_r(t):
            failed += 1
            print('failed:', t.lstrip())
    if failed == 0:
        print('All tests passed!')
    else:
        print(f'{failed} tests failed!')

