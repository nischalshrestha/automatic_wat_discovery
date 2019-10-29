import sys
import os.path
from lark import Lark, Transformer, Visitor
from lark import Tree

# https://github.com/lark-parser/lark/blob/master/lark/grammars/common.lark

# CALLS = ["c", "read.csv", "dim", "head", "slice", "filter", "select", "subset", \
#     "distinct", "arrange", "order", "desc", "summary", "summarise", "mutate", "rename", \
#     "is.na", "which", "%in%"]

r_grammar = """
    start: operation
    // after parsing dataframe variable gets labelled as df
    data: WORD -> df
    operation: func | data subset?
    func: negate? (c | isna | which | head | dim | slice | select | filter | _subset | distinct | arrange | order | mean)
    negate: "-" | "!"
    subset: (col | cols | default_subset) subset*

    // funcs or attrs
    // todo: add nested functions
    c: "c" "(" (label) ("," (label))* ")"
    isna: "is.na" "(" (data | data subset) ")"
    which: "which" "(" logical (logical_op logical)* ")"
    head: "head" "(" operation ("," NUMBER)? ")"
    dim: "dim" "(" operation ")" // for now we'll accept the operations
    slice: "slice" "(" (data | subset | func) "," range ")"
    select: "select" "(" (data | subset | func) (("," label)+ | "," (range | func) | ("," select_label)+) ")"
    filter: "filter" "(" (data | subset | func) ("," logical)+ ")"
    _subset: "subset" "(" data "," logical ")"
    distinct: "distinct" "(" (data | subset | func) ")"
    arrange: "arrange" "(" (data | subset | func) (("," (label | col_name))* | ("," desc)?) ")"
    desc: "desc" "(" col_name ")"
    order: "order" "(" negate? (data | data col) ("," negate? data col)* ")"
    select_label: negate? label
    mean: "mean" "(" operation ("," "na.rm" "=" ("TRUE" | "FALSE"))? ")"

    // single [
    col: "$" CNAME 
    col_name: CNAME
    cols: "[" c "]" | "[[" label "]]" 
    default_subset: "[" word "]" | _rows_cols | "[" logical (logical_op logical)* ","? "]"
    logical: func | llhs compare_op rrhs
    logical_op: "&" | "|"
    compare_op: "!=" | "==" | "<" | ">" | "<=" | ">=" | "%in%"
    llhs: data col | CNAME | func
    rrhs: NUMBER | SNUMBER | FLOAT | SFLOAT | WORD | func

    _rows_cols: "[" left ("," right?)? "]"
    left: _index | logical | func
    right: _index | label | func

    // common
    _index: range | NUMBER
    range: start_idx ":" end_idx | CNAME ":" CNAME
    start_idx: NUMBER+
    end_idx: NUMBER+
    label: word | "'" NUMBER "'" | NUMBER
    // double quote literal needs to be escaped
    word: CNAME | "'" CNAME "'" | "\\"" CNAME "\\""

    %import common.LETTER
    %import common.INT -> NUMBER
    %import common.SIGNED_NUMBER -> SNUMBER
    %import common.FLOAT -> FLOAT
    %import common.SIGNED_FLOAT -> SFLOAT
    %import common.WORD
    %import common.CNAME
    %import common.WS
    %ignore WS
"""

r_parser = Lark(r_grammar, keep_all_tokens=False)

rsnips = """dim(train)
dim(train[c('col1')])
head(train)
slice(train, 1:8)
train[1:8, ]
slice(train, 1:7)
train[1:7, ]
filter(train, col1 == 1, col3 == 1)
train[1:5, 1:3]
train[train$col1 == 1 & train$col3 == 1, ]
train[which(train$col1 == 1 & train$col3 == 1), ]
select(train, col1, col2)
train[c('col1')]
train[['col1']]
select(train, col1:col3)
train[2:3, 'col1']
select(train, -c(col1, col2))
distinct(select(train, col1, col2))
distinct(select(train, col1))
arrange(train, col1, col2)
train[order(train$col1, train$col2), ] 
train[order(-train$col1), ]
train$col1[train$col3 == 1]
train[train$col3 == 1, ]$col1
train[is.na(train$col1), ]
train[train$col2 %in% c('ID_3', 'ID_4'), ]
train[!is.na(train['col2']) & train$col2 %in% c('ID_3', 'ID_4')]
train[!is.na(train['col2']) & train$col2 %in% c('ID_3', 'ID_4'),]
mean(train$col1)
mean(train$col1, na.rm=TRUE)"""

def test():
    for s in rsnips.split('\n'):
        try:
            parsed = r_parser.parse(s)
            # print(parsed)
            # print(s)
        except Exception as e:
            print('error!')
            print(s, '->', e)

def parse(snippet):
    try:
        parsed = r_parser.parse(snippet)
        # print(parsed)
        # print(s)
        return True
    except Exception as e:
        print('woa', e)
        return False

if __name__ == '__main__':
    test()

