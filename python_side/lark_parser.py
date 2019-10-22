import sys
import os.path
import time
from lark import Lark, Transformer, Visitor
from lark import Tree

# https://github.com/lark-parser/lark/blob/master/lark/grammars/common.lark

# Note for loc: range with labels unsupported as this requires information
# about the data frame to produce equivalent syntax with c()
# TODO support list of col labels for loc syntax
pandas_grammar = """
    start: operation
    data: WORD                                      -> df
    operation: func | subset
    func: data (head | shape | query | drop | drop_duplicates | sort_values)
    subset: data (col | rows_logical | rows | cols | iloc | loc)   

    // funcs or attrs
    head: "." "head" "(" NUMBER? ")"
    shape: "." "shape"
    query: "." "query" "(" logical_quoted ")"
    drop: "." "drop" "(" drop_list ")"
    drop_list: label | "[" label ("," label?)*  "]" ("," ("axis" "=" "1" | "axis" "=" "0"))?
    drop_duplicates: "." "drop_duplicates" "(" inplace? drop_list? ")"
    inplace: (","? "inplace" "=" ("True" | "False"))?
    sort_values: "." "sort_values" "(" sort_list ")" 
    sort_list: label ("," "ascending" "=" ("True" | "False"))? | "[" label ("," label?)*  "]"
    
    col: "." CNAME
    // single [
    rows_logical: "[" logical ("&" logical)* "]"
    // logical expr
    logical: "(" llhs logical_op rrhs ")"
    logical_op: "!=" | "==" | "<" | ">" | "<=" | ">="
    llhs: data col
    // logical quoted expr
    logical_quoted: "'" logical_q ("&" logical_q)* "'"
    logical_q: llhs_q logical_op rrhs
    llhs_q: CNAME
    // both logical exprs use the same value type for comparison
    rrhs: NUMBER | SNUMBER | FLOAT | SFLOAT | WORD

    rows: iloc
    cols: "[[" label ("," label?)* "]]" (shape | drop_duplicates)?

    iloc: "." "iloc" _rows_cols iloc*
    _rows_cols: "[" left ("," right?)? "]"
    left: _index
    right: _index

    // loc 
    loc: ".loc" _lrows_cols loc*
    _lrows_cols: "[" lleft ("," lright?)? "]"           
    lleft: _lindex                                  -> left
    lright: _rindex                                 -> right
    _lindex: lrange | label
    _rindex: word | rrange
    rrange: word ":" word?                           -> range
    lrange: lstart ":" lend                          -> range
    lstart: (NUMBER | "'" NUMBER "'")?               -> start_idx
    lend:   (NUMBER | "'" NUMBER "'")?               -> end_idx

    // common
    _index: range | NUMBER
    range: start_idx ":" end_idx
    start_idx: NUMBER*
    end_idx: NUMBER*
    label: word | "'" NUMBER "'" | NUMBER
    word: "'" CNAME "'"

    %import common.LETTER
    %import common.INT -> NUMBER
    %import common.SIGNED_NUMBER -> SNUMBER
    %import common.FLOAT -> FLOAT
    %import common.SIGNED_FLOAT -> SFLOAT
    %import common.WORD
    %import common.CNAME
    %import common._STRING_INNER
    %import common._STRING_ESC_INNER
    %import common.WS
    %ignore WS
"""

pandas_parser = Lark(pandas_grammar, keep_all_tokens=False)
# print(pandas_parser.terminals)

# for r in pandas_parser.rules:
#     print(r)

# print(len(pandas_parser.rules))
# print(pandas_parser.parse("df['1'] df[1]").pretty())
# print(pandas_parser.parse('df[[1]]').pretty())
# print(pandas_parser.parse("df.a").pretty())
# print(pandas_parser.parse("df.head()").pretty())
# print(pandas_parser.parse("df.head(2)").pretty())

# logical condition test
# print(pandas_parser.parse("df[(df.a == 1) & (df.b == 1) ]").pretty())

# iloc and loc test
# """
# print(pandas_parser.parse("df.iloc[0:1]").pretty())
# print(pandas_parser.parse("df.iloc[0:1].iloc[0:1]").pretty())

pysnips = """train.shape
train[['col1']].shape
train.head()
train.iloc[:8]
train.iloc[:7]
train.query('col1 == 1 & col3 == 1')
train.iloc[0:5, 0:3]
train[(train.col1 == 1) & (train.col3 == 1)]
train[['col1', 'col2']]
train[['col1']]
train.loc[:, 'col1':'col3']
train.loc[1:2, 'col1']
train.drop(['col1', 'col2'], axis=1)
train[['col1', 'col2']].drop_duplicates()
train[['col1']].drop_duplicates()
train.drop_duplicates(inplace=True)
train.drop_duplicates(['col1'])
df.sort_values(['col1', 'col2'])
df.sort_values('col1')
df.sort_values('col1', ascending=False)"""

# start = time.time()
# for s in pysnips.split('\n'):
#     try:
#         parsed = pandas_parser.parse(s)
#         # print(parsed)
#         # print(s)
#     except Exception as e:
#         print('woa', e)
#         pass
# print("time taken:", time.time()-start)

def parse(snippet):
    try:
        parsed = pandas_parser.parse(snippet)
        # print(parsed)
        # print(s)
        return True
    except Exception as e:
        # print('woa', e)
        return False

"""
print(pandas_parser.parse("df.iloc[0,0]").pretty())
print(pandas_parser.parse("df.iloc[0:,0]").pretty())
print(pandas_parser.parse("df.iloc[0:1,0]").pretty())
print(pandas_parser.parse("df.iloc[0:1,0:]").pretty())
print(pandas_parser.parse("df.iloc[0:1,0:1]").pretty())
print(pandas_parser.parse("df.iloc[0:,0:]").pretty())
print(pandas_parser.parse("df.iloc[0,0:1]").pretty())
print(pandas_parser.parse("df.loc[0,'a']").pretty())
print(pandas_parser.parse("df.loc[0:,'a']").pretty())
print(pandas_parser.parse("df.loc[0:1,'a':]").pretty())
print(pandas_parser.parse("df.loc[0:1,'a']").pretty())
print(pandas_parser.parse("df.loc[0:1,'a':'b']").pretty())
print(pandas_parser.parse("df.loc[0:,'a':]").pretty())
print(pandas_parser.parse("df.loc[0,'a':'b']").pretty())
"""