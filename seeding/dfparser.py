import sys
import os.path
from lark import Lark, Transformer, Visitor
from lark import Tree

# https://github.com/lark-parser/lark/blob/master/lark/grammars/common.lark

pandas_grammar = """
    start: operation
    // after parsing dataframe variable gets labelled as df
    data: WORD -> df
    operation: func | subset
    func: data (head | shape | query | drop | drop_duplicates | sort_values)
    subset: data (col | rows_logical | rows | cols | iloc | loc)   

    // funcs or attrs
    head: "." "head" "(" NUMBER? ")"
    shape: "." "shape"
    query: "." "query" "(" logical_quoted ")"
    drop: "." "drop" "(" drop_list ")"
    drop_duplicates: "." "drop_duplicates" "(" inplace? drop_list? ")"
    drop_list: label | "[" label ("," label?)*  "]" axis?
    axis: "," ("axis" "=" "1" | "axis" "=" "0")
    inplace: (","? "inplace" "=" ("True" | "False"))?
    sort_values: "." "sort_values" "(" sort_list ascending? ")" 
    sort_list: label | "[" label ("," label?)*  "]" 
    ascending: "," "ascending" "=" ("True" | "False")
    
    // single [
    rows_logical: "[" logical ("&" logical)* "]"
    logical: "(" llhs logical_op rrhs ")"
    logical_op: "!=" | "==" | "<" | ">" | "<=" | ">="
    llhs: data col
    logical_quoted: "'" logical_q ("&" logical_q)* "'"
    logical_q: llhs_q logical_op rrhs
    llhs_q: CNAME
    rrhs: NUMBER | SNUMBER | FLOAT | SFLOAT | WORD

    // iloc
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
    col: "." CNAME
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
    %import common.WS
    %ignore WS
"""

pandas_parser = Lark(pandas_grammar, keep_all_tokens=False)

# for r in pandas_parser.rules:
#     print(r)

basics = """df.a
df[[1]]
df.head()
df.head(2)
df[(df.a == 1) & (df.b == 1)]
df.iloc[0,0]
df.iloc[0:,0]
df.iloc[0:1,0]
df.iloc[0:1,0:]
df.iloc[0:1,0:1]
df.iloc[0:,0:]
df.iloc[0,0:1]
df.loc[0,'a']
df.loc[0:,'a']
df.loc[0:1,'a':]
df.loc[0:1,'a']
df.loc[0:1,'a':'b']
df.loc[0:,'a':]
df.loc[0,'a':'b']
df.iloc[0:1].iloc[0:1]
df.loc[0:1].loc[0:1]"""

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
train.drop_duplicates(['col1', 'col2'])
df.sort_values(['col1', 'col2'])
df.sort_values('col1')
df.sort_values('col1', ascending=False)"""

def test():
    for s in basics.split('\n') + pysnips.split('\n'):
        try:
            parsed = pandas_parser.parse(s)
            # print(parsed)
            # print(s)
        except Exception as e:
            print('error!')
            print(s, '->', e)

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
df.iloc[0,0]
df.iloc[0:,0]
df.iloc[0:1,0]
df.iloc[0:1,0:]
df.iloc[0:1,0:1]
df.iloc[0:,0:]
df.iloc[0,0:1]
df.loc[0,'a']
df.loc[0:,'a']
df.loc[0:1,'a':]
df.loc[0:1,'a']
df.loc[0:1,'a':'b']
df.loc[0:,'a':]
df.loc[0,'a':'b']
"""

if __name__ == '__main__':
    test()

# TODO use visitor to rename using df label
# class LeftRightVisitor(Visitor):
#     """
#     Visits a parse tree of a data frame grammar to collect important information
#     for translation
#     """
#     loc_tree = None
#     iloc_tree = None    # iloc
#     left_tree = None    # df[left] or df.iloc[left,]
#     right_tree = None   # df.iloc[, right]
#     labels = []    # df[[label,...]]
#     # indicates  df[[label,...]] since labels can be used elsewhere
#     # for e.g: df.loc[left, [label,..]]
#     cols_tree = None 
#     col_tree = None

#     # only reason for constructor is to reset labels which won't clear 
#     # automatically
#     def __init__(self, *args, **kwargs):
#         self.labels = []
    
#     def loc(self, tree):
#         self.loc_tree = tree.children[0]

#     def iloc(self, tree):
#         self.iloc_tree = tree.children[0]
#         # print(tree.data, tree)

#     def left(self, tree):
#         self.left_tree = tree.children[0]
#         # print(tree.data, tree)
    
#     def right(self, tree):
#         self.right_tree = tree.children[0]
#         # print(tree.data, tree)
    
#     def cols(self, tree):
#         self.cols_tree = tree.children
    
#     def label(self, tree):
#         self.labels.append(tree.children[0])
#         # print(tree.data, tree)

# class IndexTranslator(Transformer):
#     """
#     Translates the index for a range from Pandas to R
#     """
#     def __init__(self, side, loc=False, *args, **kwargs):
#         self.side = side
#         self.loc = loc

#     def label(self, matches):
#         # print('label', matches)
#         if len(matches) == 1 and matches[0].type == 'NUMBER':
#             return str(int(matches[0]) + 1)
#         elif matches[0].type == 'WORD':
#             return "'"+matches[0]+"'"

#     def start_idx(self, matches):
#         return str(int(matches[0]) + 1) if len(matches) == 1 else '1'

#     def end_idx(self, matches):
#         if len(matches) == 1 and not self.loc:
#             return matches[0]
#         elif len(matches) == 1 and self.loc:
#             return str(int(matches[0]) + 1)  # bc loc is end inclusive
#         elif self.side == 'left':
#             return 'nrow(df)'
#         elif self.side == 'right':
#             return 'ncol(df)'
