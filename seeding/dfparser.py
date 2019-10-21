import sys
import os.path
from lark import Lark, Transformer, Visitor
from lark import Tree

# https://github.com/lark-parser/lark/blob/master/lark/grammars/common.lark

# Note for loc: range with labels unsupported as this requires information
# about the data frame to produce equivalent syntax with c()
# TODO support list of col labels for loc syntax
pandas_grammar = """
    start: operation
    data: "df"                                      -> df
    operation: func | subset
    func: data (head)
    subset: data (col | rows_logical | rows | cols | iloc | loc)   

    head: "." "head" "(" NUMBER? ")"
    col: "." WORD
    rows_logical: "[" logical ("&" logical)* "]"
    rows: iloc
    cols: "[[" label ("," label?)* "]]"

    logical: "(" llhs logical_op rrhs ")"
    logical_op: "!=" | "==" | "<" | ">" | "<=" | ">="
    llhs: data col
    rrhs: NUMBER | SNUMBER | FLOAT | SFLOAT | WORD

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

    iloc: "." "iloc" _rows_cols iloc*
    _rows_cols: "[" left ("," right?)? "]"
    left: _index
    right: _index

    _index: range | NUMBER
    range: start_idx ":" end_idx
    start_idx: NUMBER*
    end_idx: NUMBER*
    label: word | "'" NUMBER "'" | NUMBER
    word: "'" WORD "'"

    %import common.LETTER
    %import common.INT -> NUMBER
    %import common.SIGNED_NUMBER -> SNUMBER
    %import common.FLOAT -> FLOAT
    %import common.SIGNED_FLOAT -> SFLOAT
    %import common.WORD
    %import common.WS
    %ignore WS
"""

pandas_parser = Lark(pandas_grammar, keep_all_tokens=False)
# print(pandas_parser.terminals)
print(pandas_parser.rules[3])
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
print(pandas_parser.parse("df.iloc[0:1].iloc[0:1]").pretty())
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
