"""
This module is used to parse Python code using ast module and has functions to
test a subset of various dataframe expressions.
"""

import ast
import astor
import pandas as pd
import autopep8

# Notes:

# calls like head() query() drop() have a Name value with id and attr:
# for e.g. df.head() -> Call(func=Attribute(value=Name(id='df'), attr='head'), args=[], keywords=[])
# w. actual args like df.query('col1 == 1 & col2 == 1'):
# Call(func=Attribute(value=Name(id='df'), attr='query'), args=[Str(s='col1 == 1 & col2 == 1')], keywords=[])

# Subscripts look different, they have an Index object which contains different types
# In the case of logical expressions inside [], you have Index(value=BinOp(...))
# df[(df.col1 == 1) & (df.col2 == 1)]
# Index(
# value=BinOp(
#     left=Compare(left=Attribute(value=Name(id='df'), attr='col1'), ops=[Eq], comparators=[Num(n=1)]),
#     op=BitAnd,
#     right=Compare(left=Attribute(value=Name(id='df'), attr='col2'), ops=[Eq], comparators=[Num(n=1)])))

# In the case of one or more labels inside [], you have Index(value=List(...))
# df[['col1']]
# Index(value=List(elts=[Str(s='col1')]))
# df[['col1', 'col2]]
# Index(value=List(elts=[Str(s='col1'), Str(s='col2')]))
# df[df.col1 == 1]
# Index(value=Compare(left=Attribute(value=Name(id='df'), attr='col1'), ops=[Eq], comparators=[Num(n=1)]))

# In the case of nums:
# df.iloc[:9]
# Subscript(value=Attribute(value=Name(id='df'), attr='iloc'), slice=Slice(lower=None, upper=Num(n=9), step=None))
# df.loc['a']
# Subscript(value=Attribute(value=Name(id='df'), attr='loc'), slice=Index(value=Str(s='a')))
# df.loc[1:2, 'a'] contains the extended slice 
# Subscript(value=Attribute(value=Name(id='df'), attr='loc'),
# slice=ExtSlice(dims=[Slice(lower=Num(n=1), upper=Num(n=2), step=None), Index(value=Str(s='a'))]))

# Find all expressions in querying/filtering/sampling here:
# https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_r.html

# CALLS = ['.', 'loc', 'iloc', 'head', 'shape', 'query', 'drop', 'drop_duplicates', 
#         'sort_values', 'rename', 'assign', 'describe', 'groupby', 'len', 'read_csv', \
#         'to_csv', 'DataFrame']
CALLS = ['.', 'loc', 'iloc', 'head', 'shape', 'query', 'drop', 'drop_duplicates', 'describe', 'read_csv'] 
ASSIGN_CALLS = ['read_csv']

class AssignChecker(ast.NodeVisitor):

    valid = False

    def __init__(self, source, *args, **kwargs):
        self.code = source
    
    def check(self):
        self.visit(self.code)
        return self.valid
    
    # Ignoring subscripts for now
    # def visit_Subscript(self, node):
    #     """[, [[, loc and iloc are Subscript objects"""
    #     # print('subscript', astor.dump(node))
    #     if 'attr' in node.value.__dict__:
    #         verb = node.value.attr
    #         # print('subscript verb', verb)
    #         if verb in CALLS:
    #             self.valid = True
    #     elif 'slice' in node.__dict__:
    #         slicing = node.slice
    #         # print('subscript slice', astor.dump_tree(slicing))
    #         self.valid = True
    
    def visit_Call(self, node):
        """check if call is in calls list"""
        call = node.func.attr if 'attr' in node.func.__dict__ else node.func
        # print('call', call)
        if call in ASSIGN_CALLS:
            self.valid = True

class CallChecker(ast.NodeVisitor):

    valid = True

    def __init__(self, source, *args, **kwargs):
        self.code = source

    def check(self):
        self.visit(self.code)
        return self.valid

    def recursive(func):
        """Decorator to make visitor work recursive"""
        def wrapper(self, node):
            func(self, node)
            for child in ast.iter_child_nodes(node):
                self.visit(child)
        return wrapper
    
    @recursive
    def visit_Call(self, node):
        """check if call is in calls list"""
        # print('call', astor.dump_tree(node))
        call = node.func.attr if 'attr' in node.func.__dict__ else node.func
        if call not in CALLS:
            self.valid = False
            
class ASTChecker(ast.NodeVisitor):
    
    valid = False

    def __init__(self, source, *args, **kwargs):
        self.code = source
    
    def check(self):
        checker = CallChecker(self.code)
        if not checker.check():
            return False
        self.visit(self.code)
        return self.valid
    
    def visit_Subscript(self, node):
        """[, [[, loc and iloc are Subscript objects"""
        # print('subscript', astor.dump(node))
        if 'attr' in node.value.__dict__:
            verb = node.value.attr
            # print('subscript verb', astor.dump_tree(node))
            if verb in CALLS:
                self.valid = True
        elif 'slice' in node.__dict__:
            slicing = node.slice
            # print('subscript slice', astor.dump_tree(node))
            num_slices = 0
            for child in ast.iter_child_nodes(node):
                if type(child) == ast.ExtSlice:
                    num_slices += 1
            self.valid = False if num_slices != 0 else True
    
    def visit_Call(self, node):
        """check if call is in calls list"""
        call = node.func.attr if 'attr' in node.func.__dict__ else node.func
        # print('call', call)
        # print('call node', astor.dump_tree(node))

        if call in CALLS:
            self.valid = True
    
    def visit_Attribute(self, node):
        # print('attr', astor.dump_tree(node))
        if node.attr in CALLS:
            self.valid = True
    
    # Excluding assignments for now except for calls in CALLS
    def visit_Assign(self, node):
        self.valid = False
        # print(node.value)
        # print(node.targets)
        # Check rhs of assignment
        # rhs_checker = AssignChecker(node.value)
        # if rhs_checker.check() and type(node.targets[0]) == ast.Name:
        #     self.valid = True
    
    def visit_AugAssign(self, node):
        self.valid = False

    def visit_Delete(self, node):
        self.valid = False

    def visit_Del(self, node):
        self.valid = False

    def visit_Return(self, node):
        self.valid = False

    def visit_Import(self, node):
        self.valid = False
    
    def visit_ImportFrom(self, node):
        self.valid = False
    
    def visit_FuntionDef(self, node):
        self.valid = False
    
    def visit_For(self, node):
        self.valid = False
    
    def visit_While(self, node):
        self.valid = False
    
    def visit_Try(self, node):
        self.valid = False

    def visit_Lambda(self, node):
        self.valid = False

class Normalizer(ast.NodeTransformer):

    def __init__(self, node, *args, **kwargs):
        self.tree = node

    def normalize(self):
        result = self.visit(self.tree)
        return astor.to_source(result)

    def visit_Assign(self, node):
        self.visit(node.targets[0])
        return node
    
    def visit_Name(self, node):
        # print(astor.dump_tree(node))
        # print(node.id)
        node.id = "mslacc" 
        return node

def test_pyast():
    import sys
    test_strings = [ \
        # valid ones
        "train = pd.read_csv('train.csv')",
        "train.shape", 
        "train.head()", 
        "train.loc[1:2, 'a']",
        "train.iloc[:9]", 
        "train[train.col1 == 1]",
        "train.query('col1 == 1 & col2 == 1')", # might need to exclude as this requires parsing Str expr
        "train[(train.col1 == 1) & (train.col2 == 1)]", 
        "train[['col1']]",
        "train[['col1', 'col2']]", 
        "train.loc[:, 'col1':'col3']",
        "train.drop(cols_to_drop, axis=1)", 
        "train[['col1']].drop_duplicates()", 
        "train[['col1', 'col2']].drop_duplicates()",
        "train[:9][:9]", 
        "train[-5::-2]",
        "train[['a']].shape",
        "train_df.drop_duplicates(train_df.loc['a':'b', 3:4].query(axis=0), axis=0).drop().loc[[\"Survived\"]]", # complex case handled
        # tricky renaming cases that don't matter much for now since we focus on dfs only
        # they are still accepted
        "ax[row, col]",
        "data[cond_notnull & cond_m][\"Age\"]",
        "data[df.cond_notnull & df.cond_m][\"Age\"]",
        # invalid ones
        "train[['col1']].corr()['col2']",
        "train_df.fillna(train_df.loc['a':'b', 3:4].mean(axis=0), axis=0).corr()[\"Survived\"].shape", 
        "xTsigmax += S[i][j]*(x[i]-mu[label][i])*(x[j]-mu[label][j])",
        "del train['Cabin_num1']",
        "train[:2, :-1]",
        ]
    failed = 0
    for t in test_strings:
        test_tree = ast.parse(autopep8.fix_code(t))
        normalizer = Normalizer(test_tree)
        tree = normalizer.normalize()
        # print(t, tree)
        # recurse_tree(test_tree)
        checker = ASTChecker(test_tree)
        if not checker.check():
            failed += 1
            print('tree not valid', t)
    if failed == 0:
        print('Passed all tests!')
    else:
        print(f'{failed} tests failed!')

if __name__ == '__main__':
    test_pyast()