"""
This module is used to parse Python code using ast module and has functions to
test a subset of various dataframe expressions.
"""

import ast
import astor
import pandas as pd
import autopep8

# TODO use NodeTransformer to normalize df variables and args names
# TODO generate random values for df and args
# TODO use eval for a simple testing harness

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
CALLS = ['.', 'loc', 'iloc', 'head', 'shape', 'query', 'drop', 'drop_duplicates', 'read_csv'] 
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
            # print('call', call)
            self.valid = True

class CallChecker(ast.NodeVisitor):

    valid = True

    def __init__(self, source, *args, **kwargs):
        self.code = source

    def recursive(func):
        """ decorator to make visitor work recursive """
        def wrapper(self, node):
            func(self, node)
            for child in ast.iter_child_nodes(node):
                self.visit(child)
        return wrapper
    
    def check(self):
        self.visit(self.code)
        return self.valid
    
    @recursive
    def visit_Call(self, node):
        """check if call is in calls list"""
        # print('call', astor.dump_tree(node))
        call = node.func.attr if 'attr' in node.func.__dict__ else node.func
        if not call in CALLS:
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
            # print('subscript slice', astor.dump_tree(slicing))
            # if slicing in CALLS:
            self.valid = True
    
    def visit_Call(self, node):
        """check if call is in calls list"""
        call = node.func.attr if 'attr' in node.func.__dict__ else node.func
        # print('call', call)
        # print('call node', astor.dump_tree(node))
        if call in CALLS:
            self.valid = True
    
    def visit_Attribute(self, node):
        # print('attr', astor.dump_tree(node))
        if 'attr' in node.__dict__ and node.attr in CALLS:
            self.valid = True
    
    # Excluding assignments for now
    def visit_Assign(self, node):
        self.valid = False
        # # print(node.value)
        # # Check rhs of assignment (
        # rhs_checker = AssignChecker(node.value)
        checker = CallChecker(node.value)
        if checker.check():
            self.valid = True
            # self.valid = False
        # if rhs_checker.check():        

    def visit_AugAssign(self, node):
        self.valid = False

    def visit_Delete(self, node):
        self.valid = False

    def visit_Del(self, node):
        self.valid = False

    def visit_ExtSlice(self, node):
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
    

class SimpleVisitor(ast.NodeVisitor):
    """ simple visitor for comparison """
    stack = []

    def recursive(func):
        """ decorator to make visitor work recursive """
        def wrapper(self, node):
            func(self, node)
            for child in ast.iter_child_nodes(node):
                self.visit(child)
        return wrapper

    def visit_Assign(self,node):
        """ visit a Assign node """
        print(type(node).__name__)

    def visit_BinOp(self, node):
        """ visit a BinOp node """
        print(type(node).__name__)

    @recursive
    def visit_Subscript(self, node):
        """[, [[, loc and iloc are Subscript objects"""
        print(type(node).__name__)
        print(astor.dump_tree(node))
        # if 'attr' in node.value.__dict__:
        #     verb = node.value.attr
        #     # print('subscript verb', verb)
        #     # print('subscript node', astor.dump_tree(node))
        #     if verb in CALLS:
        #         self.valid = True
        # elif 'slice' in node.__dict__:
        #     slicing = node.slice
        #     print('subscript', astor.dump(slicing))
        #     print('subscript slice', astor.dump_tree(slicing))
        #     # if slicing in CALLS:
        #     self.valid = True

    @recursive
    def visit_Call(self,node):
        """ visit a Call node """
        print(type(node).__name__, node.func.attr)

    def visit_Lambda(self,node):
        """ visit a Function node """
        print(type(node).__name__)

    def visit_FunctionDef(self,node):
        """ visit a Function node """
        print(type(node).__name__)

def recurse_tree(test_tree):
    """ recurse on tree """
    for child in ast.iter_child_nodes(test_tree):
        # print('yo', child)
        # The 4 we need to examine: Attribute, Subscript, Index, Call
        # if type(child) == ast.Attribute:
        #     print('----')
        #     print(type(child).__name__, '\n', astor.dump_tree(child))
        #     # print(child.func.attr)
        # elif type(child) == ast.Subscript:
        #     print('----')
        #     print(type(child).__name__, '\n', astor.dump_tree(child))
        # elif type(child) == ast.Index:
        #     print('----')
        #     print(type(child).__name__,'\n', astor.dump_tree(child))
        # elif type(child) == ast.Call:
        #     print('----')
        #     print(type(child).__name__)
        # elif type(child) == ast.Name:
        #     print('----')
        #     print(type(child).__name__,'\n', astor.dump_tree(child))
        # elif type(child) == ast.Num:
        #     print('----')
        #     print(type(child).__name__,'\n', astor.dump_tree(child))
        recurse_tree(child)


# def create_df(row, col):
#     return pd.DataFrame()

def eval_expr(df, expr):
    """
    Evals a an expression given a dataframe.
    Currently, this does not factor in args for expr
    """
    code_str = "%s ; %s" % (df, expr)

    code_obj = compile(code_str, '', 'single')
    print(code_obj)
    eval(code_obj)

# eval_expr("df = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]})", "df.head()")

def test_pyast():
    import sys
    test_strings = [ \
        # valid ones
        "df = pd.read_csv('blah.csv')",
        "df.shape", 
        "df.head()", 
        "df.loc[1:2, 'a']",
        "df.iloc[:9]", 
        "df[df.col1 == 1]",
        "df.query('col1 == 1 & col2 == 1')", # might need to exclude as this requires parsing Str expr
        "df[(df.col1 == 1) & (df.col2 == 1)]", 
        "df[['col1']]",
        "df[['col1', 'col2']]", 
        "df.loc[:, 'col1':'col3']",
        "df.drop(cols_to_drop, axis=1)", 
        "df[['col1']].drop_duplicates()", 
        "df[['col1', 'col2']].drop_duplicates()",
        "df[:9][:9]", 
        "df[['a']].shape",
        "train_df.drop_duplicates(train_df.loc['a':'b', 3:4].query(axis=0), axis=0).drop().loc[[\"Survived\"]]", # complex case handled
        # invalid ones
        "df[['col1']].corr()['col2']",
        "train_df.fillna(train_df.loc['a':'b', 3:4].mean(axis=0), axis=0).corr()[\"Survived\"].shape", 
        "xTsigmax += S[i][j]*(x[i]-mu[label][i])*(x[j]-mu[label][j])",
        "del train['Cabin_num1']",
        "df[:2, :-1]" # TODO ignore these extslice type operations that stand by themselves
        ]
    
    failed = 0
    for t in test_strings:
        # print(t)
        test_tree = ast.parse(autopep8.fix_code(t))
        # eval_expr(t)
        # eval_expr("df = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]}); print(df)", t)
        # recurse_tree(test_tree)
        # self.visit(child)
        # simple_visitor = SimpleVisitor()
        # simple_visitor.visit(test_tree)

        checker = ASTChecker(test_tree)
        if not checker.check():
            failed += 1
            print('tree not valid', t)
        else:
            print('valid', t)
    if failed == 0:
        print('Passed all tests!')
    else:
         print(f'{failed} tests failed!')

if __name__ == '__main__':
    test_pyast()