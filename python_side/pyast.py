"""
This module is used to parse Python code using ast module and has functions to
test a subset of various dataframe expressions in Pandas.

It also includes the Normalizer class that tries its best to rename the dataframe
variables in Pandas expressions.
"""

import ast
import astor
import pandas as pd

# Find all expressions in querying/filtering/sampling here:
# https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_r.html

# CALLS = ['.', 'loc', 'iloc', 'head', 'shape', 'query', 'drop', 'drop_duplicates', 
#         'sort_values', 'rename', 'assign', 'describe', 'groupby', 'len', 'read_csv', \
#         'to_csv', 'DataFrame']
CALLS = ['read_csv', '.', 'loc', 'iloc', 'head', 'shape', 'query', 'drop', \
        'drop_duplicates', 'describe', 'sort_values', 'isnull', 'notnull', 'isin', 'mean'] 
ASSIGN_CALLS = ['read_csv', 'drop', 'query']

class CallChecker(ast.NodeVisitor):
    """Recursively visits and validates all Calls and their children"""
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
    
    def check(self, node):
        # First check all calls and make sure they're in CALLS
        checker = CallChecker(node)
        if not checker.check():
            return False
        self.visit(node)
        return self.valid
    
    def visit_Subscript(self, node):
        # [, [[, loc and iloc are subsumed as Subscript objects
        # print('subscript', astor.dump(node))
        if 'attr' in node.value.__dict__:
            verb = node.value.attr
            if verb in CALLS:
                self.valid = True
        elif 'slice' in node.__dict__:
            slicing = node.slice
            # Exclude the ExtSlice syntax that works on List but not DataFrame
            num_slices = 0
            for child in ast.iter_child_nodes(node):
                if type(child) == ast.ExtSlice:
                    num_slices += 1
            self.valid = False if num_slices != 0 else True

    def visit_Call(self, node):
        # Only valid if call is in CALLS list
        call = node.func.attr if 'attr' in node.func.__dict__ else node.func
        if call in CALLS:
            self.valid = True
    
    def visit_Attribute(self, node):
        # Standalone Attribute(s) should be accepted as long as they're in our list
        # for e.g. iloc/loc by themselves and not nested in another expression
        if node.attr in CALLS:
            self.valid = True
    
    def visit_Assign(self, node):
        # Excluding assignments for now except for calls in CALLS
        self.valid = False
        # Check rhs of assignment on this visitor
        # self.valid = self.check(node.value)
    
    # Exclude these:

    def visit_AugAssign(self, node): self.valid = False

    def visit_Delete(self, node): self.valid = False

    def visit_Del(self, node): self.valid = False

    def visit_Return(self, node): self.valid = False

    def visit_Import(self, node): self.valid = False
    
    def visit_ImportFrom(self, node): self.valid = False
    
    def visit_FuntionDef(self, node): self.valid = False
    
    def visit_For(self, node): self.valid = False
    
    def visit_While(self, node):self.valid = False
    
    def visit_Try(self, node): self.valid = False

    def visit_Lambda(self, node): self.valid = False

class Normalizer(ast.NodeTransformer):

    def __init__(self, node, *args, **kwargs):
        self.tree = node

    def normalize(self):
        result = self.visit(self.tree)
        renamed = astor.to_source(result)
        return renamed.replace(" ", "").strip()

    def visit_Assign(self, node):
        # print('assign', astor.dump_tree(node))
        node.targets[0].id = "df"
        self.visit(node.value)
        return node
    
    def visit_Subscript(self, node):
        # print('sub', astor.dump_tree(node))
        # rename the main variable and a nested Subscript if there is one
        for child in ast.iter_child_nodes(node):
            if type(child) == ast.Subscript:
                child.value.id = "df"
                self.visit(child)
            elif type(child) == ast.Name:
                child.id = "df"
        # then rename if there are attributes inside of the slice
        if 'slice' in node.__dict__:
            node.slice = self.visit(node.slice)
        # finally, deal with loc / iloc since they will be subsumed by this visit
        if type(node.value) == ast.Attribute:
            node.value = self.visit(node.value)
        return node
    
    def visit_Call(self, node):
        # print('call', astor.dump_tree(node))
        call = node.func.attr if 'attr' in node.func.__dict__ else node.func
        # only rename the variable we're calling on, if it's a valid call 
        if call in CALLS and call != 'read_csv':
            for child in ast.iter_child_nodes(node):
                if type(child) == ast.Attribute:
                    if type(child.value) != ast.Name:
                        child.value = self.visit(child.value)
                    else:
                        child.value.id = "df"
        return node
    
    def visit_Attribute(self, node):
        # print('attr', astor.dump_tree(node))
        if type(node.value) != ast.Name:
            node.value = self.visit(node.value)
        else:
            node.value.id = "df"
        return node

def test_pyast():
    import sys
    # Testing the module
    # TODO make this testing better by incorporating expected values
    test_strings = [ \
        # valid ones
        "train = pd.read_csv('train.csv')",
        "train.shape", 
        "train.head()", 
        "train.loc[1:2, 'a']",
        "train.iloc[:9]", 
        "train = train.loc[1:2, 'a']",
        "train = train.iloc[:9]",
        "train[train.col1 == 1]",
        "train = train[train.col1 == 1]",
        "train.query('col1 == 1 & col2 == 1')", # might need to exclude as this requires parsing Str expr
        "train = train.query('col1 == 1 & col2 == 1')",
        "train[(train.col1 == 1) & (train.col2 == 1)]", 
        "train = train[(train.col1 == 1) & (train.col2 == 1)]",
        "train[['col1']]",
        "train = train[['col1']]",
        "train[['col1', 'col2']]", 
        "train.loc[:, 'col1':'col3']",
        "train[train['col3'] == 1]['col1']",
        "train.drop(cols_to_drop, axis=1)", 
        "train[['col1']].drop_duplicates()", 
        "train = train[['col1']].drop_duplicates()",
        "train[['col1', 'col2']].drop_duplicates()",
        "train = train.drop([cols_to_drop], axis=1)",
        "train[:9][:9]", 
        "train[-5::-2]",
        "train[['a']].shape",
        # "train_df.drop_duplicates(train_df.loc['a':'b', 3:4].query(axis=0), axis=0).drop().loc[[\"Survived\"]]", # complex case handled
        # # tricky renaming cases that don't matter much for now since we focus on dfs only
        # # they are still accepted
        "ax[row, col]",
        "data[cond_notnull & cond_m][\"Age\"]",
        "data[df.cond_notnull & df.cond_m][\"Age\"]",
        # invalid ones
        "train[['col1']].corr()['col2']",
        "train_df.fillna(train_df.loc['a':'b', 3:4].mean(axis=0), axis=0).corr()[\"Survived\"].shape", 
        "xTsigmax += S[i][j]*(x[i]-mu[label][i])*(x[j]-mu[label][j])",
        "del train['Cabin_num1']",
        "train[:2, :-1]",
        "train = train.locc[1:2, 'a']",
        "train = train[['col1']].corr()['col2']",
        # weird renaming cases
        "df.drop([variable],axis=1)",
        "df[[var]][idx]",
        "df.loc[(df['Parch']>4)&(df['Parch']>4),'Parch']"
        ]
    failed = 0
    for t in test_strings:
        test_tree = ast.parse(t)
        normalizer = Normalizer(test_tree)
        tree = normalizer.normalize()
        checker = ASTChecker()
        if not checker.check(test_tree):
            failed += 1
            print('tree not valid', t)
    if failed == 0:
        print('Passed all tests!')
    else:
        print(f'{failed} tests failed!')

if __name__ == '__main__':
    test_pyast()