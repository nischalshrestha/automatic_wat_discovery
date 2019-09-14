"""
This module is used to parse Python code using ast module
"""

import ast
import astor
import pandas as pd
import autopep8

# Find all expressions in querying/filtering/sampling here:
# https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_r.html

class AssignChecker(ast.NodeVisitor):
    
    calls = ['.', 'loc', 'iloc', 'head', 'shape', 'query', 'drop', 'sort_values', 'rename', \
        'assign', 'describe', 'groupby', 'len', 'read_csv', 'to_csv', 'DataFrame']
    valid = False

    def __init__(self, source, *args, **kwargs):
        self.code = source
    
    def check(self):
        self.visit(self.code)
        return self.valid
    
    def visit_Subscript(self, node):
        """[, [[, loc and iloc are Subscript objects"""
        # print('subscript', astor.dump(node))
        if 'attr' in node.value.__dict__:
            verb = node.value.attr
            # print('subscript verb', verb)
            if verb in self.calls:
                self.valid = True
        elif 'slice' in node.__dict__:
            slicing = node.slice
            # print('subscript slice', astor.dump_tree(slicing))
            self.valid = True
    
    def visit_Call(self, node):
        """check if call is in calls list"""
        call = node.func.attr if 'attr' in node.func.__dict__ else node.func
        # print('call', call)
        if call in self.calls:
            self.valid = True
    
    # def visit_Attribute(self, node):
    #     # print(astor.dump_tree(node))
    #     if 'attr' in node.__dict__:
    #         # verb = node.attr
    #         # print('subscript verb', verb)
    #         # if verb in self.calls:
    #         self.valid = True
    #         # self.valid = True

    # def visit_Name(self, node):
    #     self.valid = True

class ASTChecker(ast.NodeVisitor):
    
    calls = ['loc', 'iloc', 'head', 'shape', 'query', 'drop', 'sort_values', 'rename', \
        'assign', 'describe', 'groupby', 'len', 'read_csv', 'to_csv', 'DataFrame']
    valid = False

    def __init__(self, source, *args, **kwargs):
        self.code = source
    
    def check(self):
        self.visit(self.code)
        return self.valid
    
    def visit_Subscript(self, node):
        """[, [[, loc and iloc are Subscript objects"""
        # print('subscript', astor.dump(node))
        if 'attr' in node.value.__dict__:
            verb = node.value.attr
            # print('subscript verb', verb)
            if verb in self.calls:
                self.valid = True
        elif 'slice' in node.__dict__:
            slicing = node.slice
            # print('subscript slice', astor.dump_tree(slicing))
            # if slicing in self.calls:
            self.valid = True
    
    def visit_Call(self, node):
        """check if call is in calls list"""
        call = node.func.attr if 'attr' in node.func.__dict__ else node.func
        # print('call', call)
        if call in self.calls:
            self.valid = True

    def visit_Assign(self, node):
        # print(node.value)
        # Check rhs of assignment
        rhs_checker = AssignChecker(node.value)
        if rhs_checker.check():        
            self.valid = True
    
    def visit_Attribute(self, node):
        # print(astor.dump_tree(node))
        if 'attr' in node.__dict__:
            self.valid = True
    
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
    
def test_pyast():
    import sys
    # cases that are not useful
    # x = a
    test_strings = """x = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]})
        y = x.head()
        z = x[['a']]
        x.head()
        x[0]
        x.a
        x[['a']]
        x.loc['a']
        x.iloc[0]
        x.shape
        x.query('a < 3')
        x.drop('a')
        x.sort_values('a')
        x.rename(columns={'a':'b'})
        x.assign(c=x.a-x.b)
        x.describe()
        x.groupby('a')""".splitlines()
    failed = 0
    for t in test_strings:
        # print(t)
        test_tree = ast.parse(autopep8.fix_code(t))
        # print(astor.dump_tree(test_tree))
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