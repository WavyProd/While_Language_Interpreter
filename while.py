import sys
from lark import Lark
from collections import OrderedDict

class InterpAST():
    def __init__(self, parser):
        self.parser = parser
        self.state = {}

    def checkVariable(self, var):
        if var in self.state:
            return self.state[var]
        return 0
        
    def booleanCHECK(self, oper, L, R):
        if oper == "<":
            return L < R
        
        elif oper == "=":
            return L == R
        
        elif oper == ">":
            return L > R
    

    def interpret(self, tree):
        stmt_oper = tree.data

        # NOT
        if stmt_oper == "not":
            if not self.interpret(tree.children[0]):
                return 1
            
            else:
                return 0 
        
        # TRUE condition
        elif stmt_oper == "true_cond":
            return 1
        
        # FALSE condition
        elif stmt_oper == "false_cond":
            return 0
        
        # AND ^
        elif stmt_oper == "and":
            return self.interpret(tree.children[0]) and self.interpret(tree.children[1])
        
        # OR v
        elif stmt_oper == "or":
            return self.interpret(tree.children[0]) or self.interpret(tree.children[1])
        
        # get number
        elif stmt_oper == "number":
            return int(tree.children[0])
        
        # peek for variable
        elif stmt_oper == "variable":
            return self.checkVariable(tree.children[0])
        
        # Variable assignment
        elif stmt_oper == "assigning":
            variable = tree.children[0].children[0]
            self.state[variable] = self.interpret(tree.children[1])

            return
        
        # COMPARE left & right
        elif stmt_oper == "compare":
            coper = tree.children[1]
            L = self.interpret(tree.children[0])
            R = self.interpret(tree.children[2])
            
            return self.booleanCHECK(coper, L, R)
        
        # ARITH operations
        elif stmt_oper in {"add", "sub", "mul", "div"}:
            L = self.interpret(tree.children[0])
            R = self.interpret(tree.children[1])
            
            if stmt_oper == 'add':
                return L + R
            
            elif stmt_oper == 'mul':
                return L * R
            
            elif stmt_oper == 'sub':
                return L - R
            
            elif stmt_oper == 'div':
                return round(L / R)
        
        # SKIP stmt: do nothing just return
        elif stmt_oper == "skip":
            return 
        
        # regular stmt
        elif stmt_oper == "sstatement":
            self.interpret(tree.children[0])
            self.interpret(tree.children[1])
            
            return
        
        # IF stmt
        elif stmt_oper == "if":
            if self.interpret(tree.children[0]):
                # condition satisfied & execute stmts
                self.interpret(tree.children[1])

            elif not self.interpret(tree.children[0]) and len(tree.children) == 3:
                # condition unsatisfied & execute stmts after if-else stmt
                self.interpret(tree.children[2])
            
            return 
        
        # TERNARY if
        elif stmt_oper == "ternary":
            var = tree.children[0].children[0]
            condition = self.interpret(tree.children[1])
            
            self.state[var] = self.interpret(tree.children[2]) if condition else self.interpret(tree.children[3])
            
            return
        

        # WHILE statement
        elif stmt_oper == "swhile":
            if self.interpret(tree.children[0]):
                # condition statisfied & execute statements
                self.interpret(tree.children[1])
                self.interpret(tree)
            
            elif not self.interpret(tree.children[0]) and tree.children[1].data == "sstatement":
                self.interpret(tree.children[1].children[1])

            return
        elif stmt_oper == "cwhile":
            if self.interpret(tree.children[0]):
                self.interpret(tree.children[1])
                self.interpret(tree)
            
            return
        
        

    def showStore(self):
        itt = OrderedDict(sorted(self.state.items()))
        store = ", ".join(str(variable) + " → " + str(data) for variable, data in itt.items())
        
        return "{" + store + "}"

    def read(self, input):
        ast = self.parser.parse(input)
        self.interpret(ast)

        return self.showStore()
          
if __name__ == '__main__':
    
    for input in sys.stdin:
        wp = Lark.open('while.lark', parser='lalr')
        output = InterpAST(wp).read(input)
        print(output)
