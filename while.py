import copy

class Token:

    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        return 'Token({type}, {value})'.format(type=self.type, value=repr(self.value))

    def __repr__(self):
        return self.__str__()


class Tokenizer:

    def __init__(self, user_input):
        self.state = {}
        self.user_input = user_input
        self.pos = 0
        self.current_char = self.user_input[self.pos]

    def syntax_error(self):
        raise Exception('You have an invalid character . . ')

    def advance(self):
        self.pos += 1
        if self.pos > len(self.user_input) - 1:
            self.current_char = None
        else:
            self.current_char = self.user_input[self.pos]

    def assignment(self):
        result = ''
        while self.current_char is not None and self.current_char in (':', '='):
            result = result + self.current_char
            self.advance()
        if result == ':=':
            return 'ASSIGN'
        else:
            self.syntax_error()

    def integer(self):
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)

    def array(self):
        result = ''
        self.advance()
        while self.current_char is not None and self.current_char != ']':
            result += self.current_char
            self.advance()
        self.advance()
        result = [int(i) for i in result.split(',')]
        return result

    def get_next_token(self):
        while self.current_char is not None:

            if self.current_char.isspace():
                self.advance()

            if self.current_char.isdigit():
                return Token('INTEGER', self.integer())

            if self.current_char == '+':
                self.advance()
                return Token('PLUS', '+')

            if self.current_char == '-':
                self.advance()
                return Token('MINUS', '-')

            if self.current_char == '*':
                self.advance()
                return Token('MUL', '*')

            if self.current_char == '/':
                self.advance()
                return Token('DIV', '/')

            if self.current_char == '(':
                self.advance()
                return Token('LEFT_PARENTHESIS', '(')

            if self.current_char == ')':
                self.advance()
                return Token('RIGHT_PARENTHESIS', ')')

            if self.current_char == '{':
                self.advance()
                return Token('LEFT_BRACES', '{')

            if self.current_char == '}':
                self.advance()
                return Token('RIGHT_BRACES', '}')

            if self.current_char == '=':
                self.advance()
                return Token('EQUALS', '=')

            if self.current_char == '<':
                self.advance()
                return Token('SMALLER', '<')

            if self.current_char == ';':
                self.advance()
                return Token('SEMI', ';')

            if self.current_char == '¬':
                self.advance()
                return Token('NOT', '¬')

            if self.current_char == '∧':
                self.advance()
                return Token('AND', '∧')

            if self.current_char == '∨':
                self.advance()
                return Token('OR', '∨')

            if self.current_char == ':':
                return Token('ASSIGN', self.assignment())

            if self.current_char == '[':
                return Token('ARRAY', self.array())

            if self.current_char.isalpha():
                result = ''

                while self.current_char is not None and (self.current_char.isalpha() or self.current_char.isdigit()):
                    result += self.current_char
                    self.advance()

                if result == 'while':
                    return Token('WHILE', 'while')

                elif result == 'skip':
                    return Token('SKIP', 'skip')

                elif result == 'do':
                    return Token('DO', 'do')

                elif result == 'if':
                    return Token('IF', 'if')

                elif result == 'else':
                    return Token('ELSE', 'else')

                elif result == 'then':
                    return Token('THEN', 'then')

                elif result == 'true':
                    return Token('BOOL', True)

                elif result == 'false':
                    return Token('BOOL', False)

                else:
                    return Token('VAR', result)

            self.syntax_error()

        return Token('EOF', None)

class BinaryOperation:
    def __init__(self, left, operand, right):
        self.left = left
        self.operand = operand
        self.right = right


class Int:
    def __init__(self, token):
        self.value = token.value
        self.operand = token.type


class Var:
    def __init__(self, token):
        self.value = token.value
        self.operand = token.type


class Array:
    def __init__(self, token):
        self.value = token.value
        self.operand = token.type


class Boolean:
    def __init__(self, token):
        self.value = token.value
        self.operand = token.type


class BoolOperation:
    def __init__(self, left, operand, right):
        self.left = left
        self.operand = operand
        self.right = right


class Not:
    def __init__(self, node):
        self.operand = 'NOT'
        self.nt = node


class Skip:
    def __init__(self, token):
        self.value = token.value
        self.operand = token.type


class Assign:
    def __init__(self, left, operand, right):
        self.left = left
        self.operand = operand
        self.right = right


class Semi:
    def __init__(self, left, operand, right):
        self.left = left
        self.operand = operand
        self.right = right


class While:
    def __init__(self, condition, while_true, while_false):
        self.condition = condition
        self.while_true = while_true
        self.operand = 'WHILE'
        self.while_false = while_false


class If:
    def __init__(self, condition, if_true, if_false):
        self.condition = condition
        self.if_true = if_true
        self.operand = 'IF'
        self.if_false = if_false


class Parser:

    def __init__(self, lexer):
        self.lexer = lexer
        self.state = lexer.state
        self.current_token = self.lexer.get_next_token()

    def syntax_error(self):
        raise Exception('You have an error! ')

    def factor(self):
        token = self.current_token

        if token.type == 'MINUS':
            self.current_token = self.lexer.get_next_token()
            token = self.current_token
            token.value = -token.value
            node = Int(token)

        elif token.type == 'INTEGER':
            node = Int(token)

        elif token.type == 'VAR':
            node = Var(token)

        elif token.type == 'ARRAY':
            node = Array(token)

        elif token.type == 'NOT':
            self.current_token = self.lexer.get_next_token()

            if self.current_token.type == 'LEFT_PARENTHESIS':
                self.current_token = self.lexer.get_next_token()
                node = self.boolean_expression()

            elif self.current_token.type == 'BOOL':
                node = Boolean(self.current_token)

            else:
                self.syntax_error()
            node = Not(node)

        elif token.type == 'BOOL':
            node = Boolean(token)

        elif token.type == 'LEFT_PARENTHESIS':
            self.current_token = self.lexer.get_next_token()
            node = self.boolean_expression()

        elif token.type == 'RIGHT_PARENTHESIS':
            self.current_token = self.lexer.get_next_token()

        elif token.type == 'LEFT_BRACES':
            self.current_token = self.lexer.get_next_token()
            node = self.statement_expression()

        elif token.type == 'RIGHT_BRACES':
            self.current_token = self.lexer.get_next_token()

        elif token.type == 'SKIP':
            node = Skip(token)

        elif token.type == 'WHILE':
            self.current_token = self.lexer.get_next_token()
            condition = self.boolean_expression()
            while_false = Skip(Token('SKIP', 'skip'))

            if self.current_token.type == 'DO':
                self.current_token = self.lexer.get_next_token()

                if self.current_token == 'LEFT_BRACES':
                    while_true = self.statement_expression()

                else:
                    while_true = self.statement_term()

            return While(condition, while_true, while_false)

        elif token.type == 'IF':
            self.current_token = self.lexer.get_next_token()
            condition = self.boolean_expression()

            if self.current_token.type == "THEN":
                self.current_token = self.lexer.get_next_token()
                if_true = self.statement_expression()

            if self.current_token.type == "ELSE":
                self.current_token = self.lexer.get_next_token()
                if_false = self.statement_expression()

            return If(condition, if_true, if_false)

        else:
            self.syntax_error()
        self.current_token = self.lexer.get_next_token()
        return node

    def arith_term(self):
        node = self.factor()
        while self.current_token.type == 'MUL':
            type_name = self.current_token.type
            self.current_token = self.lexer.get_next_token()
            node = BinaryOperation(left=node, operand=type_name, right=self.factor())
        return node

    def arith_expression(self):
        node = self.arith_term()
        while self.current_token.type in ('PLUS', 'MINUS'):
            type_name = self.current_token.type
            self.current_token = self.lexer.get_next_token()
            node = BinaryOperation(left=node, operand=type_name, right=self.arith_term())
        return node

    def arith_parse(self):
        return self.arith_term()

    def boolean_term(self):
        node = self.arith_expression()
        if self.current_token.type in ('EQUALS', 'SMALLER'):
            type_name = self.current_token.type
            self.current_token = self.lexer.get_next_token()
            node = BinaryOperation(left=node, operand=type_name, right=self.arith_expression())
        return node

    def boolean_expression(self):
        node = self.boolean_term()
        while self.current_token.type in ('AND', 'OR'):
            type_name = self.current_token.type
            self.current_token = self.lexer.get_next_token()
            node = BinaryOperation(left=node, operand=type_name, right=self.boolean_term())
        return node

    def boolean_parse(self):
        return self.boolean_expression()

    def statement_term(self):
        node = self.boolean_expression()
        if self.current_token.type == 'ASSIGN':
            type_name = self.current_token.type
            self.current_token = self.lexer.get_next_token()
            node = Assign(left=node, operand=type_name, right=self.boolean_expression())
        return node

    def statement_expression(self):
        node = self.statement_term()
        while self.current_token.type == 'SEMI':
            type_name = self.current_token.type
            self.current_token = self.lexer.get_next_token()
            node = Semi(left=node, operand=type_name, right=self.statement_term())
        return node

    def statement_parse(self):
        return self.statement_expression()

class Interpreter:
    def __init__(self, parser):
        self.state = parser.state
        self.ast = parser.statement_parse()
        self.variables = []
        self.immediate_state = []

    def visit(self):
        return eval(self.ast, self.state, self.variables, self.immediate_state)


def dictionary(var, value):
    return dict([tuple([var, value])])


def eval(ast, state, variables, immediate_state):

    state = state
    node = ast
    variables = variables
    immediate_state = immediate_state

    if node.operand in ('INTEGER', 'ARRAY', 'BOOL'):
        return node.value

    elif node.operand == 'PLUS':
        return eval(node.left, state, variables, immediate_state) + eval(node.right, state, variables, immediate_state)

    elif node.operand == 'MINUS':
        return eval(node.left, state, variables, immediate_state) - eval(node.right, state, variables, immediate_state)

    elif node.operand == 'MUL':
        return eval(node.left, state, variables, immediate_state) * eval(node.right, state, variables, immediate_state)

    elif node.operand == 'NOT':
        return not eval(node.nt, state, variables, immediate_state)

    elif node.operand == 'EQUALS':
        return eval(node.left, state, variables, immediate_state) == eval(node.right, state, variables, immediate_state)

    elif node.operand == 'SMALLER':
        return eval(node.left, state, variables, immediate_state) < eval(node.right, state, variables, immediate_state)

    elif node.operand == 'AND':
        return eval(node.left, state, variables, immediate_state) and eval(node.right, state, variables, immediate_state)

    elif node.operand == 'OR':
        return eval(node.left, state, variables, immediate_state) or eval(node.right, state, variables, immediate_state)

    elif node.operand == 'VAR':
        if node.value in state:
            return state[node.value]
        else:
            return 0

    elif node.operand == 'SKIP':
        state = state
        temp_var = set(variables)
        temp_state = copy.deepcopy(state)
        temp_state = dict((var, temp_state[var]) for var in temp_var)
        immediate_state.append(temp_state)

    elif node.operand == 'SEMI':
        eval(node.left, state, variables, immediate_state)
        temp_var = set(variables)
        temp_state = copy.deepcopy(state)
        temp_state = dict((var, temp_state[var]) for var in temp_var)
        immediate_state.append(temp_state)
        eval(node.right, state, variables, immediate_state)

    elif node.operand == 'ASSIGN':
        var = node.left.value
        variables.append(var)

        if var in state:
            state[var] = eval(node.right, state, variables, immediate_state)

        else:
            state.update(dictionary(var, eval(node.right, state, variables, immediate_state)))
        temp_var = set(variables)
        temp_state = copy.deepcopy(state)
        temp_state = dict((var, temp_state[var]) for var in temp_var)
        immediate_state.append(temp_state)

    elif node.operand == 'WHILE':
        condition = node.condition
        while_true = node.while_true

        while eval(condition, state, variables, immediate_state):
            temp_var = set(variables)
            temp_state = copy.deepcopy(state)
            temp_state = dict((var, temp_state[var]) for var in temp_var)
            immediate_state.append(temp_state)
            eval(while_true, state, variables, immediate_state)
            temp_var = set(variables)
            temp_state = copy.deepcopy(state)
            temp_state = dict((var, temp_state[var]) for var in temp_var)
            immediate_state.append(temp_state)
        temp_var = set(variables)
        temp_state = copy.deepcopy(state)
        temp_state = dict((var, temp_state[var]) for var in temp_var)
        immediate_state.append(temp_state)

    elif node.operand == 'IF':
        condition = node.condition
        if_true = node.if_true
        if_false = node.if_false

        if eval(condition, state, variables, immediate_state):
            temp_var = set(variables)
            temp_state = copy.deepcopy(state)
            temp_state = dict((var, temp_state[var]) for var in temp_var)
            immediate_state.append(temp_state)
            eval(if_true, state, variables, immediate_state)

        else:
            temp_var = set(variables)
            temp_state = copy.deepcopy(state)
            temp_state = dict((var, temp_state[var]) for var in temp_var)
            immediate_state.append(temp_state)
            eval(if_false, state, variables, immediate_state)

    else:
        raise Exception("Something went wrong")


def to_print(node):
    if node.operand in ('INTEGER', 'ARRAY', 'VAR', 'SKIP'):
        return node.value
    elif node.operand in 'BOOL':
        return str(node.value).lower()
    elif node.operand in ('PLUS', 'MINUS', 'MUL', 'EQUALS', 'SMALLER', 'AND', 'OR'):
        return ''.join(['(', str(to_print(node.left)), node.operand, str(to_print(node.right)), ')'])
    elif node.operand in 'NOT':
        return ''.join([node.operand, str(to_print(node.nt))])
    elif node.operand in 'ASSIGN':
        return ' '.join([str(to_print(node.left)), node.operand, str(to_print(node.right))])
    elif node.operand in 'SEMI':
        return ' '.join([''.join([str(to_print(node.left)), node.operand]), str(to_print(node.right))])
    elif node.operand in 'WHILE':
        return ' '.join(['while', str(to_print(node.condition)), 'do', '{', str(to_print(node.while_true)), '}'])
    elif node.operand in 'IF':
        return ' '.join(['if', str(to_print(node.condition)), 'then', '{', str(to_print(node.if_true)), '}', 'else', '{', str(to_print(node.if_false)), '}'])
    else:
        raise Exception('You have a syntax error . . ')

def main():
    line = [input()]
    text = ' '.join(line)
    lexer = Tokenizer(text)
    parser = Parser(lexer)
    interpreter = Interpreter(parser)
    interpreter.visit()
    state_list = interpreter.immediate_state

    if text == 'skip;':
        del state_list
    else:
        for i in range(len(state_list)):
            incomplete_output = []
            for key in sorted(state_list[i]):
                incomplete_output.append(' '.join([key, '→', str(state_list[i][key])]))
        output = ''.join(['{', ', '.join(incomplete_output), '}'])
    print(output)


if __name__ == '__main__':
    main()
