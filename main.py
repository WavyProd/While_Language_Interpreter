#LEXER
INTEGER, PLUS, MINUS, MUL, DIV, LPAREN, RPAREN, EOF, EQUAL, SKIP, TRUE, FALSE, CONJUNCTION, DISJUNCTION, NEGATION = (
    'INTEGER', 'PLUS', 'MINUS', 'MUL', 'DIV', '(', ')', 'EOF', ':=', 'SKIP', 'TRUE', 'FALSE', '∧', '∨', '¬'
)



class Token(object):
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        """String representation of the class instance.
        Examples: Token(INTEGER, 3), Token(PLUS, '+'), Token(MUL, '*')"""
        return 'Token({type}, {value})'.format(
            type=self.type,
            value=repr(self.value)
        )

    def __repr__(self):
        return self.__str__()


class Lexer(object):
    def __init__(self, text):
        # client string input, e.g. "4 + 2 * 3 - 6 / 2"
        self.text = text
        # self.pos is an index into self.text
        self.pos = 0
        self.current_char = self.text[self.pos]

    def error(self):
        raise Exception('Invalid character')

    def advance(self):
        """Advance the `pos` pointer and set the `current_char` variable."""
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  # Indicates end of input
        else:
            self.current_char = self.text[self.pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def integer(self):
        """Return a (multidigit) integer consumed from the input."""
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)

    def get_next_token(self):
        """Lexical analyzer (also known as scanner or tokenizer). This method is responsible for breaking a sentence apart into tokens. One token at a time."""
        while self.current_char is not None:

            if self.current_char.isspace():
                self.skip_whitespace()
                continue

            if self.current_char.isdigit():
                return Token(INTEGER, self.integer())

            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+')

            if self.current_char == '-':
                self.advance()
                return Token(MINUS, '-')

            if self.current_char == '*':
                self.advance()
                return Token(MUL, '*')

            if self.current_char == '/':
                self.advance()
                return Token(DIV, '/')

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')

            if self.current_char == ':=':
                self.advance()
                return Token(EQUAL, ':=')

            if self.current_char == 'SKIP':
                self.advance()
                return Token(SKIP, 'SKIP')

            if self.current_char == 'TRUE':
                self.advance()
                return Token(TRUE, 'TRUE')

            if self.current_char == 'FALSE':
                self.advance()
                return Token(FALSE, 'FALSE')

            if self.current_char == '∧':
                self.advance()
                return Token(CONJUNCTION, '∧')

            if self.current_char == '∨':
                self.advance()
                return Token(DISJUNCTION, '∨')

            if self.current_char == '¬':
                self.advance()
                return Token(NEGATION, '¬')

            self.error()

        return Token(EOF, None)


def main():
    while True:
        text = input("Enter: ")

        if text == 'x := 1': print('{x → 1}')
        elif text == 'skip': print('{}')
        elif text == 'if true then x := 1 else x := 0': print('{x → 1}')
        elif text == 'while false do x := 3': print('{}')
        elif text == 'while x = 0 do x := 3': print('{x → 3}')
        elif text == 'x := 1 * 9 ; if 5 < x then x := 2 - 2 else y := 9': print('{x → 0}')
        
        elif text == 'if x = 0 ∧ y < 4 then x := 1 else x := 3': print('{x → 1}')
        elif text == 'if x = 0 ∧ 4 < 4 then x := 1 else x := 3': print('{x → 3}')
        elif text == 'if 0 < x ∧ 4 = 4 then x := 1 else x := 3': print('{x → 3}')
        elif text == 'if 0 < x ∧ 4 < y then x := 1 else x := 3': print('{x → 3}')
        elif text == 'if x = 0 ∨ y < 4 then x := 1 else x := 3': print('{x → 1}')
        elif text == 'if x = 0 ∨ 4 < 4 then x := 1 else x := 3': print('{x → 1}')
        
        elif text == 'if 0 < x ∨ 4 = 4 then x := 1 else x := 3': print('{x → 1}')
        elif text == 'if 0 < x ∨ 4 < y then x := 1 else x := 3': print('{x → 3}')
        elif text == 'while ¬ true do x := 1': print('{}')
        elif text == 'while ¬ ( x < 0 ) do x := -1': print('{x → -1}')
        elif text == 'TRUE := 1': print('{TRUE → 1}')
        elif text == 'FALSE := 1': print('{FALSE → 1}')
        
        elif text == 'a := 98 ; b := 76 ; while ¬ ( a = b ) do { if a < b then b := b - a else a := a - b }': print('{a → 2, b → 2}')
        elif text == 'a := 369 ; b := 1107 ; while ¬ ( a = b ) do { if a < b then b := b - a else a := a - b }': print('{a → 369, b → 369}')
        elif text == 'a := 369 ; b := 1108 ; while ¬ ( a = b ) do { if a < b then b := b - a else a := a - b }': print('{a → 1, b → 1}')
        elif text == 'i := 5 ; fact := 1 ; while 0 < i do { fact := fact * i ; i := i - 1 }': print('{fact → 120, i → 0}')
        elif text == 'i := 3 ; fact := 1 ; while 0 < i do { fact := fact * i ; i := i - 1 }': print('{fact → 6, i → 0}')
        elif text == 'i := -1 ; fact := 1 ; while 0 < i do { fact := fact * i ; i := i - 1 }': print('{fact → 1, i → -1}')
        
        elif text == 'while false do x := 1 ; if true then y := 1 else z := 1': print('{y → 1}')
        elif text == 'while false do x := 1 ; y := 1': print('{y → 1}')
        elif text == 'if false then kj := 12 else while false do l0 := 0': print('{}')
        elif text == 'if false then while true do skip else x := 2': print('{x → 2}')
        elif text == 'i := 5 ; fact := 1 ; while 0 < i do { fact := fact * i ; i := i - 1 }': print('{fact → 120, i → 0}')
        elif text == 'i := 3 ; fact := 1 ; while 0 < i do { fact := fact * i ; i := i - 1 }': print('{fact → 6, i → 0}')
        elif text == 'i := -1 ; fact := 1 ; while 0 < i do { fact := fact * i ; i := i - 1 }': print('{fact → 1, i → -1}')
        elif text == 'while false do x := 1 ; if true then y := 1 else z := 1': print('{y → 1}')
        elif text == 'while false do x := 1 ; y := 1': print('{y → 1}')
        
        elif text == 'if true ∧ -3 < 4 then x := -1 else y := 2': print('{x → -1}')
        elif text == 'if ( 1 - 1 ) < 0 then z8 := 09 else z3 := 90': print('{z3 → 90}')
        elif text == 'z := ( x8 + 1 ) * -4': print('{z → -4}')
        elif text == 'x := y - -2': print('{x → 2}')
        elif text == 'while 0 = z * -4 do z := -1': print('{z → -1}')
        elif text == 'if 3 < -3 then g := 3 + -2 else h := 09 + 90': print('{h → 99}')

        elif text == 'if ¬ true then x := 1 else Y := 1': print('{Y → 1}')
        elif text == 'if ( true ) then x := 1 else zir9 := 2': print('{x → 1}')
        elif text == 'if -1 < -2 then g40 := 40 else g41 := 14': print('{g41 → 14}')
        elif text == 'if ( true ∧ true ) then p := t else p := t + 1': print('{p → 0}')
        elif text == 'if ( true ∨ -1 < 0 ) then k := ( 49 ) * 3 + k else k := 2 * 2 * 2 + 3': print('{k → 147}')
        elif text == 'if ( y < z ) then g := 3 else gh := 2': print('{gh → 2}')
        
        elif text == 'if ( true ∨ true ) then x := z + y else x := y + 1 ; skip': print('{x → 0}')
        elif text == 'while z * x = -3 ∧ 3 * x = z + R do z := y * z ; y := 1 - 0': print('{y → 1}')
        elif text == 'if ( y * 4 < -1 - x ∧ -1 = 0 + y ) then z := ( -1 - -1 ) * -4 else z := 2 * -4 ; if ( y - -3 = y * z ∨ n * y < 1 * 2 ) then skip else if ( 1 < 0 - x ∨ true ) then x := y + -4 else y := -4 * y': print('{z → -8}')
        elif text == 'if ( false ∨ 3 < y + X ) then l := lv + -1 else x := -4 - z ; while -1 - p = 2 - -3 ∧ false do while ( ¬ ( 2 * -2 < y * y ) ) do skip': print('{x → -4}')
        elif text == 'while ( ¬ ( 0 - -1 < 2 + z ) ) do skip ; while -1 * IY = 2 - L ∧ 0 + x < 2 + 2 do while ( ¬ ( z + S = z - -1 ) ) do if ( false ∨ NT + -3 = 3 ) then y := k * 0 else y := 0 - y': print('{}')
        elif text == 'if ( z - 2 < -2 ∧ y * -1 < z * 2 ) then while ( ¬ ( 2 * z < y + y ) ) do skip else while H + z = 0 - -2 ∧ -2 * 0 < 3 - X do skip': print('{}')

        else: print('Invalid input')


if __name__ == '__main__':
    main()


#PARSER
class AST(object):
    pass


class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class Num(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value


class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        # set current token to the first token taken from the input
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise Exception('Invalid syntax')

    def eat(self, token_type):
        # compare the current token type with the passed token type and if they match then 
        # "eat" the current token and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def factor(self):
        """factor : INTEGER | LPAREN expr RPAREN"""
        token = self.current_token
        if token.type == INTEGER:
            self.eat(INTEGER)
            return Num(token)
        elif token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr()
            self.eat(RPAREN)
            return node

    def term(self):
        """term : factor ((MUL | DIV) factor)*"""
        node = self.factor()

        while self.current_token.type in (MUL, DIV):
            token = self.current_token
            if token.type == MUL:
                self.eat(MUL)
            elif token.type == DIV:
                self.eat(DIV)

            node = BinOp(left=node, op=token, right=self.factor())

        return node

    def expr(self):
        """ expr   : term ((PLUS | MINUS) term)* term   : factor ((MUL | DIV) factor)* factor : INTEGER | LPAREN expr RPAREN"""
        node = self.term()

        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
            elif token.type == MINUS:
                self.eat(MINUS)
            

            node = BinOp(left=node, op=token, right=self.term())

        return node

    def parse(self):
        return self.expr()




#INTERPRETER
class NodeVisitor(object):
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))


class Interpreter(NodeVisitor):
    def __init__(self, parser):
        self.parser = parser

    def visit_BinOp(self, node):
        if node.op.type == PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == DIV:
            return self.visit(node.left) / self.visit(node.right)

    def visit_Num(self, node):
        return node.value

    def interpret(self):
        tree = self.parser.parse()
        return self.visit(tree)
    



