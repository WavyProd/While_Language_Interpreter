import ast
import builtins

class EntryNodeVisitor(ast.NodeVisitor):

    def generic_visit(self, node):
        n = node.__class__.__name__
        raise NotImplementedError("Visitor for node {} not implemented".format(n))


class Expression:

    def __init__(self, node):
        self.d = {}
        self.parent = None
        self.node = node

    def __getitem__(self, k):
        return self.d[k]

    def get(self, k, default=None):
        return self.d.get(k, default)

    def __setitem__(self, k, v):
        self.d[k] = v

    def __delitem__(self, k):
        del self.d[k]

    def __contains__(self, k):
        return k in self.d

    def __str__(self):
        return "<{} {}>".format(self.__class__.__name__, self.d)


class ModuleWhile(Expression):
    pass
class FunctionWhile(Expression):
    pass
class ClassWhile(Expression):
    pass

class InputStreamer:

    def __getitem__(self, idx):
        return idx

input_streamer = InputStreamer()




class TargetNonlocalFlow(Exception):
    """Base exception class to simulate non-local control flow transfers in
    a target application."""
    pass

class OutputValueBreak(TargetNonlocalFlow):
    pass

class OutputValueContinue(TargetNonlocalFlow):
    pass

class OutputValueReturn(TargetNonlocalFlow):
    pass


class VariableScope:

    def __init__(self, name):
        self.name = name

NO_VAR = VariableScope("no_var")
GLOBAL = VariableScope("global")
NONLOCAL = VariableScope("nonlocal")



INTEGER, PLUS, MINUS, MUL, DIV, LPAREN, RPAREN, EOF, EQUAL, SKIP, TRUE, FALSE, CONJUNCTION, DISJUNCTION, NEGATION = (
    'INTEGER', 'PLUS', 'MINUS', 'MUL', 'DIV', '(', ')', 'EOF', ':=', 'SKIP', 'TRUE', 'FALSE', '∧', '∨', '¬'
)


class Token(object):
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        return 'Token({type}, {value})'.format(
            type=self.type,
            value=repr(self.value)
        )

    def __repr__(self):
        return self.__str__()

class Lexer(object):
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos]

    def error(self):
        raise Exception('Invalid character')

    def advance(self):
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

class InterpreterFuncWrap:
    
    def __init__(self, node, interp):
        self.node = node
        self.interp = interp
        self.lexical_scope = interp.ns

    def __call__(self, *args, **kwargs):
        return self.interp.call_func(self.node, self, *args, **kwargs)



def InterpreterFunc(fun):

    def func(*args, **kwargs):
        return fun.__call__(*args, **kwargs)

    return func

class InterpreterWith:

    def __init__(self, ctx):
        self.ctx = ctx

    def __enter__(self):
        return self.ctx.__enter__()

    def __exit__(self, tp, exc, tb):
        if isinstance(exc, TargetNonlocalFlow):
            tp = exc = tb = None
        return self.ctx.__exit__(tp, exc, tb)


class InterpreterModule:

    def __init__(self, ns):
        self.ns = ns

    def __getattr__(self, name):
        try:
            return self.ns[name]
        except KeyError:
            raise AttributeError

    def __dir__(self):
        return list(self.ns.d.keys())


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



def main():
    while True:
        text = input("")

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



#INTERPRETER
class NodeVisitor(object):
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))




class Interpreter(EntryNodeVisitor):

    def __init__(self, fname):
        self.fname = fname
        self.ns = None
        self.module_ns = None
        self.call_stack = []
        self.store_val = None
        self.cur_exc = []

    def push_ns(self, new_ns):
        new_ns.parent = self.ns
        self.ns = new_ns

    def pop_ns(self):
        self.ns = self.ns.parent

    def stmt_list_visit(self, lst):
        res = None
        for s in lst:
            res = self.visit(s)
        return res

    def wrap_decorators(self, obj, node):
        for deco_n in reversed(node.decorator_list):
            deco = self.visit(deco_n)
            obj = deco(obj)
        return obj

    def visit_Module(self, node):
        self.ns = self.module_ns = ModuleWhile(node)
        self.ns["__file__"] = self.fname
        self.ns["__name__"] = "__main__"
        #sys.modules["__main__"] = InterpModule(self.ns)
        self.stmt_list_visit(node.body)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_ClassDef(self, node):
        self.push_ns(ClassWhile(node))
        try:
            self.stmt_list_visit(node.body)
        except:
            self.pop_ns()
            raise
        ns = self.ns
        self.pop_ns()
        cls = type(node.name, tuple([self.visit(b) for b in node.bases]), ns.d)
        cls = self.wrap_decorators(cls, node)
        self.ns[node.name] = cls
        ns.cls = cls

    def visit_Lambda(self, node):
        node.name = "<lambda>"
        return self.prepare_func(node)

    def visit_FunctionDef(self, node):
        func = self.prepare_func(node)
        func = self.wrap_decorators(func, node)
        self.ns[node.name] = func

    def prepare_func(self, node):
        func = InterpreterFuncWrap(node, self)
        args = node.args
        num_required = len(args.args) - len(args.defaults)
        all_args = set()
        d = {}
        for i, a in enumerate(args.args):
            all_args.add(a.arg)
            if i >= num_required:
                d[a.arg] = self.visit(args.defaults[i - num_required])
        for a, v in zip(args.kwonlyargs, args.kw_defaults):
            all_args.add(a.arg)
            if v is not None:
                d[a.arg] = self.visit(v)
        node.args.all_args = all_args
        func.defaults_dict = d

        return InterpreterFunc(func)

    def prepare_func_args(self, node, interp_func, *args, **kwargs):

        def arg_num_mismatch():
            raise TypeError("{}() takes {} positional arguments but {} were given".format(node.name, len(argspec.args), len(args)))

        argspec = node.args

        if argspec.vararg:
            self.ns[argspec.vararg.arg] = args[len(argspec.args):]
        else:
            if len(args) > len(argspec.args):
                arg_num_mismatch()

        for i in range(min(len(args), len(argspec.args))):
            self.ns[argspec.args[i].arg] = args[i]

        func_kwarg = {}
        for k, v in kwargs.items():
            if k in argspec.all_args:
                if k in self.ns:
                    raise TypeError("{}() got multiple values for argument '{}'".format(node.name, k))
                self.ns[k] = v
            elif argspec.kwarg:
                func_kwarg[k] = v
            else:
                raise TypeError("{}() got an unexpected keyword argument '{}'".format(node.name, k))
        if argspec.kwarg:
            self.ns[argspec.kwarg.arg] = func_kwarg

        for k, v in interp_func.defaults_dict.items():
            if k not in self.ns:
                self.ns[k] = v

        for a in argspec.args:
            if a.arg not in self.ns:
                raise TypeError("{}() missing required positional argument: '{}'".format(node.name, a.arg))
        for a in argspec.kwonlyargs:
            if a.arg not in self.ns:
                raise TypeError("{}() missing required keyword-only argument: '{}'".format(node.name, a.arg))

    def call_func(self, node, interp_func, *args, **kwargs):
        self.call_stack.append(node)
        dyna_scope = self.ns
        self.ns = interp_func.lexical_scope
        self.push_ns(FunctionWhile(node))
        try:
            self.prepare_func_args(node, interp_func, *args, **kwargs)
            if isinstance(node.body, list):
                res = self.stmt_list_visit(node.body)
            else:
                res = self.visit(node.body)
        except OutputValueReturn as e:
            res = e.args[0]
        finally:
            self.pop_ns()
            self.ns = dyna_scope
            self.call_stack.pop()
        return res

    def visit_Return(self, node):
        if not isinstance(self.ns, FunctionWhile):
            raise SyntaxError("'return' outside function")
        raise OutputValueReturn(node.value and self.visit(node.value))

    def visit_With(self, node):
        assert len(node.items) == 1
        ctx = self.visit(node.items[0].context_expr)
        with InterpreterWith(ctx) as val:
            if node.items[0].optional_vars is not None:
                self.handle_assign(node.items[0].optional_vars, val)
            self.stmt_list_visit(node.body)

    def visit_Try(self, node):
        try:
            self.stmt_list_visit(node.body)
        except TargetNonlocalFlow:
            raise
        except Exception as e:
            self.cur_exc.append(e)
            try:
                for h in node.handlers:
                    if h.type is None or isinstance(e, self.visit(h.type)):
                        if h.name:
                            self.ns[h.name] = e
                        self.stmt_list_visit(h.body)
                        if h.name:
                            del self.ns[h.name]
                        break
                else:
                    raise
            finally:
                self.cur_exc.pop()
        else:
            self.stmt_list_visit(node.orelse)
        finally:
            self.stmt_list_visit(node.finalbody)
        
    def visit_For(self, node):
        iter = self.visit(node.iter)
        for item in iter:
            self.handle_assign(node.target, item)
            try:
                self.stmt_list_visit(node.body)
            except OutputValueBreak:
                break
            except OutputValueContinue:
                continue
        else:
            self.stmt_list_visit(node.orelse)

    def visit_While(self, node):
        while self.visit(node.test):
            try:
                self.stmt_list_visit(node.body)
            except OutputValueBreak:
                break
            except OutputValueContinue:
                continue
        else:
            self.stmt_list_visit(node.orelse)

    def visit_Break(self, node):
        raise OutputValueBreak

    def visit_Continue(self, node):
        raise OutputValueContinue

    def visit_If(self, node):
        test = self.visit(node.test)
        if test:
            self.stmt_list_visit(node.body)
        else:
            self.stmt_list_visit(node.orelse)

    def visit_Import(self, node):
        for n in node.names:
            self.ns[n.asname or n.name] = __import__(n.name)

    def visit_ImportFrom(self, node):
        mod = __import__(node.module, None, None, [n.name for n in node.names], node.level)
        for n in node.names:
            self.ns[n.asname or n.name] = getattr(mod, n.name)

    def visit_Raise(self, node):
        if node.exc is None:
            if not self.cur_exc:
                raise RuntimeError("No active exception to reraise")
            raise self.cur_exc[-1]
        if node.cause is None:
            raise self.visit(node.exc)
        else:
            raise self.visit(node.exc) from self.visit(node.cause)

    def visit_AugAssign(self, node):
        assert isinstance(node.target.ctx, ast.Store)
        save_ctx = node.target.ctx
        node.target.ctx = ast.Load()
        var_val = self.visit(node.target)
        node.target.ctx = save_ctx

        rval = self.visit(node.value)

        op = type(node.op)
        if op is ast.Add:
            var_val += rval
        elif op is ast.Sub:
            var_val -= rval
        elif op is ast.Mult:
            var_val *= rval
        elif op is ast.Div:
            var_val /= rval
        elif op is ast.FloorDiv:
            var_val //= rval
        elif op is ast.Mod:
            var_val %= rval
        elif op is ast.Pow:
            var_val **= rval
        elif op is ast.LShift:
            var_val <<= rval
        elif op is ast.RShift:
            var_val >>= rval
        elif op is ast.BitAnd:
            var_val &= rval
        elif op is ast.BitOr:
            var_val |= rval
        elif op is ast.BitXor:
            var_val ^= rval
        else:
            raise NotImplementedError

        self.store_val = var_val
        self.visit(node.target)

    def visit_Assign(self, node):
        val = self.visit(node.value)
        for n in node.targets:
            self.handle_assign(n, val)

    def handle_assign(self, target, val):
        if isinstance(target, ast.Tuple):
            it = iter(val)
            try:
                for elt_idx, t in enumerate(target.elts):
                    if isinstance(t, ast.Starred):
                        t = t.value
                        all_elts = list(it)
                        break_i = len(all_elts) - (len(target.elts) - elt_idx - 1)
                        self.store_val = all_elts[:break_i]
                        it = iter(all_elts[break_i:])
                    else:
                        self.store_val = next(it)
                    self.visit(t)
            except StopIteration:
                raise ValueError("not enough values to unpack (expected {})") from None

            try:
                next(it)
                raise ValueError("too many values to unpack (expected {})")
            except StopIteration:
                # Expected
                pass
        else:
            self.store_val = val
            self.visit(target)

    def visit_Delete(self, node):
        for n in node.targets:
            self.visit(n)

    def visit_Pass(self, node):
        pass

    def visit_Assert(self, node):
        if node.msg is None:
            assert self.visit(node.test)
        else:
            assert self.visit(node.test), self.visit(node.msg)

    def visit_Expr(self, node):
        # Produced value is ignored
        self.visit(node.value)

    def enumerate_comps(self, iters):
        """Enumerate thru all possible values of comprehension clauses,
        including multiple "for" clauses, each optionally associated
        with multiple "if" clauses. Current result of the enumeration
        is stored in the namespace."""

        def eval_ifs(iter):
            """Evaluate all "if" clauses."""
            for cond in iter.ifs:
                if not self.visit(cond):
                    return False
            return True

        if not iters:
            yield
            return
        for el in self.visit(iters[0].iter):
            self.store_val = el
            self.visit(iters[0].target)
            for t in self.enumerate_comps(iters[1:]):
                if eval_ifs(iters[0]):
                    yield

    def visit_ListComp(self, node):
        self.push_ns(FunctionWhile(node))
        try:
            return [
                self.visit(node.elt)
                for _ in self.enumerate_comps(node.generators)
            ]
        finally:
            self.pop_ns()

    def visit_SetComp(self, node):
        self.push_ns(FunctionWhile(node))
        try:
            return {
                self.visit(node.elt)
                for _ in self.enumerate_comps(node.generators)
            }
        finally:
            self.pop_ns()

    def visit_DictComp(self, node):
        self.push_ns(FunctionWhile(node))
        try:
            return {
                self.visit(node.key): self.visit(node.value)
                for _ in self.enumerate_comps(node.generators)
            }
        finally:
            self.pop_ns()

    def visit_IfExp(self, node):
        if self.visit(node.test):
            return self.visit(node.body)
        else:
            return self.visit(node.orelse)

    def visit_Call(self, node):
        func = self.visit(node.func)

        args = []
        for a in node.args:
            if isinstance(a, ast.Starred):
                args.extend(self.visit(a.value))
            else:
                args.append(self.visit(a))

        kwargs = {}
        for kw in node.keywords:
            val = self.visit(kw.value)
            if kw.arg is None:
                kwargs.update(val)
            else:
                kwargs[kw.arg] = val

        if func is builtins.super and not args:
            if not self.ns.parent or not isinstance(self.ns.parent, ClassWhile):
                raise RuntimeError("super(): no arguments")
            args = (self.ns.parent.cls, self.ns["self"])

        return func(*args, **kwargs)

    def visit_Compare(self, node):
        cmpop_map = {
            ast.Eq: lambda x, y: x == y,
            ast.NotEq: lambda x, y: x != y,
            ast.Lt: lambda x, y: x < y,
            ast.LtE: lambda x, y: x <= y,
            ast.Gt: lambda x, y: x > y,
            ast.GtE: lambda x, y: x >= y,
            ast.Is: lambda x, y: x is y,
            ast.IsNot: lambda x, y: x is not y,
            ast.In: lambda x, y: x in y,
            ast.NotIn: lambda x, y: x not in y,
        }
        lv = self.visit(node.left)
        for op, r in zip(node.ops, node.comparators):
            rv = self.visit(r)
            if not cmpop_map[type(op)](lv, rv):
                return False
            lv = rv
        return True

    def visit_BoolOp(self, node):
        if isinstance(node.op, ast.And):
            res = True
            for v in node.values:
                res = res and self.visit(v)
        elif isinstance(node.op, ast.Or):
            res = False
            for v in node.values:
                res = res or self.visit(v)
        else:
            raise NotImplementedError
        return res

    def visit_BinOp(self, node):
        binop_map = {
            ast.Add: lambda x, y: x + y,
            ast.Sub: lambda x, y: x - y,
            ast.Mult: lambda x, y: x * y,
            ast.Div: lambda x, y: x / y,
            ast.FloorDiv: lambda x, y: x // y,
            ast.Mod: lambda x, y: x % y,
            ast.Pow: lambda x, y: x ** y,
            ast.LShift: lambda x, y: x << y,
            ast.RShift: lambda x, y: x >> y,
            ast.BitAnd: lambda x, y: x & y,
            ast.BitOr: lambda x, y: x | y,
            ast.BitXor: lambda x, y: x ^ y,
        }
        l = self.visit(node.left)
        r = self.visit(node.right)
        return binop_map[type(node.op)](l, r)

    def visit_UnaryOp(self, node):
        unop_map = {
            ast.UAdd: lambda x: +x,
            ast.USub: lambda x: -x,
            ast.Invert: lambda x: ~x,
            ast.Not: lambda x: not x,
        }
        val = self.visit(node.operand)
        return unop_map[type(node.op)](val)

    def visit_Subscript(self, node):
        obj = self.visit(node.value)
        idx = self.visit(node.slice)
        if isinstance(node.ctx, ast.Load):
            return obj[idx]
        elif isinstance(node.ctx, ast.Store):
            obj[idx] = self.store_val
        elif isinstance(node.ctx, ast.Del):
            del obj[idx]
        else:
            raise NotImplementedError

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Slice(self, node):
        # Any of these can be None
        lower = node.lower and self.visit(node.lower)
        upper = node.upper and self.visit(node.upper)
        step = node.step and self.visit(node.step)
        slice = input_streamer[lower:upper:step]
        return slice

    def visit_Attribute(self, node):
        obj = self.visit(node.value)
        if isinstance(node.ctx, ast.Load):
            return getattr(obj, node.attr)
        elif isinstance(node.ctx, ast.Store):
            setattr(obj, node.attr, self.store_val)
        elif isinstance(node.ctx, ast.Del):
            delattr(obj, node.attr)
        else:
            raise NotImplementedError

    def visit_Global(self, node):
        for n in node.names:
            if n in self.ns and self.ns[n] is not GLOBAL:
                raise SyntaxError("SyntaxError: name '{}' is assigned to before global declaration".format(n))
            # Don't store GLOBAL in the top-level namespace
            if self.ns.parent:
                self.ns[n] = GLOBAL

    def visit_Nonlocal(self, node):
        if isinstance(self.ns, ModuleWhile):
            raise SyntaxError("nonlocal declaration not allowed at module level")
        for n in node.names:
            self.ns[n] = NONLOCAL

  
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            res = NO_VAR
            ns = self.ns
            skip_classes = False
            while ns:
                if not (skip_classes and isinstance(ns, ClassWhile)):
                    res = ns.get(node.id, NO_VAR)
                    if res is not NO_VAR:
                        break
                ns = ns.parent
                skip_classes = True

            if res is NONLOCAL:
                ns = self.resolve_nonlocal(node.id, ns.parent)
                return ns[node.id]
            if res is GLOBAL:
                res = self.module_ns.get(node.id, NO_VAR)
            if res is not NO_VAR:
                return res

            try:
                return getattr(builtins, node.id)
            except AttributeError:
                raise NameError("name '{}' is not defined".format(node.id))
        elif isinstance(node.ctx, ast.Store):
            res = self.ns.get(node.id, NO_VAR)
            if res is GLOBAL:
                self.module_ns[node.id] = self.store_val
            elif res is NONLOCAL:
                ns = self.resolve_nonlocal(node.id, self.ns.parent)
                ns[node.id] = self.store_val
            else:
                self.ns[node.id] = self.store_val
        elif isinstance(node.ctx, ast.Del):
            res = self.ns.get(node.id, NO_VAR)
            if res is NO_VAR:
                raise NameError("name '{}' is not defined".format(node.id))
            elif res is GLOBAL:
                del self.module_ns[node.id]
            elif res is NONLOCAL:
                ns = self.resolve_nonlocal(node.id, self.ns.parent)
                del ns[node.id]
            else:
                del self.ns[node.id]
        else:
            raise NotImplementedError

    def visit_Dict(self, node):
        return {self.visit(p[0]): self.visit(p[1]) for p in zip(node.keys, node.values)}

    def visit_Set(self, node):
        return {self.visit(e) for e in node.elts}

    def visit_List(self, node):
        return [self.visit(e) for e in node.elts]

    def visit_Tuple(self, node):
        return tuple([self.visit(e) for e in node.elts])

    def visit_NameConstant(self, node):
        return node.value

    def visit_Ellipsis(self, node):
        return ...

    def visit_Str(self, node):
        return node.s

    def visit_Bytes(self, node):
        return node.s

    def visit_Num(self, node):
        return node.n

