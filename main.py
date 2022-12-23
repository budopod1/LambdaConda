alphabet = "qwertyuiopasdfghjklzxcvbnm"
letters = alphabet.lower() + alphabet.upper()
numbers = "1234567890"


class EnumValue:
    def __init__(self,enum, value):
        self.enum = enum
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, EnumValue):
            raise TypeError(f"Cannot compare EnumValue with {type(other)}")
        return self.enum == other.enum and self.value == other.value

    def __ne__(self, other):
        return not (other == self)

    def __repr__(self):
        return f"{self.enum.name}.{self.enum.values[self.value]}"

    def __str__(self):
        return repr(self)


class Enum:
    def __init__(self, name, *values):
        self.name = name
        self.values = {}
        self.contains = values
        for i, value in enumerate(values):
            self.__dict__[value] = i
            self.values[i] = value

    def __iter__(self):
        return (value for value in self.contains)

    def __getattribute__(self, key):
        __dict__ = object.__getattribute__(self, "__dict__")
        value = object.__getattribute__(self, key)
        if "contains" in __dict__:
            contains = object.__getattribute__(self, "contains")
            if key in contains:
                return EnumValue(self, value)
        return value


TokenType = Enum("TokenType", "TEXT", "TREE")
TokenSearchStatus = Enum("TokenSearchStatus", "FAIL", "CONTINUE", "FINISH",
                         "TERMINATE")
TokenVisitMode = Enum("TokenVisitMode", "DEPTH")
TokenExtraData = Enum("TokenExtraData", "CAPPED")
BasicType = Enum("BasicType", "int", "float", "bool", "str", "func", "none",
                 "tuple")
TAB = "    "


def python_to_type(obj):
    types = {
        int: BasicType.int,
        float: BasicType.float,
        bool: BasicType.bool,
        str: BasicType.str,
        tuple: BasicType.tuple,
        type(None): BasicType.none
    }
    for py_type, basic_type in types.items():
        if isinstance(obj, py_type):
            return basic_type
    return None


class Token:
    SEPERATOR = ", "
    BORDER_SEP = False
    INDENT = ""

    def __init__(self, value, tags=None):
        if tags is None:
            tags = []
        self.tags = tags
        if isinstance(value, str):
            token_type = TokenType.TEXT
        else:
            token_type = TokenType.TREE
            value = list(value)
            if value:
                tree_children = isinstance(value[0], Token)
                assert tree_children, "Tree token must have token children"
        self.value = value
        self.token_type = token_type
        self.parent = None
        self.assertions()
        for key in dir(self):
            value = getattr(self, key)
            if "start" in key and callable(value):
                value()

    def ensure_parents(self):
        if self.is_text():
            return
        for child in self.value:
            child.parent = self
            child.ensure_parents()

    def search_parent(self, type_):
        if self.parent is None:
            return None
        else:
            if self.parent.is_a(type_):
                return self.parent
            return self.parent.search_parent(type_)

    def assertions(self):
        assert type(self) != Token, "Token class is abstract"

    def is_text(self):
        return self.token_type == TokenType.TEXT

    def is_tree(self):
        return self.token_type == TokenType.TREE

    def add(self, token):
        assert self.is_tree()
        self.value.append(token)

    def is_a(self, token_type):
        return isinstance(self, token_type)

    def clone(self, value=None):
        return type(self)(value if value else self.value)

    def set_range(self, start, end, tokens, inplace=False):
        result = self.value[:start] + tokens + self.value[end + 1:]
        if inplace:
            self.value = result
        return self.clone(result)

    def replace(self, find, replace, inplace=False):
        assert self.is_tree()
        token_string = []
        start = -1
        for i, token in enumerate(self.value):
            if find[len(token_string)].matches(token):
                if len(token_string) == 0:
                    start = i
                token_string.append(token)
            else:
                token_string = []
                start = -1
            if len(find) == len(token_string):
                return self.set_range(start, i, [replace(token_string)],
                                      inplace)
        return None

    def replace_all(self, find, replace, inplace=False):
        assert inplace, "Non-inplace replace_all is not suported yet"
        result = True
        while result:
            result = self.replace(find, replace, True)

    def stringify(self, **options):
        if self.is_text():
            result = self.value
        else:
            result = ""
            for token in self.value:
                text = token.stringify(**options)
                already = result.endswith(self.SEPERATOR)
                if token.is_tree() and not already and result:
                    result += self.SEPERATOR
                result += text
                if token.is_tree():
                    result += self.SEPERATOR
            if result.endswith(self.SEPERATOR):
                result = result[:-2]

            if self.BORDER_SEP:
                result = self.SEPERATOR + result
            result = result.replace("\n", "\n" + self.INDENT)
            if self.BORDER_SEP:
                result += self.SEPERATOR
            extra_data = ", ".join(
                [f"{k}: {v}" for k, v in self.string_data().items()])
            result = f"{type(self).__name__}({result}{extra_data})"
        return result

    def string_data(self):
        return {}

    def __str__(self):
        return self.stringify()

    def __repr__(self):
        return repr(str(self))

    def visit(self, visitor, arguments, allowed=None, mode=None):
        if mode is None:  # unused currently
            mode = TokenVisitMode.DEPTH
        if allowed is None:
            allowed = lambda t: True
        cancel = visitor(self, arguments)
        if cancel:
            return cancel
        if self.is_tree():
            for child in self.value:
                if not allowed(child):
                    continue
                cancel = child.visit(visitor, arguments, allowed, mode)
                if cancel:
                    return cancel
        return None


class Instance:
    def __init__(self):
        self.type_ = Type.none

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"Instance<{self.type_}>"


class Type:
    def __init__(self, type_, generics=None):
        self.type_ = type_
        if generics is None:
            generics = []
        self.generics = generics

    def __eq__(self, other):
        if other.type_ != self.type_:
            return False
        if len(self.generics) != len(other.generics):
            return False
        for a, b in zip(self.generics, other.generics):
            if a != b:
                return False
        return True

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        generics = [str(generic) for generic in self.generics]
        return f"{self.type_}<{', '.join(generics)}>"


Type.none = Type(BasicType.none)


class TokenMatch:
    def __init__(self, token_type, value):
        self.token_type = token_type
        self.value = value

    def matches(self, token):
        if self.token_type != token.token_type:
            return False
        else:
            if token.is_tree():
                return token.is_a(self.value)
            else:
                return token.value == self.value

    def __str__(self):
        return f"<TokenMatch {self.token_type} {self.value!r}>"


class Node(Token):  # Base class for any structure in the final IR
    def assertions(self):  # Can't do super() for some reason
        assert self.is_tree(), "Node Token does not store text"


class Text(Token):  # Just text, stores snippets of code from source
    def assertions(self):
        assert self.is_text(), "Text Token stores text"

    def stringify(self):
        return self.value


class Name(Node):
    pass


class Symbolic(Token):  # Represents something temporary, but /w structure
    def assertions(self):
        assert self.is_tree(), "Symbolic Token is a tree"
        text_children = all([token.is_text() for token in self.value])
        assert text_children, "Symbolic Token cannot have tree children"

    def stringify(self):
        return f"<{type(self).__name__}>"


class Operand(Token):
    def startOPERAND(self):
        self.instance = Instance()

    def type_(self, value=None):
        if value is not None:
            if not isinstance(value, Type):
                raise TypeError("Type must be Type")
            self.instance.type_ = value
        return self.instance.type_

    def compute_type(self):
        type_ = self._compute_type()
        assert isinstance(type_, Type), "Type must be Type"
        if type_.type_ != BasicType.none:
            self.type_(type_)
        return type_

    def _compute_type(self):
        return Type.none


class Scope:
    def __init__(self, token):
        self.token = token
        self.scope = {}
        self.instances = {}

    def get(self, name):
        if name in self.scope:
            return self.scope[name]
        else:
            scope = self.token.search_parent(Block)
            if scope is None:
                return None
            else:
                return scope.scope.get(name)
        return None

    def assign(self, name, value):
        changed = name not in self.scope
        self.scope[name] = value
        return changed


class Block(Node):

    def startBLOCK(self):
        self.scope = Scope(self)

    def string_data(self):
        return self.scope.scope

    SEPERATOR = "\n"
    BORDER_SEP = True
    INDENT = TAB


class TypeToken(Operand):
    pass


class TokenConversion:  # make into data class

    def __init__(self, find, replace):
        self.find = find
        self.replace = replace


class Constants:
    def __init__(self):
        self.constants = {}
        self._id = -1

    def add(self, value):
        self._id += 1
        name = f"_CONST{self._id}"
        self.constants[name] = value
        return name

    def __iter__(self):
        return (v for v in self.constants.items())


class TextRule:
    def __init__(self, expected, replace_with):
        self.expected = expected
        self.replace_with = replace_with

    def __call__(self, token):
        text = token.value if token.is_text() else None

        if text == self.expected[0]:
            self.expected = self.expected[1:]
            if not self.expected:
                return TokenSearchStatus.FINISH
            return TokenSearchStatus.CONTINUE
        else:
            return TokenSearchStatus.FAIL

    def result(self, tokens):
        return tokenify(self.replace_with)


class SetRule:
    # Replace a set of pieces of text with a token containing
    # that text
    def __init__(self, options, holder):
        self.options = options
        self.holder = holder
        self.i = 0

    def __call__(self, token):
        text = token.value if token.is_text() else None

        options = list(filter(
            lambda option: text == option[self.i], 
            self.options
        ))
        
        self.i += 1

        for option in options:
            if self.i == len(option):
                return TokenSearchStatus.FINISH
        
        if len(options) == 0:
            return TokenSearchStatus.FAIL
            
        return TokenSearchStatus.CONTINUE
        
    def result(self, tokens):
        return [self.holder(tokens, [self.factory])]


class GroupRule:
    def __init__(self, types, keep, holder):
        self.types = types
        self.keep = keep
        self.holder = holder

    def __call__(self, token):
        expected = self.types.pop(0)
        if token.is_a(expected):
            if not self.types:
                return TokenSearchStatus.FINISH
            return TokenSearchStatus.CONTINUE
        else:
            return TokenSearchStatus.FAIL

    def result(self, tokens):
        return [
            self.holder(
                [token for i, token in enumerate(tokens) if i in self.keep],
                [self.factory]
            )
        ]


class CollapseRule:
    def __init__(self, parent, children):
        self.parent = parent
        self.children = children

    def __call__(self, token):
        if token.is_a(self.parent):
            for expected, child in zip(self.children, token.value):
                if not child.is_a(expected):
                    return TokenSearchStatus.FAIL
            return TokenSearchStatus.FINISH
        return TokenSearchStatus.FAIL

    def result(self, tokens):
        return tokens[0].value


class MergeRule:
    def __init__(self, types, center, keep, convert):
        self.types = types
        self.center = center
        self.keep = keep
        self.convert = convert

    def __call__(self, token):
        expected = self.types.pop(0)
        if token.is_a(expected):
            if not self.types:
                return TokenSearchStatus.FINISH
            return TokenSearchStatus.CONTINUE
        else:
            return TokenSearchStatus.FAIL

    def result(self, tokens):
        result = tokens[self.center].value
        before = True
        for i, token in enumerate(tokens):
            if i == self.center:
                before = False
            elif i in self.keep:
                if before:
                    result.insert(0, token)
                else:
                    result.append(token)
        return [self.convert(result)]


class ConvertRule:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __call__(self, token):
        if token.is_a(self.start):
            return TokenSearchStatus.FINISH
        else:
            return TokenSearchStatus.FAIL

    def result(self, tokens):
        token, = tokens
        return [self.end(token.value)]


class RemoveRule:
    def __init__(self, types):
        self.types = types

    def __call__(self, token):
        expected = self.types.pop(0)
        if token.is_a(expected):
            if not self.types:
                return TokenSearchStatus.FINISH
            return TokenSearchStatus.CONTINUE
        else:
            return TokenSearchStatus.FAIL

    def result(self, tokens):
        return []


class BoundaryRule:
    def __init__(self,
                 start,
                 end,
                 container,
                 blockers=None,
                 allow_text=False,
                 keep_boundaries=False):
        if blockers is None:
            blockers = []
        self.start = start
        self.end = end
        self.blockers = blockers
        self.blockers.append(start)
        self.container = container
        self.keep_boundaries = keep_boundaries
        self.allow_text = allow_text
        self.started = False

    def __call__(self, token):
        if self.started:
            if token.is_text():
                if self.allow_text:
                    return TokenSearchStatus.CONTINUE
                else:
                    return TokenSearchStatus.FAIL
            if token.is_a(self.end):
                return TokenSearchStatus.FINISH
            for blocker in self.blockers:
                if token.is_a(blocker):
                    return TokenSearchStatus.FAIL
            return TokenSearchStatus.CONTINUE
        else:
            if token.is_text():
                return TokenSearchStatus.FAIL
            if token.is_a(self.start):
                self.started = True
                return TokenSearchStatus.CONTINUE
            else:
                return TokenSearchStatus.FAIL

    def result(self, tokens):
        if not self.keep_boundaries:
            tokens = tokens[1:-1]
        return [self.container(tokens)]


def tokenify(string):
    tokens = []
    for char in string:
        tokens.append(Text(char))
    return tokens


def replace_token_search_match(original, match, new):
    original.set_range(match.start, match.end, new, True)


class TokenSearchMatch:  # Make into data class

    def __init__(self, start, end, tokens, searcher):
        self.start = start
        self.end = end
        self.tokens = tokens
        self.searcher = searcher

    def __repr__(self):
        return f"<TokenSearchMatch {self.start}-{self.end}:{self.tokens}>"

    def __str__(self):
        return repr(self)


# Conventions:

# .*Search:
# * Takes no arguments
# * Can be a function or a class
# * Does not need to be put in a lambda

# .*Rule:
# * Is a subset of .*Search
# * Must be a class
# * Must have a result() function with the signature:
#     Token[] -> Token[]
# * Needs to be used in a lambda


def token_search(tokens, searcher_factory):
    last = -1
    for i, _ in enumerate(tokens.value):
        if i <= last:
            continue
        token_string = []
        searcher = searcher_factory()
        searcher.factory = searcher_factory
        for j, token in enumerate(tokens.value[i:]):
            status = searcher(token)
            if status == TokenSearchStatus.FAIL:
                break
            elif status == TokenSearchStatus.CONTINUE:
                token_string.append(token)
            elif (status == TokenSearchStatus.FINISH
                  or status == TokenSearchStatus.TERMINATE):
                token_string.append(token)
                end = i + j
                if status == TokenSearchStatus.TERMINATE:
                    token_string.pop()
                    end -= 1
                last = end
                yield TokenSearchMatch(i, end, token_string, searcher)
                break


def perform_conversions(tokens, conversions):
    for conversion in conversions:
        tokens.replace_all(conversion.find, conversion.replace, True)


def create_text_conversion(find, replace):
    return TokenConversion(
        [TokenMatch(TokenType.TEXT, char) for char in find],
        replace
    )


def condense_tokens(tokens):
    return "".join([str(token) for token in tokens])


def tagged(token, tags):
    def inner(*args):
        return token(*args, tags=tags)
    return inner
    

# CUSTOM CODE START


class Program(Block):
    pass


class Whitespace(Symbolic):
    pass


class Space(Whitespace):
    pass


class Newline(Whitespace):
    pass


class FunctionOpen(Symbolic):
    pass


class FunctionClose(Symbolic):
    pass


class GroupOpen(Symbolic):
    pass


class GroupClose(Symbolic):
    pass


class Comma(Symbolic):
    pass


class Operator(Symbolic):
    pass


class Typer(Symbolic):
    # A colon, used the same way as weak typing
    # in python, however this typing is strong
    pass


class AssignOperator(Operator):
    pass


class AdditionOperator(Operator):
    pass


class MinusOperator(Operator):  # Could be subtraction or negation
    pass


class ReturnOperator(Operator):
    pass


class Tuple(Operand):
    def _compute_type(self):
        child_types = [child.type_() for child in self.value]
        if any([type_.type_ == BasicType.none for type_ in child_types]):
            return Type.none
        return Type(BasicType.tuple, child_types)


class UnfinishedTuple(Tuple):
    pass


class FunctionCall(Operand):
    def _compute_type(self):
        func, _ = self.value
        func_type = func.type_()
        if not func_type.generics:
            return Type.none
        return func.type_().generics[0]


class Assignment(Operand):
    def _compute_type(self):
        _, child = self.value
        return child.type_()


class Addition(Operand):
    def _compute_type(self):
        child, _ = self.value
        return child.type_()


class Subtraction(Operand):
    def _compute_type(self):
        child, _ = self.value
        return child.type_()


class Negation(Operand):
    def _compute_type(self):
        child, = self.value
        return child.type_()


class Group(Operand):
    def _compute_type(self):
        child, = self.value
        return child.type_()


class Return(Operand):
    def _compute_type(self):
        child, = self.value
        return child.type_()


class Refrence(Operand):  # Anything that can be refrenced in code
    pass


class Variable(Refrence):
    pass


class Constant(Refrence):
    pass


class Function(Operand, Block):
    def _compute_type(self):
        def return_visit(token, arguments):
            if token.is_a(Return):
                return_type = token.type_()
                if return_type:
                    return return_type
                else:
                    return Type.none
        return_type = self.visit(
            return_visit, 
            tuple(), 
            lambda t: not t.is_a(Block)
        )
        if return_type:
            if return_type.type_ == BasicType.none:
                return Type.none
            return Type(BasicType.func, [return_type])
        return Type(BasicType.func, [Type.none])


class ArgumentFunction(Function):
    pass


class Argument(Node):
    pass


class Arguments(Node):
    pass


class UnfinishedArguments(Arguments):
    pass


class StringSearch:
    def __init__(self):
        self.expected = '"'

    def __call__(self, token):
        text = token.value if token.is_text() else None
        if self.expected:
            if text is not None and text in self.expected:
                self.expected = ''
                return TokenSearchStatus.CONTINUE
            else:
                return TokenSearchStatus.FAIL
        else:
            if text == '"':
                return TokenSearchStatus.FINISH
            else:
                if text == '\\':
                    self.expected = r'"\rnt'
                return TokenSearchStatus.CONTINUE


class NumberSearch:
    def __init__(self):
        self.dot_so_far = False
        self.anything = False

    def __call__(self, token):
        text = token.value if token.is_text() else None
        if text == ".":
            if not self.anything:  # .1 not allowed
                return TokenSearchStatus.FAIL
            if self.dot_so_far:
                return TokenSearchStatus.TERMINATE
            else:
                self.dot_so_far = True
                return TokenSearchStatus.CONTINUE
        else:
            if text is not None and text in numbers:
                self.anything = True
                return TokenSearchStatus.CONTINUE
            else:
                if self.anything:
                    return TokenSearchStatus.TERMINATE
                else:
                    return TokenSearchStatus.FAIL


class NameSearch:
    def __init__(self):
        self.anything = False

    def __call__(self, token):
        text = token.value if token.is_text() else None

        if self.anything:
            if text is not None and text in letters + "_" + numbers:
                return TokenSearchStatus.CONTINUE
            else:
                return TokenSearchStatus.TERMINATE
        else:
            if text is not None and text in letters + "_":
                self.anything = True
                return TokenSearchStatus.CONTINUE
            else:
                return TokenSearchStatus.FAIL


def main(code):
    print("Starting compile...")

    code = code.strip()
    constants = Constants()

    code = Program(tokenify(code))

    print("Starting string search...")

    while True:
        string = next(token_search(code, StringSearch), None)
        if string is None:
            break
        name = constants.add(condense_tokens(string.tokens))
        replace_token_search_match(code, string, [Constant(tokenify(name))])

    print("Started tokenizing...")

    perform_conversions(code, [
        create_text_conversion("->", AssignOperator),
        create_text_conversion("-", MinusOperator),
        create_text_conversion("+", AdditionOperator),
        create_text_conversion("return", ReturnOperator),
        create_text_conversion("{", FunctionOpen),
        create_text_conversion("}", FunctionClose),
        create_text_conversion("(", GroupOpen),
        create_text_conversion(")", GroupClose),
        create_text_conversion(" ", Space),
        create_text_conversion("\n", Newline),
        create_text_conversion(",", Comma),
        create_text_conversion(":", Typer),
    ])

    print("Starting number search...")

    while True:
        number = next(token_search(code, NumberSearch), None)
        if number is None:
            break
        name = constants.add(float(condense_tokens(number.tokens)))
        replace_token_search_match(code, number, [Constant(tokenify(name))])

    print("Starting name search...")

    while True:
        name = next(token_search(code, NameSearch), None)
        if name is None:
            break
        replace_token_search_match(code, name, [Name(name.tokens)])

    print("Starting ast formation...")

    transform_groups = [
        [lambda: RemoveRule([Whitespace])],
        [
            lambda: SetRule(list(BasicType), TypeToken),
            lambda: CollapseRule(Name, (TypeToken,))
        ],
        [
            lambda: BoundaryRule(
                FunctionOpen, FunctionClose, Function, allow_text=True)
        ],
        [
            lambda: GroupRule([GroupOpen, Operand, GroupClose], (1, ), Group),
            lambda: GroupRule([GroupOpen, GroupClose], tuple(), Tuple),
            lambda: GroupRule([Operand, MinusOperator, Operand],
                              (0, 2), Subtraction),
            lambda: GroupRule([MinusOperator, Operand], (1, ), Negation),
            lambda: GroupRule([Operand, MinusOperator, Operand],
                              (0, 2), Subtraction),
            lambda: GroupRule([Operand, AdditionOperator, Operand],
                              (0, 2), Addition),
            lambda: GroupRule([Variable, AssignOperator, Operand],
                              (0, 2), Assignment),
            lambda: GroupRule([ReturnOperator, Operand],
                            (1,), Return),
            lambda: GroupRule([Name, Typer, TypeToken], (0, 2), Argument),
            lambda: ConvertRule(Name, Variable),
            lambda: GroupRule([Arguments, Function], (0, 1),
                              ArgumentFunction),

            lambda: MergeRule(
                [UnfinishedArguments, Argument, Comma],
                0, (1,), UnfinishedArguments
            ),
            lambda: GroupRule(
                [GroupOpen, Argument, Comma, Argument, Comma],
                (1, 3), UnfinishedArguments
            ),
            lambda: GroupRule(
                [GroupOpen, Argument, Comma, Argument, GroupClose],
                (1, 3), Arguments
            ),
            lambda: GroupRule([GroupOpen, Argument, GroupClose],
                              (1,), Arguments),
            lambda: MergeRule(
                [UnfinishedArguments, Argument, GroupClose], 0,
                (1,), Arguments
            ),
            
            # Hardcode for tuples of length 2 or less
            lambda: MergeRule([UnfinishedTuple, Operand, Comma], 0,
                              (1, ), UnfinishedTuple),
            lambda: GroupRule([Operand, Comma, Operand, Comma],
                              (0, 2), UnfinishedTuple),
            lambda: GroupRule([Operand, Comma, Operand],
                              (0, 2), Tuple),
            lambda: GroupRule([Operand, Comma],
                              (0, ), Tuple),
            lambda: MergeRule([UnfinishedTuple, Operand], 0, (1, ), Tuple),
            
            lambda: GroupRule([Operand, Tuple], (0, 1), FunctionCall),
            lambda: GroupRule([Operand, Group], (0, 1), FunctionCall),
        ]
    ]

    def transform_visit(token, args):
        if token.is_text():
            return
        rule, = args
        for tag in token.tags:
            if rule == tag:
                return
        match = next(token_search(token, rule), None)
        if match is None:
            return
        replace_token_search_match(token, match,
                                   match.searcher.result(match.tokens))
        return True

    for transform_group in transform_groups:
        found_any = True
        while found_any:
            found_any = False
            for rule in transform_group:
                found_any = code.visit(transform_visit, (rule,))
                if found_any:
                    print(code)
                    break

    code.ensure_parents()

    print(code)

    print("Starting type inference...")

    for name, value in constants:
        instance = Instance()
        instance.type_ = Type(python_to_type(value))
        code.scope.assign(name, instance)

    def assign_visit(token, args):
        if isinstance(token, Assignment):
            name, value = token.value
            type_ = value.type_()
            if type_.type_ == BasicType.none:
                return
            scope = token.search_parent(Block).scope
            name = condense_tokens(name.value)
            if scope.get(name) is not None:
                return
            changed = scope.assign(name, token.instance)
            if changed:
                return True

    def reference_visit(token, args):
        if isinstance(token, Refrence):
            if token.type_().type_ != BasicType.none:
                return
            name = condense_tokens(token.value)
            scope = token.search_parent(Block).scope
            instance = scope.get(name)
            if instance is None:  # type_.type_ == BasicType.none
                return
            token.instance = instance
            return True

    def compute_visit(token, args):
        if isinstance(token, Operand):
            token.compute_type()

    found_any = True
    while found_any:
        found_any = False
        code.visit(compute_visit, tuple())
        found_any = code.visit(assign_visit, tuple())
        if found_any:
            continue
        found_any = code.visit(reference_visit, tuple())
        if found_any:
            continue
        break

    print(code)


def run(file):
    with open(file) as file:
        text = file.read()
    main(text)


if __name__ == "__main__":
    run("tuple.ll")


# Cool regex:
# /for .* in .*\.value/
# /class \w*:\n\s*\n/
