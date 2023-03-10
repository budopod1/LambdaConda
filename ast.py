from pyenum import Enum
from id_ import IDGetter
from instructions import make_instruction, FunctionInstruction, AssignInstruction, InstantiationInstruction, VariableInstruction, UnpackInstruction, FunctionCallInstruction, AdditionInstruction, TupleInstruction, ArrayInstruction, NegationInstruction, ReturnInstruction, ConcatenationInstruction, JoinTupleInstruction, JoinArrayInstruction, ConstantInstruction, SubtractionInstruction, DivisionInstruction, MultiplicationInstruction, ExponentiationInstruction, EqualsInstruction, NotEqualsInstruction, OrInstruction, AndInstruction, NotInstruction, LessInstruction, GreaterInstruction, LessEqualInstruction, GreaterEqualInstruction
from type_ import BasicType, Type
from builtins_ import BUILTINS


alphabet = "qwertyuiopasdfghjklzxcvbnm"
letters = alphabet.lower() + alphabet.upper()
numbers = "1234567890"


TokenType = Enum("TokenType", "TEXT", "TREE")
TokenSearchStatus = Enum("TokenSearchStatus", "FAIL", "CONTINUE", "FINISH",
                         "TERMINATE")
TokenVisitMode = Enum("TokenVisitMode", "DEPTH")
# TokenExtraData.UNIT specifies that the token should be treated as a
# single unit
TokenExtraData = Enum("TokenExtraData", "CAPPED", "UNIT")
RuleExtraData = Enum("RuleExtraData", "SEARCHALL")
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

    def copy(self):
        return type(self)(
            [
                child.copy()
                for child in self.value
            ] if self.is_tree() else self.value,
            self.tags.copy()
        )
    
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
                result = result[:-len(self.SEPERATOR)]

            if self.BORDER_SEP:
                result = self.SEPERATOR + result
            result = result.replace("\n", "\n" + self.INDENT)
            if self.BORDER_SEP:
                result += self.SEPERATOR
            extra_data = ", ".join(
                [f"{k}:{v}" for k, v in self.string_data().items()])
            if extra_data:
                extra_data = "|" + extra_data
            result = f"{type(self).__name__}({result}{extra_data})"
        return result

    def string_data(self):
        return {}
        # {"type": self.type_ if isinstance(self, Operand) else None}

    def __str__(self):
        return self.stringify()

    def __repr__(self):
        return repr(str(self)) if self.is_text() else str(self)

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
        self.type_ = Type.none

    def compute_type(self):
        type_ = self._compute_type()
        if type_ == self.type_:
            return False
        assert isinstance(type_, Type), "Type must be Type"
        if not type_.is_none():
            self.type_ = type_
            return True
        return False

    def _compute_type(self):
        return Type.none


class Scope:
    def __init__(self, token):
        self.token = token
        self.scope = {}

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

    def instruction(self):
        return make_instruction(
            FunctionInstruction,
            self.value,
            (self.type_ if isinstance(self, Operand) 
             else Type(BasicType.func, [Type.none]))
        )

    SEPERATOR = "\n"
    BORDER_SEP = True
    INDENT = TAB


class TokenConversion:  # make into data class
    def __init__(self, find, replace):
        self.find = find
        self.replace = replace


var_id = IDGetter()


class Constants:
    def __init__(self):
        self.constants = {}
        self._id = -1

    def add(self, value):
        self._id += 1
        name = f"$CONST{self._id}"
        self.constants[name] = value
        return name

    def contains(self, key):
        return key in self.constants

    def get(self, name):
        return self.constants[name]

    def __iter__(self):
        return (v for v in self.constants.items())

    def __str__(self):
        return f"Constants{self.constants}"


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
        self.holder = capped(holder)
        self.i = 0

    def __call__(self, token):
        text = token.value if token.is_text() else None

        self.options = list(filter(
            lambda option: text == option[self.i], 
            self.options
        ))
        
        self.i += 1

        for option in self.options:
            if self.i == len(option):
                return TokenSearchStatus.FINISH
        
        if len(self.options) == 0:
            return TokenSearchStatus.FAIL
            
        return TokenSearchStatus.CONTINUE
        
    def result(self, tokens):
        return [self.holder(tokens)]


class GroupRule:
    def __init__(self, types, keep, holder):
        self.types = types
        self.i = 0
        self.keep = keep
        self.holder = capped(holder)

    def __call__(self, token):
        expected = self.types[self.i]
        self.i += 1
        if token.is_a(expected):
            if self.i == len(self.types):
                return TokenSearchStatus.FINISH
            return TokenSearchStatus.CONTINUE
        else:
            return TokenSearchStatus.FAIL

    def result(self, tokens):
        return [
            self.holder(
                [token for i, token in enumerate(tokens) if i in self.keep]
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
        self.i = 0
        self.center = center
        self.keep = keep
        self.convert = convert

    def __call__(self, token):
        expected = self.types[self.i]
        self.i += 1
        if token.is_a(expected):
            if self.i == len(self.types):
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
        self.end = capped(end)

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


def capped(token):
    return tagged(token, [TokenExtraData.CAPPED])


class DataFunction:
    def __init__(self, function, data):
        self.function = function
        self.data = data

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


# ------------------------ #


def unquote(text, quote_size):
    text = text[quote_size:-quote_size]
    result = ""
    was_backslash = False
    for char in text:
        if char == "\\" and not was_backslash:
            was_backslash = True
        else:
            if was_backslash:
                char = {
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                    "'": "'",
                    '"': '"',
                    "\\": "\\"
                }[char]
            result += char
            was_backslash = False
    return result


class Program(Block):
    def startPROGRAM(self):
        self.constants = None


class Whitespace(Symbolic):
    pass


class Space(Whitespace):
    pass


class Newline(Whitespace):
    pass


class Definition(Node):
    pass


class Define(Symbolic):
    pass


class FunctionOpen(Symbolic):
    pass


class FunctionClose(Symbolic):
    pass


class GroupOpen(Symbolic):
    pass


class GroupClose(Symbolic):
    pass


class LeftAngleBracket(Symbolic):
    pass


class RightAngleBracket(Symbolic):
    pass


class ArrayOpen(Symbolic):
    pass


class ArrayClose(Symbolic):
    pass


class Comma(Symbolic):
    pass


class Operator(Symbolic):
    pass


class LessEqualOperator(Operator):
    pass


class GreaterEqualOperator(Operator):
    pass


class Typer(Symbolic):
    # A colon, used the same way as weak typing
    # in python, however this typing is strong
    pass


class EqualsOperator(Operator):
    pass


class NotEqualsOperator(Operator):
    pass


class AndOperator(Operator):
    pass


class OrOperator(Operator):
    pass


class NotOperator(Operator):
    pass


class AssignOperator(Operator):
    pass


class AdditionOperator(Operator):
    pass


class MultiplicationOperator(Operator):
    pass


class DivisionOperator(Operator):
    pass


class ExponentiationOperator(Operator):
    pass


class Instantiater(Operator):
    pass


class MinusOperator(Operator):  # Could be subtraction or negation
    pass


class ReturnOperator(Operator):
    pass


class Tuple(Operand):
    def _compute_type(self):
        child_types = [child.type_ for child in self.value]
        if any([type_.is_none() for type_ in child_types]):
            return Type.none
        return Type(BasicType.tuple, child_types)

    def instruction(self):
        return make_instruction(
            TupleInstruction,
            self.value,
            self.type_
        )


class UnfinishedTuple(Node):
    pass


class Array(Operand):
    def _compute_type(self):
        elem = self.value[0].type_
        if elem.is_none():
            return Type.none
        return Type(BasicType.array, [elem])

    def instruction(self):
        return make_instruction(
            ArrayInstruction,
            self.value,
            self.type_
        )


class UnfinishedArray(Node):
    pass


class FunctionCall(Operand):
    def _compute_type(self):
        func, _ = self.value
        func_type = func.type_
        if not func_type.generics:
            return Type.none
        return func.type_.generics[0]

    def instruction(self):
        return make_instruction(
            FunctionCallInstruction,
            self.value,
            self.type_
        )


class Instantiation(Operand):
    def _compute_type(self):
        child, = self.value
        return child.to_type()

    def instruction(self):
        return make_instruction(
            InstantiationInstruction,
            [],
            self.type_,
            {"instantiate_type": self.type_}
        )


class Assignment(Operand):
    def _compute_type(self):
        _, child = self.value
        return child.type_

    def instruction(self):
        target, value = self.value
        return make_instruction(
            AssignInstruction,
            [value],
            self.type_,
            {"var_id": target.id_}
        )


class Unpacking(Operand):
    def _compute_type(self):
        _, child = self.value
        return child.type_

    def instruction(self):
        targets, value = self.value
        return make_instruction(
            UnpackInstruction,
            [value],
            self.type_,
            {"var_ids": [
                target.id_
                for target in targets.value
            ]}
        )


class Equals(Operand):
    def _compute_type(self):
        return Type.bool

    def instruction(self):
        return make_instruction(
            EqualsInstruction,
            self.value,
            self.type_
        )


class NotEquals(Operand):
    def _compute_type(self):
        return Type.bool

    def instruction(self):
        return make_instruction(
            NotEqualsInstruction,
            self.value,
            self.type_
        )


class Less(Operand):
    def _compute_type(self):
        return Type.bool

    def instruction(self):
        return make_instruction(
            LessInstruction,
            self.value,
            self.type_
        )


class Greater(Operand):
    def _compute_type(self):
        return Type.bool

    def instruction(self):
        return make_instruction(
            GreaterInstruction,
            self.value,
            self.type_
        )


class LessEqual(Operand):
    def _compute_type(self):
        return Type.bool

    def instruction(self):
        return make_instruction(
            LessEqualInstruction,
            self.value,
            self.type_
        )


class GreaterEqual(Operand):
    def _compute_type(self):
        return Type.bool

    def instruction(self):
        return make_instruction(
            GreaterEqualInstruction,
            self.value,
            self.type_
        )


class And(Operand):
    def _compute_type(self):
        return Type.bool

    def instruction(self):
        return make_instruction(
            AndInstruction,
            self.value,
            self.type_
        )


class Or(Operand):
    def _compute_type(self):
        return Type.bool

    def instruction(self):
        return make_instruction(
            OrInstruction,
            self.value,
            self.type_
        )


class Not(Operand):
    def _compute_type(self):
        return Type.bool

    def instruction(self):
        return make_instruction(
            NotInstruction,
            self.value,
            self.type_
        )


class Addition(Operand):
    def _compute_type(self):
        v1, v2 = self.value
        v1 = v1.type_
        v2 = v2.type_
        if v1.is_none() or v2.is_none():
            return Type.none
        if v2.type_ == BasicType.str:
            return v2
        if v1.type_ == BasicType.tuple:
            return Type(
                BasicType.tuple,
                v1.generics + v2.generics
            )
        return v1

    def instruction(self):
        instruction = None
        basic_type = self.type_.type_
        if basic_type == BasicType.float:
            instruction = AdditionInstruction
        elif basic_type == BasicType.str:
            instruction = ConcatenationInstruction
        elif basic_type == BasicType.array:
            instruction = JoinArrayInstruction
        elif basic_type == BasicType.tuple:
            instruction = JoinTupleInstruction
        return make_instruction(
            instruction,
            self.value,
            self.type_
        )


class Subtraction(Operand):
    def _compute_type(self):
        child, _ = self.value
        return child.type_

    def instruction(self):
        return make_instruction(
            SubtractionInstruction,
            self.value,
            self.type_
        )


class Exponentiation(Operand):
    def _compute_type(self):
        child, _ = self.value
        return child.type_
        
    def instruction(self):
        return make_instruction(
            ExponentiationInstruction,
            self.value,
            self.type_
        )


class Multiplication(Operand):
    def _compute_type(self):
        child, _ = self.value
        return child.type_
        
    def instruction(self):
        return make_instruction(
            MultiplicationInstruction,
            self.value,
            self.type_
        )


class Division(Operand):
    def _compute_type(self):
        child, _ = self.value
        return child.type_
        
    def instruction(self):
        return make_instruction(
            DivisionInstruction,
            self.value,
            self.type_
        )


class Negation(Operand):
    def _compute_type(self):
        child, = self.value
        return child.type_

    def instruction(self):
        return make_instruction(
            NegationInstruction,
            self.value,
            self.type_
        )


class Group(Operand):
    def _compute_type(self):
        child, *_ = self.value
        assert not _, self.value
        return child.type_

    def instruction(self):
        child, = self.value
        return child.instruction()


class Return(Operand):
    def _compute_type(self):
        child, = self.value
        return child.type_

    def instruction(self):
        return make_instruction(
            ReturnInstruction,
            self.value,
            self.type_
        )


class Refrence(Operand):  # Anything that can be refrenced in code
    pass


class Variable(Refrence):
    def startVARIABLE(self):
        self.id_ = None

    def string_data(self):
        return {"id_": self.id_}

    def instruction(self):
        return make_instruction(
            VariableInstruction,
            [],
            self.type_,
            {"var_id": self.id_}
        )


class Constant(Refrence):
    def instruction(self):
        name = condense_tokens(self.value)
        return make_instruction(
            ConstantInstruction,
            [],
            self.type_,
            {"value": self.search_parent(Program).constants.get(name)}
        )


class RefrenceWrapper: # A object linking refrences in the AST
    def __init__(self, type_, id_=None):
        assert isinstance(type_, Type), f"Type must be Type (found {type(type_)})"
        self.type_ = type_
        self.id_ = id_

    def __str__(self):
        return repr(self) # str(self.type_)

    def __repr__(self):
        return f"RefrenceWrapper<{self.type_}>"


class Function(Operand, Block):
    def _compute_type(self):
        def return_visit(token, arguments):
            if token.is_a(Return):
                return_type = token.type_
                if return_type:
                    return return_type
                else:
                    return Type.tuple
        return_type = self.visit(
            return_visit, 
            tuple(), 
            lambda t: not t.is_a(Block)
        )
        if return_type:
            if return_type.is_none():
                return Type.none
            return Type(BasicType.func, [return_type])
        return Type(BasicType.func, [Type.tuple])



class TypeToken(Node):
    def to_type(self):
        return Type(BasicType[condense_tokens(self.value)])


class GenericList(Node):
    pass


class UnfinishedGenericList(Node):
    pass


class GenericTypeToken(TypeToken):
    def to_type(self):
        basic_type, generics = self.value
        return basic_type.to_type().with_generics([
            generic.to_type()
            for generic in generics.value
        ])


class ArgumentFunction(Operand):
    def _compute_type(self):
        args, func = self.value
        if func.type_.is_none():
            return Type.none
        return_value, = func.type_.generics
        return func.type_.with_generics(
            [
                return_value,
                *[
                    generic.value[1].to_type()
                    for generic in args.value
                ]
            ]
        )
    
    def instruction(self):
        args, func = self.value
        return make_instruction(
            FunctionInstruction,
            func.value,
            self.type_,
            {"arguments": args.value}
        )


class Argument(Node):
    def startARGUMENT(self):
        self.id_ = None

    def string_data(self):
        return {"id_": self.id_}


class Arguments(Node):
    pass


class UnfinishedArguments(Node):
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


class DefenitionRefrenceSearch:
    def __init__(self, name):
        self.name = name
    
    def __call__(self, token):
        if not token.is_a(Name):
            return TokenSearchStatus.FAIL
        name = condense_tokens(token.value)
        if name == self.name:
            return TokenSearchStatus.FINISh


class DefinitionRule:
    def __init__(self, code):
        self.code = code

    def __call__(self, token):
        if not token.is_a(Definition):
            return TokenSearchStatus.FAIL
        definition_name, value = token.value
        definition_name = condense_tokens(
            definition_name.value
        )
        def refrence_search(token, _):
            if token.is_text():
                return
            for i, sub in enumerate(token.value):
                if not sub.is_a(Name):
                    continue
                name = condense_tokens(sub.value)
                if name != definition_name:
                    continue
                token.value[i] = value.copy()
        self.code.visit(refrence_search, tuple())
        return TokenSearchStatus.FINISH

    def result(self, _):
        return []


class ReturnRule:
    def __call__(self, token):
        if not token.is_a(Function):
            return TokenSearchStatus.FAIL
        child, *extra = token.value
        if extra:
            return TokenSearchStatus.FAIL
        return (
            TokenSearchStatus.FINISH 
            if child.is_a(Operand) and not child.is_a(Return) else
            TokenSearchStatus.FAIL
        )

    def result(self, tokens):
        token, = tokens
        return_value, = token.value
        return [token.clone([Return([return_value])])]


def parse(code, verbose=False):
    # Not a great solution but whatever
    if not verbose:
        print = lambda *a, **b: None
    else:
        import builtins
        print = builtins.print
    
    print("Starting compile...")

    code = code.strip()
    constants = Constants()

    code = Program(tokenify(code))

    print("Starting string search...")

    while True:
        string = next(token_search(code, StringSearch), None)
        if string is None:
            break
        name = constants.add(unquote(
            condense_tokens(string.tokens), 1
        ))
        replace_token_search_match(code, string, [Constant(tokenify(name))])

    print("Removing comments...")

    # This is so spagehti code
    last_char = None
    multi_line_comment = False
    one_line_comment = False
    result = []
    for token in code.value:
        skip_this = False
        if token.is_text():
            char = token.value
            if not multi_line_comment:
                if char == "#":
                    one_line_comment = True
            if one_line_comment:
                if char == "\n":
                    one_line_comment = False
            if not (multi_line_comment or one_line_comment):
                if char == "*" and last_char == "/":
                    result = result[:-1]
                    multi_line_comment = True
            if multi_line_comment:
                if char == "/" and last_char == "*":
                    multi_line_comment = False
                    skip_this = True
            last_char = char
        else:
            last_char = None
        if not (multi_line_comment or one_line_comment or skip_this):
            result.append(token)

    code = Program(result)
    code.constants = constants

    print("Started tokenizing...")

    perform_conversions(code, [
        create_text_conversion("<=", LessEqualOperator),
        create_text_conversion(">=", GreaterEqualOperator),
        create_text_conversion("!=", NotEqualsOperator),
        create_text_conversion("=", EqualsOperator),
        create_text_conversion("&", AndOperator),
        create_text_conversion("|", OrOperator),
        create_text_conversion("!", NotOperator),
        create_text_conversion("->", AssignOperator),
        create_text_conversion("-", MinusOperator),
        create_text_conversion("+", AdditionOperator),
        create_text_conversion("/", DivisionOperator),
        create_text_conversion("**", ExponentiationOperator),
        create_text_conversion("*", MultiplicationOperator),
        create_text_conversion("return", ReturnOperator),
        create_text_conversion("{", FunctionOpen),
        create_text_conversion("}", FunctionClose),
        create_text_conversion("(", GroupOpen),
        create_text_conversion(")", GroupClose),
        create_text_conversion(" ", Space),
        create_text_conversion("\n", Newline),
        create_text_conversion(",", Comma),
        create_text_conversion(":", Typer),
        create_text_conversion("[", ArrayOpen),
        create_text_conversion("]", ArrayClose),
        create_text_conversion("<", LeftAngleBracket),
        create_text_conversion(">", RightAngleBracket),
        create_text_conversion("new", Instantiater),
        create_text_conversion("def", Define),
    ])

    print("Starting name search...")

    while True:
        name = next(token_search(code, NameSearch), None)
        if name is None:
            break
        replace_token_search_match(code, name, [
            tagged(Name, [TokenExtraData.UNIT])(name.tokens)
        ])

    print("Starting number search...")

    while True:
        number = next(token_search(code, NumberSearch), None)
        if number is None:
            break
        name = constants.add(float(condense_tokens(number.tokens)))
        replace_token_search_match(code, number, [Constant(tokenify(name))])

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
            # Generics
            lambda: MergeRule(
                [UnfinishedGenericList, TypeToken, Comma],
                0, (1,), UnfinishedGenericList
            ),
            lambda: GroupRule(
                [LeftAngleBracket, TypeToken, Comma, TypeToken, Comma],
                (1, 3), UnfinishedGenericList
            ),
            lambda: GroupRule(
                [LeftAngleBracket, TypeToken, Comma, TypeToken, RightAngleBracket],
                (1, 3), GenericList
            ),
            lambda: GroupRule([LeftAngleBracket, TypeToken, RightAngleBracket],
                              (1,), GenericList),
            lambda: MergeRule(
                [UnfinishedGenericList, TypeToken, RightAngleBracket], 0,
                (1,), GenericList
            ),
            
            lambda: GroupRule([TypeToken, GenericList], (0, 1), GenericTypeToken),
            lambda: GroupRule([Name, Typer, TypeToken], (0, 2), Argument),
            lambda: GroupRule([Name, Typer, Name], (0, 2), Argument),
        ],
        [   
            # Argument lists
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
        ],
        [
            lambda: GroupRule([GroupOpen, GroupClose], tuple(), Tuple),
            lambda: BoundaryRule(GroupOpen, GroupClose, Group, allow_text=True),
        ],
        [
            lambda: GroupRule(
                [Define, Name, Group],
                (1, 2), Definition
            ),
            lambda: GroupRule(
                [Define, Name, TypeToken],
                (1, 2), Definition
            ),
        ],
        [
            lambda: DefinitionRule(code),
        ],
        [
            lambda: GroupRule([Instantiater, TypeToken], (1,), Instantiation)
        ],
        [
            lambda: ConvertRule(Name, Variable),
            lambda: GroupRule([Arguments, Function], (0, 1),
                              ArgumentFunction),
        ],
        [
            lambda: GroupRule([Operand, ExponentiationOperator, Operand],
                              (0, 2), Exponentiation),
            lambda: GroupRule([Operand, MultiplicationOperator, Operand],
                              (0, 2), Multiplication),
            lambda: GroupRule([Operand, DivisionOperator, Operand],
                              (0, 2), Division),
            lambda: GroupRule([Operand, AdditionOperator, Operand],
                              (0, 2), Addition),
            lambda: GroupRule([Operand, MinusOperator, Operand],
                              (0, 2), Subtraction),
            lambda: GroupRule([MinusOperator, Operand], (1,), Negation),

            lambda: GroupRule([Operand, EqualsOperator, Operand],
                              (0, 2), Equals),
            lambda: GroupRule([Operand, NotEqualsOperator, Operand],
                              (0, 2), NotEquals),
            
            lambda: GroupRule([Operand, LeftAngleBracket, Operand],
                              (0, 2), Less),
            lambda: GroupRule([Operand, RightAngleBracket, Operand],
                              (0, 2), Greater),
            
            lambda: GroupRule([Operand, LessEqualOperator, Operand],
                              (0, 2), LessEqual),
            lambda: GroupRule([Operand, GreaterEqualOperator, Operand],
                              (0, 2), GreaterEqual),
            
            lambda: GroupRule([Operand, AndOperator, Operand],
                              (0, 2), And),
            lambda: GroupRule([Operand, OrOperator, Operand],
                              (0, 2), Or),
            lambda: GroupRule([NotOperator, Operand],
                              (1,), Not),

            # Arrays
            lambda: MergeRule(
                [UnfinishedArray, Operand, Comma],
                0, (1,), UnfinishedArray
            ),
            lambda: GroupRule(
                [ArrayOpen, Operand, Comma, Operand, Comma],
                (1, 3), UnfinishedArray
            ),
            lambda: GroupRule(
                [ArrayOpen, Operand, Comma, Operand, ArrayClose],
                (1, 3), Array
            ),
            lambda: GroupRule([ArrayOpen, Operand, ArrayClose],
                              (1,), Array),
            lambda: MergeRule(
                [UnfinishedArray, Operand, ArrayClose], 0,
                (1,), Array
            ),
            
            # Tuples
            lambda: MergeRule([UnfinishedTuple, Operand, Comma], 0,
                              (1, ), UnfinishedTuple),
            lambda: GroupRule([Operand, Comma, Operand, Comma],
                              (0, 2), UnfinishedTuple),
            lambda: GroupRule([Operand, Comma, Operand],
                              (0, 2), Tuple),
            lambda: GroupRule([Operand, Comma],
                              (0, ), Tuple),
            lambda: MergeRule([UnfinishedTuple, Operand], 0, (1, ), Tuple),
            
            lambda: GroupRule([Variable, AssignOperator, Operand],
                              (0, 2), Assignment),
            lambda: GroupRule([Tuple, AssignOperator, Operand],
                              (0, 2), Unpacking),
            lambda: GroupRule([Operand, Tuple], (0, 1), FunctionCall),
            lambda: GroupRule([Operand, Group], (0, 1), FunctionCall),
            lambda: GroupRule([ReturnOperator, Operand],
                            (1,), Return),
        ],
        [
            DataFunction(
                lambda: ReturnRule(), 
                [RuleExtraData.SEARCHALL]
            ),
        ]
    ]

    def transform_visit(token, args):
        if token.is_text():
            return
        rule, = args
        extra_data = []
        if isinstance(rule, DataFunction):
            extra_data = rule.data
        if TokenExtraData.CAPPED in token.tags and RuleExtraData.SEARCHALL not in extra_data:
            return
        if id(rule) in token.tags:
            return
        match = next(token_search(token, rule), None)
        valid = True
        if match is None:
            valid = False
        if valid and TokenExtraData.UNIT in token.tags:
            if match.start != 0 or match.end < len(token.value) - 1:
                valid = False
        if not valid:
            token.tags.append(id(rule))
            return
        token.tags = list(filter(
            lambda tag: TokenExtraData.has(tag),
            token.tags
        ))
        result = match.searcher.result(match.tokens)
        replace_token_search_match(token, match, result)
        return True

    for transform_group in transform_groups:
        print("Starting new transform group...")
        found_any = True
        while found_any:
            found_any = False
            for rule in transform_group:
                found_any = code.visit(transform_visit, (rule,))
                if found_any:
                    # print(code)
                    break

    print(code)
    code.ensure_parents()

    print("Starting type inference...")

    for name, value in constants:
        constant = RefrenceWrapper(Type(python_to_type(value)), var_id())
        code.scope.assign(name, constant)
    
    for name, type_ in BUILTINS.items():
        constant = RefrenceWrapper(type_, var_id())
        code.scope.assign(name, constant)

    def assign_visit(token, args):
        if isinstance(token, Assignment):
            name, value = token.value
            type_ = value.type_
            if type_.is_none():
                return
            scope = token.search_parent(Block).scope
            target = condense_tokens(name.value)
            if scope.get(target) is not None:
                return
            changed = scope.assign(
                target, RefrenceWrapper(value.type_, name.id_)
            )
            if changed:
                return True

    def reference_visit(token, args):
        if isinstance(token, Refrence):
            if not token.type_.is_none():
                return
            name = condense_tokens(token.value)
            scope = token.search_parent(Block).scope
            source = scope.get(name)
            if source is None:
                return
            type_ = source.type_
            if type_ is None or type_.is_none():
                return
            token.type_ = type_
            if source.id_ is not None:
                token.id_ = source.id_
            return True

    def compute_visit(token, args):
        if isinstance(token, Operand):
            if not token.type_.is_none():
                return
            changed = token.compute_type()
            return changed

    def arguments_visit(token, args):
        if isinstance(token, ArgumentFunction):
            arguments, function = token.value
            for argument in arguments.value:
                name, type_ = argument.value
                id_ = var_id()
                argument.id_ = id_
                # print(id(function.scope), condense_tokens(name.value),
                #       type_.to_type())
                refrence = RefrenceWrapper(type_.to_type(), id_)
                function.scope.assign(
                    condense_tokens(name.value), 
                    refrence
                )

    def unpack_visit(token, args):
        if isinstance(token, Unpacking):
            values, tuple = token.value
            tuple_type = tuple.type_
            if tuple_type.type_ != BasicType.tuple:
                return
            scope = token.search_parent(Block).scope
            changed_any = False
            # print(tuple_type, tuple_type.generics)
            for value, type_ in zip(values.value, tuple_type.generics):
                name = condense_tokens(value.value)
                if scope.get(name) is not None:
                    continue
                # print(scope.scope, id(scope), name, type_)
                if type_ is None or type_.is_none():
                    continue
                changed = scope.assign(name, RefrenceWrapper(type_, var_id()))
                # print(scope.scope)
                if changed:
                    changed_any |= True
            return changed_any

    def variable_visit(token, args):
        if isinstance(token, Variable):
            token.id_ = var_id()
    
    found_any = True
    code.visit(arguments_visit, tuple())
    code.visit(variable_visit, tuple())
    while True:
        # print(code)
        found_any = False
        found_any = code.visit(compute_visit, tuple())
        if found_any:
            continue
        found_any = code.visit(assign_visit, tuple())
        if found_any:
            continue
        found_any = code.visit(reference_visit, tuple())
        if found_any:
            continue
        found_any = code.visit(unpack_visit, tuple())
        if found_any:
            continue
        break

    print(code)
    
    return code


# Cool regex:
# /for .* in .*\.value/
# /class \w*:\n\s*\n/
