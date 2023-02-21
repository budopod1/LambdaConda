from id_ import IDGetter
from interpreter import SpecialInstruction, SpecialInstructionType
from type_ import Type, BasicType
from builtins_ import BUILTINS


def make_instruction(instruction_type, children, type_, extra=None):
    return FloatingInstruction(
        instruction_type,
        [
            (
                child 
                if is_instruction(child) 
                else child.instruction()
            )
            for child in children
        ],
        type_,
        extra
    )


def is_instruction(x):
    if isinstance(x, FloatingInstruction):
        return True
    if isinstance(x, Instruction):
        return True
    if isinstance(x, int):
        return True


class Executable:
    def __init__(self):
        self.functions = {}
        self.builtins = {}

    def add_function(self, name, signature, arguments):
        function = Function(name, signature, arguments, self)
        self.functions[name] = function
        return function

    def get_function(self, name):
        return self.functions[name]

    def __str__(self):
        return "\n".join(
            [str(function) for function in self.functions.values()]
        )

    def __repr__(self):
        return str(self)


value_id = IDGetter()


class Function:
    def __init__(self, name, signature, arguments, executable):
        self.name = name
        self.signature = signature
        self.executable = executable
        self.arguments = arguments
        self.instructions = []

    def __iter__(self):
        return (
            instruction 
            for instruction in self.instructions
        )

    def append(self, instruction):
        self.instructions.append(instruction)

    def add_instruction(self, instruction):
        instruction, params, type_, data = instruction
        instruction = instruction(params, type_, self, data, **data)
        self.instructions.append(instruction)
        return instruction

    def __str__(self):
        text = f"Function {self.name}: ({self.signature})\n"
        for instruction in self.instructions:
            text += "\t" + str(instruction) + "\n"
        return text

    def __repr__(self):
        return str(self)


class FloatingInstruction:  # dataclass?
    def __init__(self, instruction, params, type_, data=None):
        if data is None:
            data = {}
        self.instruction = instruction
        self.params = params
        self.type_ = type_ # The type when used as a parameter
        self.data = data

    def __iter__(self):
        yield self.instruction
        yield self.params
        yield self.type_
        yield self.data


class Instruction:
    def __init__(self, params, type_, function, extra):
        self.extra = extra
        self.id_ = value_id()
        self.type_ = type_
        self.params = params
        self.function = function
        self.setup()
        params = []
        for param in self.params:
            if isinstance(param, FloatingInstruction):
                instruction = self.function.add_instruction(param)
                params.append(instruction.id_)
            elif isinstance(param, Instruction):
                params.append(param.id_)
            elif isinstance(param, int):
                params.append(param)
        self.params = params

    def setup(self):
        pass

    def __str__(self):
        name = type(self).__name__
        params = ", ".join([str(param) for param in self.params])
        extra = "|" + str(self.extra) if self.extra else ""
        return f"{self.id_}: {name}<{self.type_}>({params}{extra})"

    def __repr__(self):
        return str(self)


class ConstantInstruction(Instruction):
    def __init__(self, *args, value):
        self.value = value
        super().__init__(*args)

    def interpret(self, scope):
        return self.value


class AssignInstruction(Instruction):
    def __init__(self, *args, var_id):
        self.var_id = var_id
        super().__init__(*args)

    def interpret(self, scope, value):
        scope[self.var_id] = value
        return value


# class RefrencerInstruction


class VariableInstruction(Instruction):
    def __init__(self, *args, var_id):
        super().__init__(*args)
        self.var_id = var_id

    def interpret(self, scope):
        if scope.is_builtin(self.var_id):
            return scope.get_builtin(self.var_id)
        return scope[self.var_id]


class InstantiationInstruction(Instruction):
    def __init__(self, *args, instantiate_type):
        super().__init__(*args)
        self.instantiate_type = instantiate_type
        
    def interpret(self, scope):
        def instantiate(type_):
            btype = type_.type_
            if btype == BasicType.str:
                return ""
            elif btype == BasicType.float:
                return 0.0
            elif btype == BasicType.int:
                return 0
            elif btype == BasicType.bool:
                return False
            elif btype == BasicType.func:
                return_type, *_ = btype.generics
                return lambda *a: instantiate(return_type)
            elif btype == BasicType.tuple:
                return tuple((instantiate(generic) for generic in type_.generics))
            elif btype == BasicType.array:
                return []
        return instantiate(self.instantiate_type)


class FunctionInstruction(Instruction):
    def __init__(self, *args, arguments=None):
        self.arguments = arguments
        super().__init__(*args)
        self.params = []
        self.new_function = self.function
        self.function = self.old_function
    
    def setup(self):
        executable = self.function.executable
        self.old_function = self.function
        self.function = executable.add_function(self.id_, self.type_,
                                                self.arguments)

    def interpret(self, scope):
        return self.new_function.name


class UnpackInstruction(Instruction):
    def __init__(self, *args, var_ids):
        self.var_ids = var_ids
        super().__init__(*args)

    def interpret(self, scope, value):
        for var_id, item in zip(self.var_ids, value):
            scope[var_id] = item
        return value


class FunctionCallInstruction(Instruction):
    def interpret(self, scope, function, arguments):
        return SpecialInstruction(
            SpecialInstructionType.CALLFUNCTION, 
            [function, arguments]
        )


class SubtractionInstruction(Instruction):
    def interpret(self, scope, v1, v2):
        return float(v1) - float(v2)


class AdditionInstruction(Instruction):
    def interpret(self, scope, v1, v2):
        return float(v1) + float(v2)


class ConcatenationInstruction(Instruction):
    def interpret(self, scope, v1, v2):
        return str(v1) + str(v2)


class JoinArrayInstruction(Instruction):
    def interpret(self, scope, v1, v2):
        return list(v1) + (v2)


class JoinTupleInstruction(Instruction):
    def interpret(self, scope, v1, v2):
        return tuple(v1) + tuple(v2)


class NegationInstruction(Instruction):
    def interpret(self, scope, value):
        return -value


class TupleInstruction(Instruction):
    def interpret(self, scope, *values):
        return tuple(values)


class ReturnInstruction(Instruction):
    def interpret(self, scope, value):
        return SpecialInstruction(
            SpecialInstructionType.RETURN, 
            [value]
        )


class ArrayInstruction(Instruction):
    def interpret(self, scope, *values):
        return list(values)


def convert(code, verbose=False):
    executable = Executable()
    
    for name in BUILTINS:
        refrence = code.scope.get(name)
        executable.builtins[refrence.id_] = name
    
    main = executable.add_function(
        "main", Type(BasicType.func, [Type.none]), None
    )

    function_instruction = main.add_instruction(code.instruction())
    tuple_instruction = main.add_instruction(FloatingInstruction(
        TupleInstruction,
        tuple(),
        Type(BasicType.tuple)
    ))
    main.add_instruction(FloatingInstruction(
        FunctionCallInstruction, 
        (function_instruction, tuple_instruction),
        None
    ))

    if verbose:
        print(executable)

    return executable
