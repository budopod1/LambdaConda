from id_ import IDGetter


def make_constant(constant_type, value):
    return FloatingInstruction(
        ConstantInstruction,
        [],
        constant_type,
        {"value": value}
    )


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
        self.functions = []
        self.constants = {}

    def add_function(self, name):
        function = Function(name, self)
        self.functions.append(function)
        return function

    def add_constant(self, id_, value):
        self.constants[id_] = value

    def __str__(self):
        return "\n".join(
            [str(function) for function in self.functions]
        )

    def __repr__(self):
        return str(self)


value_id = IDGetter()


class Function:
    def __init__(self, name, executable):
        self.name = name
        self.executable = executable
        self.instructions = []

    def append(self, instruction):
        self.instructions.append(instruction)

    def add_instruction(self, instruction):
        instruction, params, type_, data = instruction
        instruction = instruction(params, type_, self, **data)
        self.instructions.append(instruction)
        return instruction

    def __str__(self):
        text = "Function:\n"
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
    def __init__(self, params, type_, function):
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
        return f"{self.id_}: {name}<{self.type_}>({params})"

    def __repr__(self):
        return str(self)


class ConstantInstruction(Instruction):
    def __init__(self, *args, value):
        self.value = value
        super().__init__(*args)
    
    def setup(self):
        self.function.executable.add_constant(
            self.id_,
            self.value
        )


class AssignInstruction(Instruction):
    def __init__(self, *args, var_id):
        self.var_id = var_id
        super().__init__(*args)


# class RefrencerInstruction


class VariableInstruction(Instruction):
    def __init__(self, *args, var_id):
        super().__init__(*args)
        self.var_id = var_id


class InstantiationInstruction(Instruction):
    pass


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
        self.function = executable.add_function(self.id_)


class UnpackInstruction(Instruction):
    def __init__(self, *args, var_ids):
        self.var_ids = var_ids
        super().__init__(*args)


class FunctionCallInstruction(Instruction):
    pass


class AdditionInstruction(Instruction):
    pass
    # Replace AdditionInstruction /w instructions for individual types
    # (eg AdditonInstruction, ConcatinationInstruction, JoinInstruction)


class ConcatenationInstruction(Instruction):
    pass


class JoinInstruction(Instruction):
    pass


class NegationInstruction(Instruction):
    pass


class TupleInstruction(Instruction):
    pass


class ReturnInstruction(Instruction):
    pass


class ArrayInstruction(Instruction):
    pass


def convert(code):
    executable = Executable()
    main = executable.add_function("main")

    function_instruction = main.add_instruction(code.instruction())
    main.add_instruction(FloatingInstruction(
        FunctionCallInstruction, 
        (function_instruction,),
        None
    ))

    return executable
