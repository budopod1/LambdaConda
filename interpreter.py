from pyenum import Enum
from builtins_ import BUILTINS
from type_ import Type, BasicType
from cast import stringify, boolify, floatify


class Scope:
    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.outputs = {}
        self.variables = {}

    def is_builtin(self, var_id):
        return var_id in self.interpreter.executable.builtins

    def get_builtin(self, var_id):
        return self.interpreter.get_builtin(var_id)

    def __getitem__(self, key):
        return self.variables[key]

    def __setitem__(self, key, value):
        self.variables[key] = value

    def __str__(self):
        result = ""
        result += f"Outputs: {self.outputs}\n"
        result += f"Variables: {self.variables}"
        return result


SpecialInstructionType = Enum(
    "SpecialInstruction", 
    "CALLFUNCTION",
    "RETURN"
)


class SpecialInstruction:
    def __init__(self, command_type, arguments):
        self.command_type = command_type
        self.arguments = arguments


def interpret(executable):
    interpreter = Interpreter(executable)
    interpreter.interpret()


class Interpreter:
    def __init__(self, executable):
        self.executable = executable
        self.scope = Scope(self)
        self.builtins = {
            "print": self.func_print,
            "for": self.func_for,
            "if": self.func_if,
            "while": self.func_while,
            "true": True,
            "false": False
        }

    def get_builtin(self, var_id):
        name = self.executable.builtins[var_id]
        builtin = self.builtins[name]
        if callable(builtin): # If it's a function, return the name
            return name
        return builtin # Otherwise return the value

    def func_print(self, value):
        print(stringify(value))

    def func_for(self, value, func):
        if isinstance(value, float):
            value = int(value)
        if isinstance(value, int):
            value = range(value)
        value = iter(value)
        for item in value:
            self.call_function(func, item)

    def func_if(self, cond, block):
        if boolify(self.call_function(cond)):
            self.call_function(block)

    def func_while(self, cond, block):
        while boolify(self.call_function(cond)):
            self.call_function(block)

    def interpret(self):
        main_func = self.executable.get_function("main")
        self.interpret_function(main_func)

    def call_builtin(self, func_name, arguments):
        return self.builtins[func_name](*arguments)

    def call_function(self, func_name, arguments=tuple()):
        if func_name in self.builtins:
            signature =  BUILTINS[func_name]
        else:
            function = self.executable.get_function(func_name)
            signature = function.signature
        param_num = len(signature.generics) - 1
        if param_num == 0:
            arguments = tuple()
        elif param_num == 1:
            arguments = (arguments,)
        if func_name in self.builtins:
            return self.call_builtin(func_name, arguments)
        return self.interpret_function(
            function, *arguments
        )
    
    def special_instruction(self, instruction):
        command_type = instruction.command_type
        if command_type == SpecialInstructionType.CALLFUNCTION:
            func_name, arguments = instruction.arguments
            return self.call_function(func_name, arguments), None
        elif command_type == SpecialInstructionType.RETURN:
            value, = instruction.arguments
            return None, value

    def interpret_function(self, function, *arguments):
        if function.arguments is not None:
            for argument, value in zip(function.arguments, arguments):
                self.scope[argument.id_] = value
        for instruction in function:
            result = instruction.interpret(
                self.scope, 
                *[
                    self.scope.outputs.pop(param) 
                    for param in instruction.params
                ]
            )
    
            if isinstance(result, SpecialInstruction):
                result, return_value = self.special_instruction(result)
                if return_value is not None:
                    return return_value
            self.scope.outputs[instruction.id_] = result
        return tuple()
