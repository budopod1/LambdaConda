from pyenum import Enum


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
            "for": self.func_for
        }

    def get_builtin(self, var_id):
        name = self.executable.builtins[var_id]
        builtin = self.builtins[name]
        if callable(builtin): # If it's a function, return the name
            return name
        return builtin # Otherwise return the value

    def func_print(self, arguments):
        print(arguments)

    def func_for(self, arguments):
        value, func = arguments
        if isinstance(value, float):
            value = int(value)
        if isinstance(value, int):
            value = range(value)
        value = iter(value)
        for item in value:
            self.interpret_function(func, (item,))

    def interpret(self):
        self.interpret_function("main", tuple())

    def call_builtin(self, func_name, arguments):
        return self.builtins[func_name](arguments)

    def special_instruction(self, instruction):
        command_type = instruction.command_type
        if command_type == SpecialInstructionType.CALLFUNCTION:
            func_name, arguments = instruction.arguments
            if func_name in self.builtins:
                return self.call_builtin(func_name, arguments)
            return self.interpret_function(
                func_name, arguments
            )

    def interpret_function(self, function_name, arguments):
        function = self.executable.get_function(function_name)
        for instruction in function:
        
            result = instruction.interpret(
                self.scope, 
                *[
                    self.scope.outputs.pop(param) 
                    for param in instruction.params
                ]
            )
    
            if isinstance(result, SpecialInstruction):
                result = self.special_instruction(result)
            self.scope.outputs[instruction.id_] = result
