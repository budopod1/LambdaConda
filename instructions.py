# WIP
class Executable:
    def __init__(self):
        self.functions = []


class Function:
    def __init__(self, executable):
        self.executable = executable
        self.instructions = []


class Instruction:
    def __init__(self, children):
        self.children = children


class AddInstruction(Instruction):
    pass
