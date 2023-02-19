from ast import parse
from instructions import convert
from interpreter import interpret


def run(file, verbose=False):
    with open(file) as file:
        code = file.read()
    # print(code)
    code = parse(code, verbose)
    # print(code)
    code = convert(code, verbose)
    # print(code)
    interpret(code)


def runNamed(file):
    print(f"Running {file}...")
    run(file)


if __name__ == "__main__":
    runNamed("examples/argument.ll")
    runNamed("examples/array.ll")
    # runNamed("examples/definition.ll")
    runNamed("examples/for.ll")
    runNamed("examples/hello.ll")
    runNamed("examples/scope.ll")
    runNamed("examples/test.ll")
    runNamed("examples/tuple.ll")
