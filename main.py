from ast import parse
from instructions import convert
from interpreter import interpret


def run(file):
    with open(file) as file:
        code = file.read()
    # print(code)
    code = parse(code)
    # print(code)
    code = convert(code)
    # print(code)
    interpret(code)


if __name__ == "__main__":
    # run("examples/argument.ll")
    # run("examples/array.ll")
    # run("examples/scope.ll")
    # run("examples/tuple.ll")
    # run("examples/hello.ll")
    run("examples/test.ll")
    # run("examples/for.ll")
