from ast import parse
from instructions import convert


def run(file):
    with open(file) as file:
        text = file.read()
    code, constants = parse(text)
    print(code)
    code = convert(code)
    print(code)
    print(constants)


if __name__ == "__main__":
    run("examples/argument.ll")
    run("examples/array.ll")
    run("examples/scope.ll")
    run("examples/tuple.ll")
