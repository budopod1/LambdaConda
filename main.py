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


def run_named(file, verbose=False):
    print(f"Running {file}...")
    run(file, verbose)


if __name__ == "__main__":
    run_named("examples/argument.ll")
    run_named("examples/array.ll")
    run_named("examples/definition.ll", True)
    run_named("examples/for.ll")
    run_named("examples/hello.ll")
    run_named("examples/scope.ll")
    run_named("examples/test.ll")
    run_named("examples/tuple.ll")
