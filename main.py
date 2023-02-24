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
    from timer import Timer
    timer = Timer()
    run_named("examples/argument.lc")
    run_named("examples/array.lc")
    run_named("examples/bool.lc")
    run_named("examples/definition.lc")
    run_named("examples/for.lc")
    run_named("examples/hello.lc")
    run_named("examples/math.lc")
    run_named("examples/scope.lc")
    run_named("examples/test.lc")
    run_named("examples/tuple.lc")
    print(timer.elapsed())
