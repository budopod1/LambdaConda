from ast import parse


def run(file):
    with open(file) as file:
        text = file.read()
    code, constants = parse(text)
    print(code)
    print(constants)


if __name__ == "__main__":
    run("examples/argument.ll")
    run("examples/array.ll")
    run("examples/scope.ll")
    run("examples/tuple.ll")
