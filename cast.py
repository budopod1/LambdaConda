def stringify(value):
    if isinstance(value, bool):
        if value:
            return "true"
        else:
            return "false"
    if isinstance(value, (float, int, str)):
        return str(value)
    iterable = False
    if isinstance(value, list):
        iterable = True
        start = "["
        end = "]"
        sep = ", "
    if isinstance(value, tuple):
        iterable = True
        start = "("
        end = ")"
        sep = ", "
    if iterable:
        result = start
        for i, part in enumerate(value):
            if i != 0:
                result += sep
            result += stringify(part)
        result += end
        return result


def floatify(value):
    return float(value)


def boolify(value):
    return bool(value)
