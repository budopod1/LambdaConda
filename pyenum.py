class EnumValue:
    def __init__(self, enum, value):
        self.enum = enum
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, EnumValue):
            raise TypeError(f"Cannot compare EnumValue with {type(other)}")
        return self.enum == other.enum and self.value == other.value

    def __ne__(self, other):
        return not (other == self)

    def __repr__(self):
        return f"{self.enum.name}.{self.enum.values[self.value]}"

    def __str__(self):
        return repr(self)


class Enum:
    def __init__(self, name, *values):
        self.name = name
        self.values = {}
        self.contains = values
        for i, value in enumerate(values):
            self.__dict__[value] = i
            self.values[i] = value

    def __iter__(self):
        return (value for value in self.contains)

    def __getattribute__(self, key):
        __dict__ = object.__getattribute__(self, "__dict__")
        value = object.__getattribute__(self, key)
        if "contains" in __dict__:
            contains = object.__getattribute__(self, "contains")
            if key in contains:
                return EnumValue(self, value)
        return value

    def __getitem__(self, key):
        return EnumValue(self, self.__dict__[key])
