from pyenum import Enum


BasicType = Enum("BasicType", "int", "float", "bool", "str", "func", "none",
                 "tuple", "array", "arbitrary")


class Type:
    def __init__(self, type_, generics=None):
        self.type_ = type_
        if generics is None:
            generics = []
        self.generics = generics

    def with_generics(self, generics):
        return Type(self.type_, generics)

    def is_none(self):
        return self.type_ == BasicType.none

    def __eq__(self, other):
        if self.type_==BasicType.arbitrary or other.type_==BasicType.arbitrary:
            return True
        if other.type_ != self.type_:
            return False
        if len(self.generics) != len(other.generics):
            return False
        for a, b in zip(self.generics, other.generics):
            if a != b:
                return False
        return True

    def __ne__(self, other):
        return not (self == other)

    def __str__(self):
        generics = [str(generic) for generic in self.generics]
        return f"{self.type_}<{', '.join(generics)}>"

    def __repr__(self):
        return str(self)


Type.none = Type(BasicType.none)
