from type_ import Type, BasicType


BUILTINS = {
    "print": Type(BasicType.func, [Type.none, Type(BasicType.arbitrary)]),
    "for": Type(
        BasicType.func, 
        [Type.none, Type(BasicType.arbitrary), Type(BasicType.arbitrary)]
    ),
}


__all__ = ["BUILTINS"]
