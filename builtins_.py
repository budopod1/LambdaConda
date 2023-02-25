from type_ import Type, BasicType


BUILTINS = {
    "print": Type(BasicType.func, [Type.tuple, Type(BasicType.arbitrary)]),
    "for": Type(
        BasicType.func, 
        [Type.tuple, Type(BasicType.arbitrary), Type(BasicType.arbitrary)]
    ),
    "if": Type(
        BasicType.func,
        [
            Type.tuple,
            Type(
                BasicType.func,
                [
                    BasicType.bool
                ]
            ),
            Type(
                BasicType.func,
                [
                    Type.tuple
                ]
            )
        ]
    ),
    "while": Type(
        BasicType.func,
        [
            Type.tuple,
            Type(
                BasicType.func,
                [
                    BasicType.bool
                ]
            ),
            Type(
                BasicType.func,
                [
                    BasicType.tuple
                ]
            )
        ]
    ),
    "true": Type(BasicType.bool),
    "false": Type(BasicType.bool),
}


__all__ = ["BUILTINS"]
