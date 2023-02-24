from type_ import Type, BasicType


BUILTINS = {
    "print": Type(BasicType.func, [Type.none, Type(BasicType.arbitrary)]),
    "for": Type(
        BasicType.func, 
        [Type(BasicType.none), Type(BasicType.arbitrary), Type(BasicType.arbitrary)]
    ),
    "if": Type(
        BasicType.func,
        [
            Type.none,
            Type(
                BasicType.func,
                [
                    BasicType.bool
                ]
            ),
            Type(
                BasicType.func,
                [
                    BasicType.none
                ]
            )
        ]
    ),
    "while": Type(
        BasicType.func,
        [
            Type.none,
            Type(
                BasicType.func,
                [
                    BasicType.bool
                ]
            ),
            Type(
                BasicType.func,
                [
                    BasicType.none
                ]
            )
        ]
    ),
    "true": Type(BasicType.bool),
    "false": Type(BasicType.bool),
}


__all__ = ["BUILTINS"]
