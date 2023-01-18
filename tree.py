from ast import *


code = Addition([
    Variable([Text("a")]),
    Subtraction([
        Variable([Text("b")]),
        Variable([Text("c")]),
    ])
])
code.ensure_parents()
token = code
index = []
while True:
    # visit "token"
    if token.value and token.is_tree():
        index.append(0)
        token = token.value[0]
    else:
        done = False
        while True:
            index[-1] += 1
            curr_index = index[-1]
            # print(token.parent)
            children = token.parent.value
            if curr_index < len(children):
                break
            token = token.parent
            index.pop()
            if not index:
                done = True
                break
        if done:
            break
        token = token.parent.value[index[-1]]
