# Implement definitions as shown here later

def Tester tuple<
    float,
    tuple<
        str,
        str
    >
>

t1 -> new Tester
t2 -> (1, ("a", "b"))

f -> (a: Tester) {
    v1, v2 = a
    v3, v4 = v1
    print(v1 + " " + v3 + " " + v4)
}

f(t1)
f(t2)