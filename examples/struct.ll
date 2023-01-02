# Implement structs as shown here later

struct Tester tuple<
    float,
    tuple<
        str,
        str
    >
>

# NOTE: initate all primitives in struct as if they were being called by 
# constructor in python (e.g. float()=0)
t1 -> new Tester
t2 -> (1, ("a", "b"))

f -> (a: Tester) {
    v1, v2 = a
    v3, v4 = v1
    print(v1 + " " + v3 + " " + v4)
}

f(t1)
f(t2)