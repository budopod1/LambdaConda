hello -> (person: tuple<str, float>) {
    name, age -> person
    print("Hello " + name + ", you are " + age + " years old!")
}

/*
Also test comments
*/

hello2 -> (name: str, age: float) {
    print("Hello " + name + ", you are " + age + " years old!")
}

# This:
hello(("me", 4),)
hello(("you", 2),)

# Does the same as this:
hello2("me", 3)
hello2("you", 1)
