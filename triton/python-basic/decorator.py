# version1
# nested function
def outer(x):
    print(f"x is {x}")
    def inner(y):
        print(f"y is {y}")
        return x + y
    return inner

# 传入x = 5，然后返回的object是inner(y)
add_five = outer(5)
# inner()中传入6，所以是5 + 6
result = add_five(6)
print(f"result is {result}")

# version2
# pass function as parameter
def add(x, y):
    return x + y

def outer(fn, x, y):
    return fn(x,y)

print(outer(add, 3, 4))

# version3 return function as value
def greet(word):
    def inner():
        return f"hello {word}"
    return inner

sentence = greet("leon")
print(sentence())

# version4 a step before decorated function
def makePretty(func):
    def inner():
        print("decorated here")
        func()

    return inner

def greet():
    print("Hello!")

decorated = makePretty(greet)
decorated()

# version5 a simple decorator design pattern
def makePretty(func):
    def inner():
        print("decorated here!")
        func()
    return inner

@makePretty
def greet():
    print("hello")

greet()

# version6 a decorator that can check args safety
def smart_divide(fn):
    def inner(a, b):
        print("I am going to divide", a, "and", b)
        if b == 0:
            print("Whoops! cannot divide")
            return

        return fn(a, b)
    return inner

@smart_divide
def divide(a, b):
    print(a/b)

divide(3,4)
divide(3,0)

# version7, a chained decorator example
def stars(func):
    def inner(*args, **kwargs):
        print("*" * 15)
        func(*args, **kwargs)
        print("*" * 15)

    return inner

def squre(func):
    def inner(*args, **kwargs):
        print("-" * 15)
        func(*args, **kwargs)
        print("-" * 15)

    return inner

@stars
@squre
def printLeon():
    print("I am leonDou")

printLeon()

temp = squre(stars(printLeon))
temp()


