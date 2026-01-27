import types

class ClassA():
    def __init__(self):
        self.b = ClassB()
    
    def forward(self, x):
        return self.b.forward(x) *2

class ClassB():
    def __init__(self):
        self.c = [ClassC(), ClassC()]
    
    def forward(self, x):
        return (self.c[0].forward(x) + self.c[1].forward(x)) *3

class ClassC():
    def __init__(self):
        self.myvar = 3
    
    def forward(self, x):
        return x *4


def fw_c(self, x):
    return x + 1

def fw_b(self, x):
    return (self.c[0].forward(x) + self.c[1].forward(x))+ 2

def fw_a(self, x):
    return self.b.forward(x) + 3


y = ClassA()
print(y.forward(1))
y_c = y.b.c
y_b = y.b

for d in y_c:
    d.forward = types.MethodType(fw_c, d)
y_b.forward = types.MethodType(fw_b, y_b)
y.forward = types.MethodType(fw_a, y)
print(y.forward(1))



