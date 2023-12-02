import utils as U
import math as m

class Value:
    def __init__(self, data, _children = (), _op = "", label=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, (self, other),  "+")
        return out
    
    def __radd__(self, other):
        return self.__add__(self, other)
    
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        return out
    
    def __rmul__(self, other):
        return self.__mul__(self, other)
    
    def exp(self):
        e = m.exp(self.data)
        return Value(e, (self, ), "exp")
    
    def tanh(self):
        t = (m.exp(2*self.data) + 1) / (m.exp(2*self.data) - 1)
        return Value(t, (self, ), "tanh")
            

    
    def visualize(self):
        return U.visualize_micro_grad(self)
    

