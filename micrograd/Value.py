import utils as U
import math as m

class Value:
    def __init__(self, data, _children = (), _op = "", label=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

        self.grad = 0.0
        self._backward = None

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, (self, other),  "+")

        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad

        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self.__add__(self, other)
    
    def __mul__(self, other): 
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad

        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self.__mul__(self, other)
    
    def exp(self):
        e = m.exp(self.data)
        out = Value(e, (self, ), "exp")

        def _backward():
            self.grad = e * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        t = (m.exp(2*self.data) + 1) / (m.exp(2*self.data) - 1)
        out = Value(t, (self, ), "tanh")

        def _backward():
            self.grad = (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        def build_top():    

            
    def visualize(self):
        return U.visualize_micro_grad(self)
    

