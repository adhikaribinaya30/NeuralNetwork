import math

class Value:
  def __init__(self, data, label="", _children=(), _op=""):
    self.data = data
    self.label = label
    self._op = _op
    self._backward = lambda: None
    self.grad = 0.0
    self._prev = set(_children)

  def __repr__(self):
    return f"{self.label}: {self.data}"

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, _children=(self, other), _op="+")

    def _backward():
      self.grad += out.grad 
      other.grad += out.grad
    out._backward = _backward
    return out

  def __rmul__(self, other):
    return self * other

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, _children=(self, other), _op="*")

    def _backward():
      self.grad += out.grad*other.data
      other.grad += out.grad*self.data

    out._backward = _backward
    return out

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) -1)/(math.exp(2*x) +1)
    out = Value(t, _children=(self, ), _op="tanh")

    def _backward():
      self.grad += (1 - out.data**2) * out.grad

    out._backward = _backward
    return out

  def backward(self):
    topo = []
    visited = set()

    def build(n):
      if n not in visited:
        visited.add(n)
        for child in n._prev:
          build(child)
        topo.append(n)
    build(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()


a = Value(2.0, "a")
b = Value(3.0, "b")

c = Value(-5.0, "c")

d = a*b
d.label = "d"

e = d + c
e.label = "e"

e.backward()

print(f"{a.label}: {a.grad} {b.label}: {b.grad} {c.label}: {c.grad} {d.label}: {d.grad} {e.label}: {e.grad}")
