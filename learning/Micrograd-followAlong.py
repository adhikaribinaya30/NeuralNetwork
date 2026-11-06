import math
import random

class Value:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self.label = label
    self._prev = set(_children)
    self._op = _op

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

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, _children=(self, other), _op="*")

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad 

    out._backward = _backward
    return out

  def __rmul__(self, other):
    return self * other

  def __pow__(self, other):
    assert isinstance(other, (int, float))
    out = Value(self.data ** other, (self, ), f"**{other}")

    def _backward():
      self.grad += other * self.data ** (other-1) * out.grad
    out._backward = _backward
    return out


  def __truediv__(self, other):
    return self * other**-1

  def exp(self):
    out = Value(math.exp(self.data), (self, ), 'exp')

    def _backward():
      self.grad += out.data * out.grad

    out._backward = _backward
    return out

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) -1)/(math.exp(2*x) +1)
    out = Value(t, (self, ), 'tanh')

    def _backward():
      self.grad += (1 - out.data**2) * out.grad

    out._backward = _backward
    return out

  # this is the function that applies back propagation at the end of the forward pass to set gradients for each inputs and weights using topological sort 
  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()

def getNodes(root):

  nodes = []
  def build(n):
    if n not in nodes:
      nodes.append((n, n.label))
      for child in n._prev:
        build(child)
        

  build(root)
  return nodes

#Setting values for all inputs bias and weights

x1, x2 = Value(2.0, label="x1"), Value(0.0, label="x2")

w1,w2 = Value(-3.0, label="w1"), Value(1.0, label="w2")

b = Value(6.884845468, label="b")

#Calculating the Output

x1w1 = x1 * w1
x1w1.label = "x1w1"
x2w2 = x2 * w2
x2w2.label = "x2w2"

x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1w1x2w2"

n = x1w1x2w2 + b
n.label = "n"

# o = n.tanh()
e = (2*n).exp()
e.label = "e"
o = (e - 1)/(e + 1)
o.label = "o"

# Back Propagation

o.backward()

#Getting all the nodes with their values and gradients

# nodes = getNodes(o)
# for node in nodes:
#   print(node, round(node[0].grad, 2))

#Making a neuron
class Neuron:
  def __init__(self, nin):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))

  def __call__(self, x):
    act = sum((xi*wi for xi, wi in zip(self.w, x)), self.b)
    out = act.tanh()
    return out

class Layer:
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

class MLP:
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      out = layer(x)
    return out

x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
print(n(x))

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 0.5],
  [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]
ypred = [n(x) for x in xs]
print(ypred)