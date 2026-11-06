import math
import random

# Laying out the micrograd
class Value:
  def __init__(self, data, label="", _children=(), _op=""):
    self.label = label
    self.data = data
    self._prev = set(_children)
    self._op = _op
    self.grad = 0.0
    self._backward = lambda: None

  def __repr__(self):
    if (self.label):
      return f'{self.label} | {self.data}'
    else:
      return f"{self.data}"

  def __radd__(self, other):
    return self + other

  def __rmul__(self, other):
    return self * other

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, _children=(self, other), _op="+")

    def _backward():
      self.grad += out.grad
      other.grad += out.grad

    out._backward = _backward
    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, _children=(self, other), _op="*")

    def _backward():
      self.grad += out.grad * other.data
      other.grad += out.grad * self.data

    out._backward = _backward
    return out

  def __pow__(self, other):
    out = Value(self.data ** other, _children=(self, ), _op=f"**{other}")
    def _backward():
      self.grad += other*(self.data**(other-1)) * out.grad

    out._backward = _backward
    return out

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(t, _children = (self, ), _op="tanh")

    def _backward():
      self.grad += (1 - t**2) * out.grad

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


# A Basic neural Network
class Neuron:
  def __init__(self, nin):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))

  def __call__(self, x):
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = act.tanh()
    return out

  def parameter(self):
    return [self.b] + self.w

class Layer:
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]

  def __call__(self, x):
    outs = [neuron(x) for neuron in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for n in self.neurons for p in n.parameter()]

class MLP:
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    for layer in self.layers:
      out = layer(x)
    return out

  def parameters(self):
    return [p for n in self.layers for p in n.parameters()]

#Inputs
xs = [[2.0, 3.0, -1.0], [3.0, -5.0, 2.0]]
ys = [0.65, -0.95]

# Setting up the Multi-Layer-Perceptron
n = MLP(3, [4,4,1])

for k in range(1000000):
  # Getting results and calculating loss (Forward pass)
  ypred = [n(x) for x in xs]
  loss = sum((py-dy)**2 for py, dy in zip(ypred, ys))

  for p in n.parameters():
    p.grad = 0.0
  loss.backward()

  for param in n.parameters():
    param.data += -0.01 * param.grad

  print(k , loss)

print(ypred)