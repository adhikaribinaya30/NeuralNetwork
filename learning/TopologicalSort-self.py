
def backward(v):
  topo = []
  visited = set()

  def build(n):
    if n not in visited:
      visited.add(n)
      for child in n._prev:
        build(child)
      topo.append(n)

  build(v)

# Doesn't work here since we don't have object with ._prev
