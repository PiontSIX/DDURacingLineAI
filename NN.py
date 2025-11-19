import math
import numpy as np
import random
import copy

alphab = 0#0.02
db = 1
alphaw = 0.02
dw = 1
alphaac = 0.2
alpharc = 0#0.01
alphaan = 0#0.2
gensize = 10
keepsize = 3


def act(x):
  return np.log(1+x)


class agent:
  def __init__(self, inn, outn, totn = 0, w = 0, b = 0, a = 0, p = 0, d = 0, i = 0): #do not touch. interacts with cursed deepcopy.
    self.inn = inn
    self.outn = outn
    self.totn = totn
    if totn == 0:
      self.totn = inn+outn
    self.w = w
    if w == 0:
      self.w = [[] for i in range(self.totn)] #weights
    self.b = b
    if b == 0:
      self.b = [0 for i in range(self.totn)] #biases
    self.a = a
    if a == 0:
      self.a = [0 for i in range(self.totn)] #neuron activation
    self.p = p
    if p == 0:
      self.p = [False for i in range(self.totn)] #neuron processed?
    self.d = d
    if d == 0:
      self.d = [[] for i in range(self.totn)] #dependent neurons
    self.i = i
    if i == 0:
      self.i = list(range(self.totn)) #initialized nodes


  def __deepcopy__(self, memo): #cursed. do not touch.
    id_self = id(self)
    _copy = memo.get(id_self)
    #if _copy is None: #(index next line)
    _copy = type(self)(copy.deepcopy(self.inn, memo), copy.deepcopy(self.outn, memo), copy.deepcopy(self.totn, memo), copy.deepcopy(self.w, memo), copy.deepcopy(self.b, memo), copy.deepcopy(self.a, memo), copy.deepcopy(self.p, memo), copy.deepcopy(self.d, memo), copy.deepcopy(self.i, memo))
      #memo[id_self] = _copy
    return _copy


  def node(self, x): #processes neuron if dependent neurons are processed, activates connected neurons recursively
    if self.p[x]:
      return
    for i in self.d[x]:
      if not self.p[i]:
        return
    self.p[x] = True
    self.a[x] += self.b[x]
    if x >= self.inn+self.outn:
      self.a[x] = act(self.a[x])
    for i in self.w[x]:
      self.a[i[0]] += self.a[x]*i[1]
      self.node(i[0])


  def model(self, inv): #processes the nnet
    self.a = [0 for i in range(self.totn)]
    self.p = [False for i in range(self.totn)]
    for i in range(self.inn):
      self.a[i] = inv[i]
    for i in self.i:
      self.node(i)
    ans = []
    for i in range(self.outn):
      ans.append(self.a[i+self.inn])
    return ans


  def mutate(self): #self explanatory
    for i in range(self.inn, self.totn): #biases
      if np.random.rand() < alphab:
        self.b[i] += np.random.rand()*2*db-db
    for i in range(len(self.w)): #weights and removing connections
      for j in range(len(self.w[i])):
        if np.random.rand() < alphaw:
          self.w[i][j][1] += np.random.rand()*2*dw-dw
        if np.random.rand() < alpharc:
          self.d[self.w[i][j][0]].remove(i)
          if self.d[self.w[i][j][0]] == []:
            self.i.append(self.w[i][j][0])
          self.w[i].pop(j)
    if np.random.rand() < alphaac and self.totn-self.outn-1 >= 0: #add connections
      a = random.randint(0, self.totn-self.outn-1)
      if a >= self.inn:
        a += self.outn
      b = random.randint(self.inn, self.totn-1)
      c = np.random.rand()*2*dw-dw
      self.w[a].append([b, c])
      if self.d[b] == []:
        self.i.remove(b)
      self.d[b].append(a)
      self.model([0 for i in range(self.inn)])
      for i in range(self.inn, self.inn + self.outn):
        if not self.p[i]:
          self.d[b].remove(a)
          if self.d[b] == []:
            self.i.append(b)
          self.w[a].remove([b, c])
          break
    if np.random.rand() < alphaan and self.totn-self.outn-1 >= 0: #add nodes
      a = random.randint(0, self.totn-self.outn-1)
      if a >= self.inn:
        a += self.outn
      b = random.randint(self.inn, self.totn-1)
      c1 = np.random.rand()*2*dw-dw
      s1 = np.random.rand()*2*db-db
      c2 = np.random.rand()*2*dw-dw
      self.totn += 1
      self.b.append(s1)
      self.w.append([])
      self.d.append([])
      self.w[a].append([self.totn-1, c1])
      self.d[self.totn-1].append(a)
      self.w[self.totn-1].append([b, c2])
      if self.d[b] == []:
        self.i.remove(b)
      self.d[b].append(a)
      self.model([0 for i in range(self.inn)])
      for i in range(self.inn, self.inn + self.outn):
        if not self.p[i]:
          self.d[b].remove(self.totn-1)
          if self.d[b] == []:
            self.i.append(b)
          self.w[a].remove([self.totn-1, c1])
          self.w.pop()
          self.d.pop()
          self.b.pop()
          self.totn -= 1
          break
gn = 0
gs = [[[0, i, agent(1, 1)] for i in range(gensize)]]
ps = []
while True:
  for i in range(gensize):
    gs[gn][i][0] = gs[gn][i][2].model([1])[0] - gs[gn][i][2].model([-1])[0]
  gs[gn] = sorted(gs[gn], reverse=True)
  print(gs[gn])
  ps.append([])
  for i in range(gensize):
    ps[gn].append(gs[gn][i][0])
  print(ps[gn])
  if(input("END? ") == "y"):
    break
  gs.append(gs[gn])
  for i in range(keepsize, gensize):
    gs[gn+1][i] = [0, gs[gn][i][1], copy.deepcopy(gs[gn][random.randint(0, gensize-1)][2])]
    gs[gn+1][i][2].mutate()
  gn += 1
