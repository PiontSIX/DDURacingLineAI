import math
import numpy as np
import random


alphab = 0.02
db = 1
alphaw = 0.02
dw = 1
alphaac = 0.2
alpharc = 0.01


def act(x):
  return np.log(1+x)


class agent:
  def __init__(self, inn, outn):
    self.inn = inn
    self.outn = outn
    self.totn = inn+outn
    self.w = [[] for i in range(self.totn)] #weights
    self.b = [0 for i in range(self.totn)] #biases
    self.a = [0 for i in range(self.totn)] #neuron activation
    self.p = [False for i in range(self.totn)] #neuron processed?
    self.d = [[] for i in range(self.totn)] #dependent neurons
    self.i = list(range(inn))


  def node(self, x): #processes neuron if dependent neurons are processed, activates connected neurons recursively
    if self.p[x]:
      return
    for i in self.d[x]:
      if not self.p[i]:
        return
    self.p[x] = True
    if x >= self.inn+self.outn:
      self.a[x] += self.b[x]
      self.a[x] = act(self.a[x])
    for i in self.w[x]:
      self.a[i[0]] += self.a[x]*i[1]
      self.node(i[0])


  def model(self, inv): #processes the nnet
    self.a = [0 for i in range(self.totn)]
    self.p = [False for i in range(self.totn)]
    for i in range(self.inn):
      self.a[i] = inv[i]
      self.node(i)
    ans = []
    for i in range(self.outn):
      ans.append(self.a[i+self.inn])
    return ans


  def mutate(self): #self explanatory
    for i in self.b: #biases
      if np.random.rand() < alphab:
        i += np.random.rand*2*db-db
    for i in range(len(self.w)): #weights and removing connections
      for j in self.w[i]:
        if np.random.rand() < alphaw:
          j[1] += np.random.rand()*2*dw-dw
        if np.random.rand() < alpharc:
          self.d[j[0]].remove(i)
          self.w[i].remove(j)
    if np.random.rand() < alphaac: #add connections
      a = random.randint(0, self.totn-self.outn-1)
      if a >= self.inn:
        a += self.outn
      b = random.randint(self.inn, self.totn-1)
      self.w[a].append((b, np.random.rand()*2*dw-dw))
      self.d[b].append(a)
