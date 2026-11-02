from sklearn.datasets import make_blobs
import numpy as np, matplotlib.pyplot as plt

def forward(X): return sig(X @ w + b)

def bce(p, y): return -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))


X, y = make_blobs(200, centers=2, cluster_std=1.2, random_state=1)

sig = lambda z: 1 / (1 + np.exp(-z))
rng = np.random.default_rng(0)
w = rng.normal(0, 0.1, 2); b = 0.0
print(((forward(X) > .5) == y).mean())

print(bce(forward(X), y))

lr, hist = 0.1, []
for i in range(1500):
    p = forward(X); 
    dz = p - y
    w -= lr * X.T @ dz / len(X); 
    b -= lr * dz.mean()
    hist.append(bce(p, y))
print('accuracy:', ((forward(X) > .5) == y).mean())

xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200), np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
zz = forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

import torch

Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
w = torch.zeros(2, requires_grad=True); b = torch.zeros(1, requires_grad=True)
p = torch.sigmoid(Xt @ w + b)
loss = torch.nn.functional.binary_cross_entropy(p, yt)
loss.backward(); print(w.grad, b.grad)

w = torch.zeros(2, requires_grad=True); b = torch.zeros(1, requires_grad=True)
for i in range(500):
    p = torch.sigmoid(Xt @ w + b)
    loss = torch.nn.functional.binary_cross_entropy(p, yt)
    loss.backward()
with torch.no_grad():
    w -= 0.1 * w.grad; b -= 0.1 * b.grad
    w.grad.zero_(); b.grad.zero_()
print(loss.item())

import torch.nn as nn
import numpy as np, matplotlib.pyplot as plt

model = nn.Linear(2, 1)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for i in range(500):
    opt.zero_grad()
    loss = loss_fn(model(Xt).squeeze(1), yt)
    loss.backward()
    opt.step()
print(loss.item(), model.weight, model.bias)

def train(make_opt, steps=300):
    m = nn.Linear(2, 1); opt = make_opt(m.parameters()); h = []
    for i in range(steps):
        opt.zero_grad(); l = loss_fn(m(Xt).squeeze(1), yt); l.backward(); opt.step(); h.append(l.item())
    return h

plt.plot(train(lambda p: torch.optim.SGD(p, lr=0.1)), label='SGD')
plt.plot(train(lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)), label='SGD+momentum')
plt.plot(train(lambda p: torch.optim.Adam(p, lr=0.01)), label='Adam')
plt.legend(); plt.show()

from sklearn.datasets import make_moons

Xm, ym = make_moons(300, noise=.2, random_state=0)
Xt = torch.tensor(Xm, dtype=torch.float32); yt = torch.tensor(ym, dtype=torch.float32)
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for i in range(1000):
    opt.zero_grad(); l = loss_fn(model(Xt).squeeze(1), yt); l.backward(); opt.step()
print('accuracy:', ((model(Xt).squeeze(1) > 0).float() == yt).float().mean().item())

