from sklearn.datasets import make_blobs
import numpy as np, matplotlib.pyplot as plt

def forward(X): return sig(X @ w + b)

def bce(p, y): return -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))


X, y = make_blobs(200, centers=2, cluster_std=1.2, random_state=1)
plt.scatter(X[:, 0], X[:, 1], c=y); plt.show()

sig = lambda z: 1 / (1 + np.exp(-z))
rng = np.random.default_rng(0)
w = rng.normal(0, 0.1, 2); b = 0.0
print(((forward(X) > .5) == y).mean())

print(bce(forward(X), y))

lr, hist = 0.1, []
for i in range(500):
    p = forward(X); 
    dz = p - y
    w -= lr * X.T @ dz / len(X); 
    b -= lr * dz.mean()
    hist.append(bce(p, y))
plt.plot(hist); plt.show()
print('accuracy:', ((forward(X) > .5) == y).mean())

xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200), np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
zz = forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, zz, levels=[0, .5, 1], alpha=.3); plt.scatter(X[:,0], X[:,1], c=y); plt.show()