import numpy as np, matplotlib.pyplot as plt

def loss(w, b): return np.mean((w * X + b - y) ** 2)

def grads(w, b):
    p = w * X + b
    return np.mean(2 * (p - y) * X), np.mean(2 * (p - y))

f = lambda x: (x - 3) ** 2
df = lambda x: 2 * (x - 3)
x, lr, path = 10.0, 0.1, []
for i in range(30):
    path.append(x); x = x - lr * df(x)

print(x)
xs = np.linspace(-2, 12, 100); plt.plot(xs, f(xs)); plt.scatter(path, [f(p) for p in path], c='r'); plt.show()


rng = np.random.default_rng(0)
X = rng.uniform(0, 5, 50)
y = 2 * X + 1 + rng.normal(0, 0.5, 50)
plt.scatter(X, y); plt.show()

w, b, lr, hist = 0.0, 0.0, 0.05, []

for i in range(200):
    dw, db = grads(w, b); w -= lr * dw; b -= lr * db; hist.append(loss(w, b))

print(w, b)
plt.plot(hist); plt.show()
plt.scatter(X, y); plt.plot(X, w * X + b, 'r'); plt.show()