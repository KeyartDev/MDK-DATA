import numpy as np, matplotlib.pyplot as plt

rng1 = np.random.default_rng(0)
X = rng1.uniform(0, 5, 50)
y = 3 * X**2 - 2*X + 1 + rng1.normal(0, 0.5, 50)
plt.scatter(X, y); plt.title("Домашнее повторение"); plt.show()
print(X)
print(y)

def loss1(w, b, c): 
    y_pred = w*X**2+b*X+c
    return np.mean((y_pred - y)**2)

def grads1(w, b, c):
    y_pred = w*(X**2)+b*X+c 
    err = y_pred - y
    dw = np.mean(2*err*(X**2))
    db = np.mean(2*err*X)
    dc = np.mean(2*err)
    return dw, db, dc

w, b, c, lr, hist = 0.0, 0.0, 0.0, 0.0001, []

for i in range(2000):
    dw, db, dc = grads1(w, b, c)
    w -= lr * dw 
    b -= lr * db
    c -= lr * dc
    hist.append(loss1(w, b, c))

print(w, b, c)

plt.plot(hist); plt.show()

plt.scatter(X, y, alpha=0.6)
xs = np.linspace(0, 5, 100)
plt.plot(xs, w*xs**2+b*xs+c,  'r')
plt.plot(xs, 3*xs**2-2*xs+1,  'g--')
plt.show() 
print(X)
print(y)