import numpy as np

a = np.array([3, 1, 4, 1, 5])
#print(a * 2, a + 10, a ** 2)
#print(a.mean(), a.min(), a.max(), a.sum())

M = np.arange(12).reshape(3, 4) 
# print(M, M.shape)
# print('строка 0:', M[0]); print('столбец 1:', M[:, 1]); print('элемент:', M[2, 3])
# print(M.T.shape) 

# print(M.mean(axis=0))  # 4 числа
# print(M.mean(axis=1))  # 3 числа

A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
w = np.array([[1], [0], [-1]])          # (3, 1)
# print(A @ w)                            # (2, 1)
# print(A + np.array([10, 20, 30]))

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5))
w = np.array([0.5, -1.0, 2.0, 0.0, 1.5]); b = 0.3
y = X @ w + b
print(X.shape, w.shape, y.shape, y[:5])
#-----------------------ДОМАШНЕЕ ПОВТОРЕНИЕ-----------------------
X = rng.normal(size=(1000, 8)) # Изменяем размер матрицы с (100, 5) на (1000, 8)
w = np.array([0.5, -1.0, 2.0, 0.0, 1.5, -1.5, 1.0, -2.0]); b = 0.3 # Изменяем размер столбчатой матрицы с (5,) на (8,), 
y = X @ w + b                                                      # т.к. в ином случае получим ошибку из-за 
print(X.shape, w.shape, y.shape, y[:5])                            # несоответствия правилам умножения матриц
                         
