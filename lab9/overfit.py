from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import copy


rng = np.random.default_rng(0)
X = np.sort(rng.uniform(0, 1, 30))
y = np.sin(2*np.pi*X) + rng.normal(0, .3, 30)
Xv = np.sort(rng.uniform(0, 1, 30))
yv = np.sin(2*np.pi*Xv) + rng.normal(0, .3, 30)
loss_fn = nn.MSELoss()
for d in [1, 4, 15]:
    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(X[:, None], y)
    print(d, np.mean((m.predict(X[:, None]) - y)**2), np.mean((m.predict(Xv[:, None]) - yv)**2))


Xtr = torch.tensor(X[:, None], dtype=torch.float32)
ytr = torch.tensor(y, dtype=torch.float32)
Xva = torch.tensor(Xv[:, None], dtype=torch.float32)
yva = torch.tensor(yv, dtype=torch.float32)
def fit(model, opt, steps=2000):
    tr, va = [], []
    for i in range(steps):
        model.train(); opt.zero_grad()
        l = loss_fn(model(Xtr).squeeze(1), ytr)
        l.backward()
        opt.step()
        if i % 50 == 0:
            model.eval()
            with torch.no_grad(): tr.append(l.item())
            va.append(loss_fn(model(Xva).squeeze(1), yva).item())
    return tr, va
big = nn.Sequential(nn.Linear(1,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,1))
tr, va = fit(big, torch.optim.Adam(big.parameters(), 1e-2))
plt.plot(tr, label='train')
plt.plot(va, label='val')
plt.legend()
plt.show()


overfit_model = nn.Sequential(
    nn.Linear(1, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 1)
)
opt2 = torch.optim.Adam(overfit_model.parameters(), 1e-2)

tr2, va2 = [], []

best_val_loss2 = float('inf')
best_state_dict2 = None
patience_counter = 0
PATIENCE = 10
steps = 2000
eval_every = 50

for i in range(steps):
    overfit_model.train()
    opt2.zero_grad()
    l = loss_fn(overfit_model(Xtr).squeeze(1), ytr)
    l.backward()
    opt2.step()
    
    if i % eval_every == 0:
        overfit_model.eval()
        with torch.no_grad():
            val_l = loss_fn(overfit_model(Xva).squeeze(1), yva).item()
        tr2.append(l.item())
        va2.append(val_l)
        
        if val_l < best_val_loss2:
            best_val_loss2 = val_l
            best_state_dict2 = copy.deepcopy(overfit_model.state_dict()) 
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"Early stopping at step {i} (best val={best_val_loss2:.4f})")
            break

if best_state_dict2 is not None:
    overfit_model.load_state_dict(best_state_dict2)
    print(f"Best val loss: {best_val_loss2:.4f}")

plt.plot(tr2, label='train')
plt.plot(va2, label='val')
plt.axhline(best_val_loss2, color='red', linestyle='--', alpha=0.5, label=f'best val={best_val_loss2:.4f}')
plt.legend()
plt.show()


reg = nn.Sequential(
    nn.Linear(1, 128), nn.ReLU(), nn.Dropout(0.3), 
    nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3), 
    nn.Linear(128, 1)
)
tr2, va2 = fit(reg, torch.optim.Adam(reg.parameters(), 1e-2, weight_decay=1e-2))


def make_cls_data(n, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(0, 1, n))
    y_cont = np.sin(2*np.pi*X) + rng.normal(0, noise, n)
    y = (y_cont > 0).astype(np.float32)          
    return (torch.tensor(X[:, None], dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))

Xtr, ytr = make_cls_data(30,  0)
Xva, yva = make_cls_data(30,  1)
Xte, yte = make_cls_data(200, 2)
Xtr_big, ytr_big = make_cls_data(1000, 0)

loss_fn = nn.BCEWithLogitsLoss()

def make_model(dropout=0.0):
    L = [nn.Linear(1, 128), nn.ReLU()]
    if dropout: L.append(nn.Dropout(dropout))
    L += [nn.Linear(128, 128), nn.ReLU()]
    if dropout: L.append(nn.Dropout(dropout))
    L += [nn.Linear(128, 128), nn.ReLU()]
    if dropout: L.append(nn.Dropout(dropout))
    L += [nn.Linear(128, 1)]
    return nn.Sequential(*L)

@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    p = (torch.sigmoid(model(X).squeeze(1)) > 0.5).float()
    return (p == y).float().mean().item()

def fit(model, opt, Xtr, ytr, steps=2000, eval_every=50, patience=10, early=True):
    best_val, best_state, counter = float('inf'), None, 0
    for i in range(steps):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(Xtr).squeeze(1), ytr)
        loss.backward(); opt.step()
        if i % eval_every == 0:
            model.eval()
            with torch.no_grad():
                v = loss_fn(model(Xva).squeeze(1), yva).item()
            if v < best_val:
                best_val, best_state, counter = v, copy.deepcopy(model.state_dict()), 0
            else:
                counter += 1
                if early and counter >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val

def run(name, model, opt, X_tr, y_tr, **kw):
    bv = fit(model, opt, X_tr, y_tr, **kw)
    return name, bv, accuracy(model, Xva, yva), accuracy(model, Xte, yte)

rows = []
torch.manual_seed(0); m = make_model()
rows.append(run("Без регуляризации", m, torch.optim.Adam(m.parameters(), 1e-2), Xtr, ytr, early=False))

torch.manual_seed(0); m = make_model()
rows.append(run("Ранняя остановка", m, torch.optim.Adam(m.parameters(), 1e-2), Xtr, ytr, early=True))

torch.manual_seed(0); m = make_model()
rows.append(run("Weight decay (1e-2)", m, torch.optim.Adam(m.parameters(), 1e-2, weight_decay=1e-2), Xtr, ytr, early=False))

torch.manual_seed(0); m = make_model(dropout=0.3)
rows.append(run("Dropout 0.3", m, torch.optim.Adam(m.parameters(), 1e-2), Xtr, ytr, early=False))

torch.manual_seed(0); m = make_model()
rows.append(run("Больше данных (n=1000)", m, torch.optim.Adam(m.parameters(), 1e-2), Xtr_big, ytr_big, early=False))

print(f"{'Метод':<24} {'best val loss':>14} {'val acc':>10} {'test acc':>10}")
print("-" * 62)
for name, bl, va, ta in rows:
    print(f"{name:<24} {bl:>14.4f} {va:>10.3f} {ta:>10.3f}")