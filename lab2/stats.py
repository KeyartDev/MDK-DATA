import math as mth

def mean(lst):
    total = 0
    for v in lst:
        total += v
    return total / len(lst)
 
def minmax(lst):
    lo, hi = lst[0], lst[0]
    for v in lst:
        if v < lo: lo = v
        if v > hi: hi = v
    return lo, hi

def median(lst):
    l = len(lst)
    m = l/2
    if l % 2 == 0:
        return float(mean([lst[int(m-1)], lst[int(m)]]))
    else:
        return lst[mth.floor(m)]
    
with open('data/titanic.csv', encoding='utf-8') as f:
    header = f.readline().strip().split(',')
    rows = [line.strip().split(',') for line in f]

cols = {h: [] for h in header}
for r in rows:
    for h, v in zip(header, r):
        if v == '':
            continue
        try:
            cols[h].append(float(v))
        except ValueError:
            cols[h].append(v)

for h, vals in cols.items():
    if not vals:
        continue
    if isinstance(vals[0], float):
        lo, hi = minmax(vals)
        print(f'{h:12s} n={len(vals):4d} mean={mean(vals):8.2f} min={lo:6.1f} max={hi:6.1f}')
    else:
        print(f'{h:12s} n={len(vals):4d} unique={len(set(vals))}')

print(median(cols["age"][:10]))