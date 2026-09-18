import math as mth, numpy as np

def mean(lst):      #Среднее значение
    total = 0
    for v in lst:
        total += v
    return total / len(lst)
 
def minmax(lst):    #Минимальное и максимальное
    lo, hi = lst[0], lst[0]
    for v in lst:
        if v < lo: lo = v
        if v > hi: hi = v
    return lo, hi

def median(lst):    #Медиана
    l = len(lst)
    m = l/2
    if l % 2 == 0:
        return float(mean([lst[int(m-1)], lst[int(m)]]))
    else:
        return lst[mth.floor(m)]
    
with open('data/titanic.csv', encoding='utf-8') as f:   #получение столбцов и строк
    header = f.readline().strip().split(',')
    rows = [line.strip().split(',') for line in f]

cols = {h: [] for h in header}  #Генерация пустого словаря(В качестве ключей - названия столбцов)

for r in rows:                  #Заполнение словаря
    for h, v in zip(header, r):
        if v == '':
            continue
        try:
            cols[h].append(float(v))
        except ValueError:
            cols[h].append(v)

for h, vals in cols.items():    #Подсчёт и вывод статистики
    if not vals:
        continue
    if isinstance(vals[0], float):
        lo, hi = minmax(vals)
        print(f'{h:12s} n={len(vals):4d} mean={mean(vals):8.2f} min={lo:6.1f} max={hi:6.1f}')
    else:
        print(f'{h:12s} n={len(vals):4d} unique={len(set(vals))}')

print('Медиана: ', median(cols["age"][:10]))
#-------------------------LESSON 3 COMPARISON----------------------------
np_arr = np.array(cols['age'])
print('------------------------AGE STATISTICS WITH NUMPY----------------------------------')
print(f'{'Age':12s} mean={np_arr.mean():8.2f} min={np_arr.min():6.1f} max={np_arr.max():6.1f} median={np.median(cols['age'][:10]):6.1f}')