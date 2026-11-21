text = open('C:/Users/user/Desktop/3ISIP-24-1/MDK-DATA/lab14/docs.txt', encoding='utf-8').read()
chunks = [text[i:i+400] for i in range(0, len(text), 350)]
print(len(chunks)); print(chunks[0])


from transformers import logging
logging.set_verbosity_error()

from sentence_transformers import SentenceTransformer
emb = SentenceTransformer('cointegrated/rubert-tiny2')
V = emb.encode(chunks, normalize_embeddings=True)
print(V.shape)


import numpy as np

phrases = [
    "Кошка спит",         
    "Кошка ест",             
    "Теория вероятности и математической статистики",  
]

V_test = emb.encode(phrases, normalize_embeddings=True)  
S = V_test @ V_test.T

np.set_printoptions(precision=3, suppress=True)
print(S)

print(f"\nБлизость 1 и 2: {S[0,1]:.3f} (должно быть > 0.7)")
print(f"Близость 1 и 3: {S[0,2]:.3f} (должно быть < 0.4)")
print(f"Близость 2 и 3: {S[1,2]:.3f} (должно быть < 0.4)")


import numpy as np
def search(q, k=3):
    qv = emb.encode([q], normalize_embeddings=True)[0]
    scores = V @ qv
    idx = np.argsort(-scores)[:k]
    return [(float(scores[i]), chunks[i]) for i in idx]
for s, c in search('Сколько часов длится конкурс?'): print(round(s, 3), c[:100])
for s, c in search('Сколько минут в часе?'): print(round(s, 3), c[:100])
for s, c in search('Какая погода?'): print(round(s, 3), c[:100])
for s, c in search('Сколько пар у меня сегодня?'): print(round(s, 3), c[:100])
for s, c in search('Сколько цифр в чиле 3829489342?'): print(round(s, 3), c[:100])


import requests
MODEL = 'qwen2.5:1.5b'
def ask(prompt, system='', temperature=0.7):
    r = requests.post('http://localhost:11434/api/generate', json={
        'model': MODEL, 'prompt': prompt, 'system': system, 'stream': False,
        'options': {'temperature': temperature}})
    return r.json()['response']

def rag(q, k=2):
    ctx = '\n---\n'.join(c for _, c in search(q, k))
    p = f'Ответь на вопрос только по контексту. Если ответа в контексте нет, скажи: не знаю.\n\nКонтекст:\n{ctx}\n\nВопрос: {q}\nОтвет:'
    return ask(p, temperature=0)
print(rag('Сколько часов длится конкурс?'))
#print(rag('Какая столица Австралии?'))  # ожидаем: не знаю


test_data = [
    ("В чем заключается актуальность проекта?", 0),          
    ("Что является предметом исследования?", 0),              
    ("Какая цель создания информационной системы?", 0),      
    ("Что такое информационная система?", 1),                
    ("Какие компоненты включает в себя ИС?", 1),              
    ("Из каких двух частей состоит проект?", 2),             
    ("Какая программа использовалась для создания дизайна?", 3),
    ("Какие языки использовались для вёрстки?", 4),           
    ("Какую роль играет JavaScript в проекте?", 9),          
    ("Какие роли были созданы в конфигурации 1С?", 8)]        

def recall_at_k(q, correct_idx, k=3):
    qv = emb.encode([q], normalize_embeddings=True)[0]
    scores = V @ qv
    top_k_idx = np.argsort(-scores)[:k]
    return correct_idx in top_k_idx

hits = 0
print(f"{'Вопрос':<45} | {'Чанк':<5} | {'В топ-3?':<10}")
print("-" * 75)

for q, correct_idx in test_data:
    is_hit = recall_at_k(q, correct_idx, k=3)
    hit = "Да" if is_hit else "Нет"
    if is_hit: hits += 1
    print(f"{q[:45]:<45} | {correct_idx:<5} | {hit:<10}")

print("-" * 75)
print(f"Recall@3: {hits}/{len(test_data)} = {hits/len(test_data)*100:.0f}%")

print("\nПроверка LLM (0/1):")
for q, correct_idx in test_data:
    ans = rag(q)
    print(f"Вопрос: {q}")
    print(f"Ответ LLM: {ans}\n")
