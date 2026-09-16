import requests
r = requests.post('http://localhost:11434/api/generate', json={
    'model': 'qwen2.5:0.5b',
    'prompt': 'Что такое нейросеть? Ответь в двух предложениях по-русски.',
    'stream': False})
print(r.json()['response'])
