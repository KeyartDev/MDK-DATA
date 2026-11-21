import requests
MODEL = 'qwen2.5:1.5b'
def ask(prompt, system='', temperature=0.7):
    r = requests.post('http://localhost:11434/api/generate', json={
        'model': MODEL, 'prompt': prompt, 'system': system, 'stream': False,
        'options': {'temperature': temperature}})
    return r.json()['response']
if __name__ == '__main__':
    print(ask('Объясни, что такое нейросеть, в двух предложениях.'))


question = 'Придумай название для робота-пылесоса.'
for _ in range(3):
    print(ask(question, temperature=1.5))



sys_prompt = 'Ты ассистент колледжа. Отвечай кратко, вежливо, только по-русски.'
question_4 = 'Как пройти в библиотеку?'
print("--- Без системного промпта ---")
print(ask(question_4))
print("--- С системным промптом ---")
print(ask(question_4, system=sys_prompt))

print("--- ДА/НЕТ ---")
sys_strict = 'Отвечай только ДА или НЕТ.'
print(ask('Этот колледж хороший?', system=sys_strict))
print(ask('Сколько будет 2+2?', system=sys_strict)) 


p = '''Определи категорию обращения одним словом.
Фраза: Где расписание на завтра? Категория: расписание
Фраза: Столовая сегодня работает? Категория: столовая
Фраза: Мне нужна справка для военкомата. Категория: документы
Фраза: Во сколько закрывается буфет? Категория:'''
print(ask(p, temperature=0))



import json
r = requests.post('http://localhost:11434/api/generate', json={
    'model': MODEL, 'stream': False, 'format': 'json',
    'prompt': 'Верни JSON с полями intent (одно слово) и answer (одно предложение) для запроса: Где столовая?'})
print(json.loads(r.json()['response']))

print("--- Запрос даты ---")
print(ask('Какое сегодня число и день недели?'))

print("\n--- Запрос расписания ---")
print(ask('Какое расписание занятий в моем колледже на завтра?'))