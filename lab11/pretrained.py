from transformers import pipeline
clf = pipeline('sentiment-analysis', model='cointegrated/rubert-tiny-sentiment-balanced')
for t in ['Отличный курс!', 'Ничего не понял', 'Нормально, но долго']:
    print(t, clf(t))


tr = pipeline('text-generation', model='Helsinki-NLP/opus-mt-ru-en')
print(tr('Нейросеть — это цепочка матричных умножений.')[0]['translation_text'])


import os
from huggingface_hub import constants
print(constants.HF_HUB_CACHE)
for d in os.listdir(constants.HF_HUB_CACHE): print(d)


import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

w = MobileNet_V3_Small_Weights.DEFAULT 
m = mobilenet_v3_small(weights=w).eval()
tf = w.transforms()
img = Image.open('photo.jpg').convert('RGB')
with torch.no_grad(): p = m(tf(img).unsqueeze(0)).softmax(1)[0]
for v, i in zip(*p.topk(5)): print(f'{w.meta["categories"][i]:25s} {v:.3f}')


import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from transformers import AutoModel

mnist_model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3),   
    nn.ReLU(),
    nn.MaxPool2d(2),                  
    nn.Flatten(),
    nn.Linear(8 * 13 * 13, 10)        
)
mnist_params = sum(p.numel() for p in mnist_model.parameters())

mobilenet_params = sum(p.numel() for p in m.parameters())

rubert_model = AutoModel.from_pretrained('cointegrated/rubert-tiny-sentiment-balanced')
rubert_params = sum(p.numel() for p in rubert_model.parameters())

print(f"MNIST (простая CNN): ~{mnist_params/1000:.1f} тыс. параметров")
print(f"MobileNetV3 Small:   {mobilenet_params/1_000_000:.2f} млн параметров")
print(f"RuBERT Tiny:         {rubert_params/1_000_000:.2f} млн параметров")


import time

def get_model_size_mb(model):
    """Размер весов модели в МБ (float32)."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

input_tensor = tf(img).unsqueeze(0)
start = time.time()
with torch.no_grad():
    for _ in range(10):
        _ = m(input_tensor)
mobilenet_time = (time.time() - start) / 10

start = time.time()
for _ in range(10):
    _ = clf(["Тестовое предложение для замера скорости."])
rubert_time = (time.time() - start) / 10

print(f"{'Модель':<15} | {'Задача':<22} | {'Размер (МБ)':<11} | {'Время (CPU)':<11} | {'Где ошиблась'}")
print("-" * 110)
print(f"{'MobileNetV3':<15} | {'Классификация фото':<22} | {get_model_size_mb(m):<11.2f} | {mobilenet_time:<11.4f} | "
      f"Путает похожие породы, теряет мелкие детали")
print(f"{'RuBERT Tiny':<15} | {'Сентимент анализ':<22} | {get_model_size_mb(rubert_model):<11.2f} | {rubert_time:<11.4f} | "
      f"Не понимает сарказм и сложные отрицания")
print(f"{'MNIST CNN':<15} | {'Классификация цифр':<22} | {get_model_size_mb(mnist_model):<11.2f} | {'~0.0005':<11} | "
      f"Путает 4/9 и 3/8 при плохом почерке")