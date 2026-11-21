import os
import sys
import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from PIL import Image

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

HERE = Path(__file__).resolve().parent
SENT_MODEL = 'cointegrated/rubert-tiny-sentiment-balanced'
TR_MODEL = 'Helsinki-NLP/opus-mt-ru-en'

PHOTOS = [Path(p) for p in sys.argv[1:]] or [HERE / 'photo.jpg']


def hf_size(repo_id):
    """Размер скачанных весов модели в кэше Hugging Face."""
    base = Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')) / 'hub'
    d = base / ('models--' + repo_id.replace('/', '--'))
    if not d.exists():
        return None
    return sum(f.stat().st_size for f in d.rglob('*') if f.is_file())


from transformers import pipeline

t0 = time.perf_counter()
clf = pipeline('sentiment-analysis', model=SENT_MODEL)
print('первый вызов (скачивание + загрузка):', round(time.perf_counter() - t0, 1), 'с')
for t in ['Отличный курс!', 'Ничего не понял', 'Нормально, но долго', 'Сложно, но интересно', 'Больше не приду']:
    print(t, clf(t))

from transformers import MarianMTModel, MarianTokenizer

try:
    tok = MarianTokenizer.from_pretrained(TR_MODEL)
except Exception:
    from huggingface_hub import constants
    import shutil
    mirror = Path(r'C:\Users\Public\mdk_cache') / ('models--' + TR_MODEL.replace('/', '--'))
    snap = next((Path(constants.HF_HUB_CACHE) / ('models--' + TR_MODEL.replace('/', '--')) / 'snapshots').iterdir())
    mirror.mkdir(parents=True, exist_ok=True)
    for name in ('source.spm', 'target.spm', 'vocab.json', 'tokenizer_config.json', 'special_tokens_map.json'):
        if (snap / name).exists():
            shutil.copy(snap / name, mirror / name)
    tok = MarianTokenizer.from_pretrained(str(mirror))
tr_model = MarianMTModel.from_pretrained(TR_MODEL)


def translate(text):
    out = tr_model.generate(**tok(text, return_tensors='pt'), max_new_tokens=64)
    return tok.decode(out[0], skip_special_tokens=True)


print(translate('Нейросеть — это цепочка матричных умножений.'))

from huggingface_hub import constants
print(constants.HF_HUB_CACHE)
for d in os.listdir(constants.HF_HUB_CACHE):
    print(' ', d)

import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

w = MobileNet_V3_Small_Weights.DEFAULT; m = mobilenet_v3_small(weights=w).eval(); tf = w.transforms()
img_tf = tf(Image.open(HERE / 'photo.jpg').convert('RGB'))
t0 = time.perf_counter()
with torch.no_grad():
    p = m(img_tf.unsqueeze(0)).softmax(1)[0]
mob_time = time.perf_counter() - t0
print()
print('MobileNetV3-small, топ-5 ImageNet:')
for v, i in zip(*p.topk(5)):
    print(f'  {w.meta["categories"][i]:25s} {v:.3f}')

fig, axes = plt.subplots(1, len(PHOTOS), figsize=(5 * len(PHOTOS), 4))
axes = axes if len(PHOTOS) > 1 else [axes]
for ax, path in zip(axes, PHOTOS):
    if not path.exists():
        print('нет файла', path, '— положи фото рядом со скриптом')
        ax.axis('off')
        continue
    im = Image.open(path).convert('RGB')
    with torch.no_grad():
        pp = m(tf(im).unsqueeze(0)).softmax(1)[0]
    top = pp.topk(3)
    ax.imshow(im); ax.axis('off')
    ax.set_title('\n'.join(f'{w.meta["categories"][i]}: {v:.2f}' for v, i in zip(*top)))
    print(path.name, '→', ', '.join(f'{w.meta["categories"][i]} {v:.2f}' for v, i in zip(*top)))
fig.suptitle('фото через MobileNetV3-small: топ-3 из ImageNet')
plt.tight_layout(); plt.savefig(HERE / 'photos.png', bbox_inches='tight'); plt.show()

mnist = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))
n_mnist = sum(p.numel() for p in mnist.parameters())
n_mob = sum(p.numel() for p in m.parameters())
n_sent = clf.model.num_parameters()
n_tr = tr_model.num_parameters()
print()
print(f'MNIST-сеть: {n_mnist:,} параметров, 10 классов')
print(f'MobileNetV3-small: {n_mob:,} параметров, 1000 классов')
print(f'{SENT_MODEL}: {n_sent:,} параметров')
print(f'{TR_MODEL}: {n_tr:,} параметров')

t0 = time.perf_counter(); clf('Нормально, но долго'); sent_time = time.perf_counter() - t0
t0 = time.perf_counter(); translate('Нейросеть — это цепочка матричных умножений.'); tr_time = time.perf_counter() - t0


def mb(repo_id):
    s = hf_size(repo_id)
    return f'{s / 1e6:.1f} МБ' if s else '—'


rows = [
    ('тональность RU', SENT_MODEL, mb(SENT_MODEL), sent_time, 'ирония, смешанные оценки'),
    ('перевод RU→EN', TR_MODEL, mb(TR_MODEL), tr_time, 'длинные термины'),
    ('фото ImageNet', 'mobilenet_v3_small', f'{n_mob * 4 / 1e6:.1f} МБ (fp32)', mob_time, 'не-ImageNet классы'),
]
print()
print(f'{"задача":<17} {"модель":<40} {"на диске":<18} {"ответ, с":>8}  где ошибается')
for task, name, size, sec, err in rows:
    print(f'{task:<17} {name:<40} {size:<18} {sec:>8.2f}  {err}')
