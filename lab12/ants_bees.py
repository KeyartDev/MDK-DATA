from torchvision import datasets, transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])
tr = datasets.ImageFolder('C:/Users/user/Desktop/3ISIP-24-1/MDK-DATA/lab12/train', tf)
va = datasets.ImageFolder('C:/Users/user/Desktop/3ISIP-24-1/MDK-DATA/lab12/val', tf)
print(tr.classes, len(tr), len(va))


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T

tf_96 = T.Compose([
    T.Resize((96, 96)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
tr_96 = datasets.ImageFolder('C:/Users/user/Desktop/3ISIP-24-1/MDK-DATA/lab12/train', tf_96)
va_96 = datasets.ImageFolder('C:/Users/user/Desktop/3ISIP-24-1/MDK-DATA/lab12/val', tf_96)

tr_loader = DataLoader(tr_96, batch_size=32, shuffle=True)
va_loader = DataLoader(va_96, batch_size=32, shuffle=False)

cnn_model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2), # 96x96 -> 48x48
    
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2), 
    
    nn.Flatten(),
    nn.Linear(32 * 24 * 24, 2) 
)

opt_cnn = torch.optim.Adam(cnn_model.parameters(), lr=1e-3)
loss_fn_cnn = nn.CrossEntropyLoss()

import time
print("Начинаем обучение CNN...")
for epoch in range(5):
    start_time = time.time()
    cnn_model.train()
    total_loss = 0
    for x, y in tr_loader:
        opt_cnn.zero_grad()
        pred = cnn_model(x)
        loss = loss_fn_cnn(pred, y)
        loss.backward()
        opt_cnn.step()
        total_loss += loss.item()
    
    cnn_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in va_loader:
            pred = cnn_model(x)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
            
    val_acc = correct / total
    epoch_time = time.time() - start_time
    print(f"Эпоха {epoch+1}/5 | Время: {epoch_time:.1f}с | Val accuracy: {val_acc:.3f}")

print(f"Параметров в CNN: {sum(p.numel() for p in cnn_model.parameters())}")


import torch, torch.nn as nn
from torch.utils.data import DataLoader

body = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
body.classifier = nn.Identity()
body.eval()
def extract(ds):
    xs, ys = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, 32): 
            xs.append(body(x)) 
            ys.append(y)
    return torch.cat(xs), torch.cat(ys) 

Ftr, ytr = extract(tr); Fva, yva = extract(va)
Ftr, ytr = extract(tr); Fva, yva = extract(va)
print(Ftr.shape)
torch.save((Ftr, ytr, Fva, yva), 'features.pt')


head = nn.Linear(576, 2); opt = torch.optim.Adam(head.parameters(), 1e-3); loss_fn = nn.CrossEntropyLoss()
for ep in range(100):
    opt.zero_grad(); loss_fn(head(Ftr), ytr).backward(); opt.step()
print('val acc:', (head(Fva).argmax(1) == yva).float().mean().item())
print(sum(p.numel() for p in head.parameters()), 'обучаемых параметров')



body.classifier = head
body.eval()
img, label = va[0]
print(body(img[None]).argmax(1).item(), label)
torch.save(body.state_dict(), 'ants_bees.pt')


from PIL import Image
import torch.nn.functional as F

def predict_image(image_path, model, transform):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0) 
    
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
    
    classes = ['Муравей', 'Пчела'] 
    
    print(f"\nФото: {image_path}")
    for i, prob in enumerate(probs):
        print(f"  {classes[i]}: {prob.item():.4f} ({prob.item()*100:.2f}%)")

predict_image('C:/Users/user/Desktop/3ISIP-24-1/MDK-DATA/lab12/bee.jpg', body, tf) 
predict_image('C:/Users/user/Desktop/3ISIP-24-1/MDK-DATA/lab12/ant.jpg', body, tf)
predict_image('C:/Users/user/Desktop/3ISIP-24-1/MDK-DATA/lab12/cat.jpg', body, tf) 


total_params = sum(p.numel() for p in body.parameters())
head_params = sum(p.numel() for p in head.parameters())
percent = (head_params / total_params) * 100

print(f"Всего параметров в MobileNet: {total_params:,}")
print(f"Обучаемых параметров (наша голова): {head_params:,}")
print(f"Процент обученных параметров: {percent:.3f}%")
print("\nВывод: Мы заморозили основу (Feature Extraction) и обучили крошечную добавку.")
print("Это та же идея, что лежит в основе LoRA для больших языковых моделей (LLM).")