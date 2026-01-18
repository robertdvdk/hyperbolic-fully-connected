from torchvision.datasets import CIFAR100
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import random
import numpy as np
from pathlib import Path
import sys
from torch.optim.lr_scheduler import SequentialLR, MultiStepLR, LinearLR

parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir.parent))
from layers import LorentzConvNet, Lorentz
import time

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(0)



trainset = torchvision.datasets.CIFAR100("./cifar", train=True, download=True, transform=torchvision.transforms.Compose(
    [torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276))]
    ))
valset = torchvision.datasets.CIFAR100("./cifar", train=False, download=True, transform=torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize((0.5074, 0.4867, 0.4411), (0.267, 0.256, 0.276)),]
    ))
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False)

manifold = Lorentz(k=1.0)
model = LorentzConvNet(input_dim=3, hidden_dim=16, num_classes=100, num_layers=5, manifold=manifold, activation=nn.ReLU()).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.compile()

# warmup_scheduler = LinearLR(optimizer,
#                             start_factor=0.01,
#                             end_factor=1.0,
#                             total_iters=10)
# step_scheduler = MultiStepLR(
#     optimizer, milestones=[m - 10 for m in [60, 120, 160]], gamma=0.2
# )
# lr_scheduler = SequentialLR(optimizer,
#                             schedulers=[warmup_scheduler, step_scheduler],
#                             milestones=[10])
loss_fn = torch.nn.CrossEntropyLoss()

params = 0
# for n, p in model.named_parameters():
#     print(n, p.numel())
#     if not n.endswith('auxiliary'):
#         params += p.numel()

# print(params)


for _ in range(100):
    running_loss, acc, counts = 0.0, 0.0, 0
    start = time.time()
    for step, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(x.cuda()).squeeze()
        loss = F.cross_entropy(logits, y.cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if running_loss == 0.0:
            running_loss = loss.item()
        else:
            running_loss = 0.99*running_loss + 0.01*loss.item()
        acc += (logits.argmax(dim=1) == y.cuda()).float().sum()
        counts += x.shape[0]
        if step % 200 == 0:
            print("running loss:", running_loss)
    print("training acc:", acc / counts)
    print("time:", time.time() - start)
    with torch.no_grad():
        running_loss, acc, counts = 0.0, 0.0, 0
        for x, y in val_loader:
            logits = model(x.cuda()).squeeze()
            loss = F.cross_entropy(logits, y.cuda(), reduction='sum')
            running_loss += loss.item()
            acc += (logits.argmax(dim=1) == y.cuda()).float().sum()
            counts += x.shape[0]
        
        print("val loss:", running_loss / counts)
        print("val acc:", acc / counts)