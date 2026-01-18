from torchvision.datasets import CIFAR100
import torch
import torchvision
import random
import numpy as np
from pathlib import Path
import sys
from torch.optim.lr_scheduler import SequentialLR, MultiStepLR, LinearLR

from model import ourModel
parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir.parent))
from layers.lorentz import Lorentz
import time

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(0)

dataset = CIFAR100(root='../data', train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                                      torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

manifold = Lorentz(k=0.1)

model = ourModel(manifold).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

for epoch in range(200):
    running_loss = 0.0
    total_correct, total_samples = 0, 0
    start = time.time()
    for idx, (images, labels) in enumerate(dataloader):
        images = images.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        images = images.permute(0, 2, 3, 1)
        images = manifold.expmap0(images)
        images = images.permute(0, 3, 1, 2)


        output = model(images)
        loss = loss_fn(output, labels)
        # print(loss)
        loss.backward()
        # print(total_norm.item())
        optimizer.step()
        running_loss = running_loss * (idx / (idx + 1)) + loss.item() * (1 / (idx + 1))
        preds = output.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    print(f"TIME: {time.time() - start}")
    # lr_scheduler.step()
        # if idx % 10 == 0 and idx > 0:
        #     print(f'Batch {idx}, Average Loss: {running_loss:.3f}, Accuracy: {(output.argmax(dim=1) == labels).float().mean().item() * 100:.2f}%')
    print(f'Epoch {epoch+1} completed. Average Loss: {running_loss:.3f}, Accuracy: {(total_correct / total_samples) * 100:.2f}%')