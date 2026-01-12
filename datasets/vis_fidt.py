import torch
import matplotlib.pyplot as plt
from PIL import Image

from datasets.transforms import FIDT


image = Image.open('img.png').convert('RGB')
points = []
with open('ann.txt', 'r') as f:
    for line in f:
        if line.strip() == '':
            continue
        x, y = map(float, line.strip().split())
        points.append([x, y])
points = torch.tensor(points, dtype=torch.int64)
labels = torch.ones(len(points), dtype=torch.int64)

target = {
    "points": points,
    "labels": labels
}

img_t, fidt_map = FIDT(alpha=0.02, beta=0.75, radius=1, down_ratio=2)(image, target) # fidt_map: Tensor (3, H, W)
fidt_map = fidt_map.squeeze(0)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.title("Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("FIDT")
plt.imshow(fidt_map, cmap="jet")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(image)
plt.imshow(fidt_map, cmap="jet", alpha=0.5)
plt.axis("off")

plt.show()
