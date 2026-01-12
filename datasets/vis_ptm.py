import torch
import matplotlib.pyplot as plt
from PIL import Image

from datasets.transforms import PointsToMask


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

ptm = PointsToMask(radius=2, num_classes=2, onehot=False, squeeze=True)
img_t, mask = ptm(image, target) # mask: Tensor (256, 256)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.title("Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("PointToMask")
plt.imshow(mask, cmap="jet")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(image)
plt.imshow(mask, cmap="jet", alpha=0.5)
plt.axis("off")

plt.show()
