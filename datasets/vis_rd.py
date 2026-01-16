import torch
import matplotlib.pyplot as plt
from PIL import Image

from datasets.transforms import PointsToMask, RD


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

rd = RD(sigma_base=2.5, sigma_density=3, lambda_inhibit=0.5, gamma=1.0, add_fidt=False, build_qs=True)
img_t, rd_map = rd(image, target)
rd_map = rd_map.squeeze(0)

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.title("Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Reaction-Diffusion Heatmap")
plt.imshow(rd_map, cmap="jet")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(image)
plt.imshow(rd_map, cmap="jet", alpha=0.5)
plt.axis("off")

plt.show()
