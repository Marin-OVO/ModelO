import torch
import matplotlib.pyplot as plt
from PIL import Image

from datasets.transforms import PointsToMask, DensityMap, FIDT


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

img_t, fidt_map = FIDT(alpha=0.02, beta=0.75, radius=1, down_ratio=1)(image, target) # fidt_map: Tensor (3, H, W)
fidt_map = fidt_map.squeeze(0)

gs = DensityMap(sigma=4.0)
img_t, gs_map = gs(image, target)
gs_map = gs_map.squeeze(0)

mask_f = mask.float()
gs_f = gs_map.float()
fidt_f = fidt_map.float()
joint_map = mask_f * gs_f
joint_map_1 = fidt_f * gs_f

plt.figure(figsize=(18, 5))

plt.subplot(1, 4, 1)
plt.title("Mask")
plt.imshow(fidt_f, cmap="jet")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Density map")
plt.imshow(gs_f, cmap="jet")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Mask × Density")
plt.imshow(joint_map_1, cmap="jet")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Overlay")
plt.imshow(image)
plt.imshow(joint_map, cmap="jet", alpha=0.5)
plt.axis("off")

plt.show()
