import PIL
from PIL import Image
import numpy
import numpy as np
import torch
import torchvision
import torch.nn.functional as F

import scipy
from typing import Dict, Optional, Union, Tuple, List, Any

from utils.registry import Registry

TRANSFORMS = Registry(name = 'transforms', module_key = 'datasets.transforms')

__all__ = ['TRANSFORMS', *TRANSFORMS.registry_names]


def _point_buffer(x: int, y: int, mask: torch.Tensor, radius: int) -> torch.Tensor:
    x_t, y_t = torch.arange(0, mask.size(1)), torch.arange(0, mask.size(0))
    buffer = (x_t.unsqueeze(0) - x) ** 2 + (y_t.unsqueeze(1) - y) ** 2 < radius ** 2

    return buffer

# def gaussian_blur_torch(x: torch.Tensor, sigma: float) -> torch.Tensor:
#     """
#         x: (1, 1, H, W)
#     """
#     radius = int(3 * sigma)
#     size = 2 * radius + 1
#
#     coords = torch.arange(size, device=x.device) - radius
#     kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
#     kernel = kernel / kernel.sum()
#
#     kernel_x = kernel.view(1, 1, 1, -1)
#     kernel_y = kernel.view(1, 1, -1, 1)
#
#     x = F.conv2d(x, kernel_x, padding=(0, radius))
#     x = F.conv2d(x, kernel_y, padding=(radius, 0))
#
#     return x


@TRANSFORMS.register()
class MultiTransformsWrapper:
    """
        Independently applies each input transformation to the called input and
        returns the results separately in the same order as the specified transforms

        Args:
            transforms(list): list of transforms that take image (PIL or Tensor) and
                target (dict) as inputs
    """

    def __init__(self, transforms: List[object]) -> None:
        self.transforms = transforms

    def __call__(
            self,
            image: Union[PIL.Image.Image, torch.Tensor],
            target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """
            Args:
                image (PIL.Image.Image or torch.Tensor): image of reference [C,H,W], only for
                    pipeline convenience
                target (dict): corresponding target containing at least 'points' and 'labels'
                    keys, with torch.Tensor as value. Labels must be integers!

            Returns:
                Tuple[torch.Tensor, Tuple[torch.Tensor]]:
                    the transormed image and the tuple of transformed outputs in the same
                    order as the specified transforms
        """

        outputs = []
        for trans in self.transforms:
            img, tr_trgt = trans(image, target)
            outputs.append(tr_trgt)

        return img, tuple(outputs)


@TRANSFORMS.register()
class SampleToTensor:
    """
        Convert image and target to Tensors
    """
    def __call__(
            self,
            img: Union[Image.Image, torch.Tensor],
            target: Dict[str, torch.Tensor],
            crow_type: str = 'point'
            ) -> Tuple[torch.Tensor, str]:
        """
            img: PIL image with [C, H, W] shape
            target: corresponding target
            crowd_type: crowd-labeled type, including point, bbox, and density. In this project, point type is used.
            return: Tuple[torch.Tensor, dict]: the transormed image and target
        """

        tr_img = torchvision.transforms.ToTensor()(img)

        tr_target = {}
        tr_target.update(dict(**target))

        tr_target['points'] = torch.as_tensor(tr_target['points'], dtype=torch.int64)

        tr_target['labels'] = torch.as_tensor(tr_target['labels'], dtype=torch.int64)

        return tr_img, tr_target


@TRANSFORMS.register()
class UnNormalize:
    "Reverse normalization"

    def __init__(
            self,
            mean: Optional[Tuple[float, float, float]] = (0.485, 0.456, 0.406),
            std: Optional[Tuple[float, float, float]] = (0.229, 0.224, 0.225)
            ) -> None:

        self.mean = mean
        self.std = std

    def __call__(self, img: torch.Tensor) -> torch.Tensor:

        for i, m, s in zip(img, self.mean, self.std):
            i.mul_(s).add_(m)

        return img


@TRANSFORMS.register()
class Normalize:
    """
        normalization
    """
    def __init__(
            self,
            mean: Optional[Tuple[float, float, float]] = (0.485, 0.456, 0.406),
            std: Optional[Tuple[float, float, float]] = (0.229, 0.224, 0.225)
            ) -> None:

        self.mean = mean
        self.std = std

    def __call__(self, image: torch.Tensor, target) -> torch.Tensor:
        if self.mean is not None and self.std is not None:
            mean = torch.tensor(self.mean, device=image.device).view(-1, 1, 1)
            std = torch.tensor(self.std, device=image.device).view(-1, 1, 1)
            image = (image - mean) / std

        return image, target


# gt points annotation -> /2
@TRANSFORMS.register()
class DownSample:
    """
        DownSample img by a ratio
    """
    def __init__(
            self,
            down_ratio: int = 2,
            crowd_type: str = 'point'
            ) -> None:

        assert crowd_type in ['bbox', 'point'], \
            f'Annotations type must be \'bbox\' or \'point\', got \'{crowd_type}\''

        self.down_ratio = down_ratio
        self.crowd_type = crowd_type

    def __call__(
            self,
            img: Union[Image.Image, torch.Tensor],
            target: Dict[str, torch.Tensor]
            ) -> Dict[str, torch.Tensor]:

        if isinstance(img, PIL.Image.Image):
            img = torchvision.transforms.ToTensor()(img)

        target['points'] = torch.div(target['points'], self.down_ratio, rounding_mode='floor')

        return img, target


# point -> GS mask
@TRANSFORMS.register()
class PointsToMask:
    """
        Convert points annotation to mask with a buffer option
        based on https://github.com/Alexandre-Delplanque/HerdNet/blob/main/animaloc/data/transforms.py
    """

    def __init__(
            self,
            radius: int = 1,
            num_classes: int = 2,
            onehot: bool = False,
            squeeze: bool = True,
            down_ratio: Optional[int] = None,
            target_type: str = 'long'
    ) -> None:
        """
            Args:
                radius (int, optional): buffer (pixel radius) to define a point in
                    the mask. Defautls to 1 (i.e. non buffer)
                num_classes (int, optional): number of classes, background included.
                    Defaults to 2
                onehot (bool, optional): set to True do enable one-hot encoding.
                    Defaults to False
                squeeze (bool, optional): when onehot is False, set to True to squeeze the
                    mask to get a Tensor of shape [H,W], otherwise the returned mask has
                    a shape of [1,H,W].
                    Defaults to False
                down_ratio (int, optional): if specified, the target will be downsampled
                    according to the ratio.
                    Defaults to None
                target_type (str, optional): output data type of target. Defaults to 'long'.
        """

        assert target_type in ['long', 'float'], \
            f"target type must be either 'long' or 'float', got {target_type}"

        self.radius = radius
        self.num_classes = num_classes - 1
        self.onehot = onehot
        self.squeeze = squeeze
        self.down_ratio = down_ratio
        self.target_type = target_type


    def __call__(
            self,
            image: Union[PIL.Image.Image, torch.Tensor],
            target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Args:
                image (PIL.Image.Image or torch.Tensor): image of reference [C,H,W], only for
                    pipeline convenience
                target (dict): corresponding target containing at least 'points' and 'labels'
                    keys, with torch.Tensor as value. Labels must be integers!

            Returns:
                Tuple[torch.Tensor, torch.Tensor]:
                    the transormed image and the mask
        """
        if isinstance(image, PIL.Image.Image):
            image = torchvision.transforms.ToTensor()(image)

        self.img_height, self.img_width = image.size(1), image.size(2)
        if self.down_ratio is not None:
            self.img_height = self.img_height // self.down_ratio
            self.img_width = self.img_width // self.down_ratio
            _, target = DownSample(down_ratio=self.down_ratio, crowd_type='point')(
                image, target.copy()
            )

        mask = torch.zeros((1, self.img_height, self.img_width)).long()

        # fill the mask
        if len(target['points']) > 0:
            for point, label in zip(target['points'], target['labels']):
                x, y = point[0], point[1]
                point_buffer = _point_buffer(x, y, mask[0], self.radius)
                mask[0, point_buffer] = label

        if self.onehot:
            mask = self._onehot(mask)

        if self.squeeze:
            mask = mask.squeeze(0) # (H, W)

        if self.target_type == 'float':
            mask = mask.float()

        return image, mask # -> hard disk mask (1, H, W)

    def _onehot(self, mask: torch.Tensor):
        onehot_mask = torch.nn.functional.one_hot(mask, self.num_classes + 1)
        onehot_mask = torch.movedim(onehot_mask, -1, -3)
        return onehot_mask


# FIDT
@TRANSFORMS.register()
class FIDT:
    """
        Convert points annotations into Focal-Inverse-Distance-Transform map.

        In case of multi-class, returns one-hot encoding masks.

        For binary case, you can let the num_classes argument by default, this will return a
        density map of one channel only [1, H, W].

        Inspired from:
        Liang et al. (2021) - "Focal Inverse Distance Transform Maps for Crowd Localization
        and Counting in Dense Crowd"
    """

    def __init__(
            self,
            alpha: float = 0.02,
            beta: float = 0.75,
            c: float = 1.0,
            radius: int = 1,
            num_classes: int = 2,
            add_bg: bool = False,
            down_ratio: Optional[int] = None
    ) -> None:
        """
            Args:
                alpha (float, optional): parameter, can be adjusted. Defaults to 0.02
                beta (float, optional): parameter, can be adjusted. Defaults to 0.75
                c (float, optional): parameter, can be adjusted. Defaults to 1.0
                radius (int, optional): buffer (pixel radius) to define a point in
                    the mask. Defautls to 1 (i.e. non buffer)
                num_classes (int, optional): number of classes, background included. If
                    higher than 2, returns one-hot encoding masks [C, H, W], otherwise
                    returns a binary mask [1, H, W] even if different categories of labels
                    are called. Defaults to 2
                add_bg (bool, optional): set to True to add background map in any case. It
                    is built by substracting all positive locations from ones tensor.
                    Defaults to False
                down_ratio (int, optional): if specified, the target will be downsampled
                    according to the ratio.
                    Defaults to None
        """

        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.radius = radius
        self.num_classes = num_classes - 1
        self.add_bg = add_bg
        self.down_ratio = down_ratio

    def __call__(
        self,
        image: Union[PIL.Image.Image, torch.Tensor],
        target: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Args:
                image (PIL.Image.Image or torch.Tensor): image of reference [C,H,W], only for
                    pipeline convenience
                target (dict): corresponding target containing at least 'points' and 'labels'
                    keys, with torch.Tensor as value. Labels must be integers!

            Returns:
                Tuple[torch.Tensor, torch.Tensor]:
                    the transormed image and the FIDT map(s)
        """

        if isinstance(image, PIL.Image.Image):
            image = torchvision.transforms.ToTensor()(image)

        self.img_height, self.img_width = image.size(1), image.size(2)
        if self.down_ratio is not None:
            self.img_height = self.img_height // self.down_ratio
            self.img_width = self.img_width // self.down_ratio
            _, target = DownSample(down_ratio=self.down_ratio, crowd_type='point')(
                image, target.copy()
            )

        if self.num_classes == 1:
            new_target = target.copy()
            new_target.update(labels=[1] * len(new_target['labels']))
            dist_map = self._onehot(image, new_target)
        else:
            dist_map = self._onehot(image, target)

        if self.add_bg:
            dist_map = self._add_background(dist_map)

        return image, dist_map.type(image.type())

    def _get_fidt(self, mask: torch.Tensor) -> torch.Tensor:

        dist_map = scipy.ndimage.distance_transform_edt(mask)
        dist_map = torch.from_numpy(dist_map)
        dist_map = 1 / (torch.pow(dist_map, self.alpha * dist_map + self.beta) + self.c)
        dist_map = torch.where(dist_map < 0.01, 0., dist_map)

        return dist_map

    def _onehot(self, image: torch.Tensor, target: torch.Tensor):

        dist_maps = torch.zeros((self.num_classes, self.img_height, self.img_width))

        if len(target['points']) > 0:
            labels = numpy.unique(target['labels'])
            masks = torch.ones((self.num_classes, self.img_height, self.img_width))

            for point, label in zip(target['points'], target['labels']):
                x, y = point[0], point[1]
                point_buffer = _point_buffer(x, y, masks[label - 1], self.radius)
                masks[label - 1, point_buffer] = 0

            dist_maps = torch.ones((self.num_classes, self.img_height, self.img_width), dtype=torch.float64)
            for i, mask in enumerate(masks):
                mask = self._get_fidt(mask)
                if i + 1 in labels:
                    dist_maps[i] = mask
                else:
                    dist_maps[i] = torch.zeros((self.img_height, self.img_width), dtype=torch.float64)

        return dist_maps

    def _add_background(self, dist_map: torch.Tensor) -> torch.Tensor:
        background = torch.ones((1, *dist_map.shape[1:]))
        merged_dist = dist_map.sum(dim=0, keepdim=True)
        background = torch.sub(background, merged_dist)
        output = torch.cat((background, dist_map), dim=0)

        return output


@TRANSFORMS.register()
class DensityMap:
    """
        Convert point annotations into Gaussian density map.

        Each Gaussian kernel is explicitly normalized such that:
            sum(kernel) = 1

        Therefore:
            sum(density_map) = number of points

        This transform is intended ONLY for counting supervision.
    """
    def __init__(
        self,
        sigma: float = 4.0,
        down_ratio: Optional[int] = None
    ) -> None:
        """
            Args:
                sigma (float): standard deviation of Gaussian kernel
                down_ratio (int, optional): downsample ratio for density map
        """
        self.sigma = sigma
        self.down_ratio = down_ratio

    def __call__(
        self,
        image: Union[PIL.Image.Image, torch.Tensor],
        target: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if isinstance(image, PIL.Image.Image):
            image = torchvision.transforms.ToTensor()(image)

        H, W = image.size(1), image.size(2)

        if self.down_ratio is not None:
            H = H // self.down_ratio
            W = W // self.down_ratio
            _, target = DownSample(
                down_ratio=self.down_ratio,
                crowd_type='point'
            )(image, target.copy())

        density_map = torch.zeros((1, H, W), dtype=torch.float32)

        if len(target['points']) > 0:
            for point in target['points']:
                x, y = int(point[0]), int(point[1])
                self._add_gaussian(density_map[0], x, y)

        return image, density_map.type(image.type())

    def _add_gaussian(self, density: torch.Tensor, x: int, y: int):
        """
            Add a normalized 2D Gaussian centered at (x, y)
        """

        H, W = density.shape
        radius = int(3 * self.sigma)
        size = 2 * radius + 1

        yy, xx = torch.meshgrid(
            torch.arange(size),
            torch.arange(size),
            indexing='ij'
        )

        cy = cx = radius
        gaussian = torch.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * self.sigma ** 2)
        )

        gaussian = gaussian / gaussian.sum()

        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(W, x + radius + 1)
        y2 = min(H, y + radius + 1)

        gx1 = radius - (x - x1)
        gy1 = radius - (y - y1)
        gx2 = gx1 + (x2 - x1)
        gy2 = gy1 + (y2 - y1)

        density[y1:y2, x1:x2] += gaussian[gy1:gy2, gx1:gx2]


class CustomTransformWrapper:
    def __init__(self, fidt_transform, density_transform):
        self.fidt_transform = fidt_transform
        self.density_transform = density_transform

    def __call__(self, image, target):
        original_points = target['points'].clone()

        img1, fidt_map = self.fidt_transform(image, target)
        img2, density_map = self.density_transform(image, target)

        return img1, {
            'fidt_map': fidt_map,
            'density_map': density_map,
            'points': original_points
        }


# class RD:
#     """
#         Reaction–Diffusion based GT generator
#     """
#     def __init__(
#         self,
#         sigma_base: float=2.5,
#         sigma_density: float=3.0,
#         lambda_inhibit: float=0.4,
#         gamma: float = 1.0,
#         num_classes: int = 2,
#         add_bg: bool = False,
#         down_ratio: int = None,
#         add_fidt: bool = False,
#         fidt_alpha: float = 0.02,
#         fidt_beta: float = 0.75,
#         build_qs: bool = False,
#         sigma_qs: float = 8.0,
#     ) -> None: # 2.5, 3.0, 0.4, 1.0
#         self.sigma_base = sigma_base
#         self.sigma_density = sigma_density
#         self.lambda_inhibit = lambda_inhibit
#         self.gamma = gamma
#         self.num_classes = num_classes - 1
#         self.add_bg = add_bg
#         self.down_ratio = down_ratio
#         self.add_fidt = add_fidt
#         self.fidt_alpha = fidt_alpha
#         self.fidt_beta = fidt_beta
#
#         self.build_qs = build_qs
#         self.sigma_qs = sigma_qs
#
#     def __call__(
#         self,
#         image: Union[PIL.Image.Image, torch.Tensor],
#         target: Dict[str, torch.Tensor],
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#
#         if isinstance(image, PIL.Image.Image):
#             image = torchvision.transforms.ToTensor()(image)
#
#         self.img_height, self.img_width = image.size(1), image.size(2)
#         if self.down_ratio is not None:
#             self.img_height = self.img_height // self.down_ratio
#             self.img_width = self.img_width // self.down_ratio
#             _, target = DownSample(down_ratio=self.down_ratio, crowd_type='point')(
#                 image, target.copy()
#             )
#
#         if self.add_fidt:
#             all_maps = []
#             device = target["points"].device if "points" in target else torch.device("cpu")
#             for cls in range(self.num_classes):
#                 cls_points = target["points"][target["labels"] == (cls + 1)]
#                 hybrid_map = self._hybrid_transform(cls_points, device)
#                 all_maps.append(hybrid_map)
#             rd_map = torch.stack(all_maps, dim=0) if self.num_classes > 1 else all_maps[0].unsqueeze(0)
#         else:
#             if self.num_classes == 1:
#                 new_target = target.copy()
#                 new_target["labels"] = torch.ones_like(target["labels"])
#                 rd_map = self._onehot(new_target)
#             else:
#                 rd_map = self._onehot(target)
#
#         if self.add_bg:
#             rd_map = self._add_background(rd_map)
#
#         rd_map = rd_map.type(image.type())
#
#         if self.build_qs:
#             if self.num_classes == 1:
#                 qs_map = self._build_qs_gt(
#                     target["points"],
#                     self.img_height,
#                     self.img_width,
#                     self.sigma_qs
#                 )
#             else:
#                 qs_maps = []
#                 device = target["points"].device
#                 for cls in range(self.num_classes):
#                     cls_points = target["points"][target["labels"] == (cls + 1)]
#                     qs_map_cls = self._build_qs_gt(
#                         cls_points,
#                         self.img_height,
#                         self.img_width,
#                         self.sigma_qs
#                     )
#                     qs_maps.append(qs_map_cls)
#                 qs_map = torch.cat(qs_maps, dim=0) if len(qs_maps) > 1 else qs_maps[0]
#
#             qs_map = qs_map.type(image.type())
#             return image, qs_map
#
#         return image, rd_map
#
#     def _onehot(self, target: Dict[str, torch.Tensor]) -> torch.Tensor:
#         """
#             Generate RD gs_map per class
#         """
#         device = target["points"].device
#         rd_map = torch.zeros(
#             (self.num_classes, self.img_height, self.img_width),
#             device=device,
#         )
#
#         if len(target["points"]) == 0:
#             return rd_map
#
#         for cls in range(self.num_classes):
#             cls_points = target["points"][target["labels"] == (cls + 1)]
#             if len(cls_points) == 0:
#                 continue
#
#             rd_map[cls] = self._reaction_diffusion(cls_points, device)
#
#         return rd_map
#
#     def _reaction_diffusion(
#         self,
#         points: torch.Tensor,
#         device: torch.device,
#     ) -> torch.Tensor:
#         """
#             Core RD formulation:
#             u = G_base(S) / (1 + λ * G_density(S))
#         """
#         # S(x)
#         S = torch.zeros((1, 1, self.img_height, self.img_width), device=device)
#         indices_y = points[:, 1].long().clamp(0, self.img_height - 1)
#         indices_x = points[:, 0].long().clamp(0, self.img_width - 1)
#         S.view(-1).put_(indices_y * self.img_width + indices_x,
#                         torch.ones(len(points), device=device), accumulate=True)
#
#         # GdS = Gd * S(x)
#         GdS = gaussian_blur_torch(S, self.sigma_density)
#         # GbS = Gb * S(x)
#         GbS = gaussian_blur_torch(S, self.sigma_base)
#
#         # U = Gb * S(x) / (1 + lambda * Gd * S(x))
#         # U = Gb * S(x) / (eps + 1.0 + lambda * Gd * S(x))
#         # U = Gb * S(x) / (eps + Gb * S(x) + lambda * Gd * S(x))
#         u = GbS / (1.0 + self.lambda_inhibit * GdS + 1e-6)
#
#         u = u / (u.max() + 1e-7)
#         u = torch.pow(u, self.gamma)
#         u = torch.max(u, S)
#
#         return u.squeeze(0)
#
#     def _add_background(self, heatmap: torch.Tensor) -> torch.Tensor:
#         background = torch.ones((1, *heatmap.shape[1:]), device=heatmap.device)
#         merged = heatmap.sum(dim=0, keepdim=True)
#         background = torch.clamp(background - merged, min=0.0)
#
#         return torch.cat((background, heatmap), dim=0)
#
#     def _hybrid_transform(
#             self,
#             points: torch.Tensor,
#             device: torch.device,
#     ) -> torch.Tensor:
#         """
#             Hybrid RD + FIDT transform
#             Uses FIDT implementation consistent with FIDT class
#         """
#         H, W = self.img_height, self.img_width
#
#         # Create point mask S(x)
#         S = torch.zeros((1, 1, self.img_height, self.img_width), device=device)
#         indices_y = points[:, 1].long().clamp(0, self.img_height - 1)
#         indices_x = points[:, 0].long().clamp(0, self.img_width - 1)
#         S.view(-1).put_(indices_y * self.img_width + indices_x,
#                         torch.ones(len(points), device=device), accumulate=True)
#
#         GdS = gaussian_blur_torch(S, self.sigma_density)
#         inhibition_factor = 1.0 / (1.0 + self.lambda_inhibit * GdS + 1e-6)
#
#         mask = torch.ones((self.img_height, self.img_width), device=device)
#         if len(points) > 0:
#             for point in points:
#                 x, y = point[0], point[1]
#                 point_buffer = _point_buffer(x, y, mask, 1)
#                 mask[point_buffer] = 0
#
#         mask_cpu = mask.cpu().numpy()
#         dist_map = scipy.ndimage.distance_transform_edt(mask_cpu)
#         dist_map = torch.from_numpy(dist_map).to(device=device)
#
#         fidt_map = 1 / (torch.pow(dist_map, self.fidt_alpha * dist_map + self.fidt_beta) + 1.0)
#         fidt_map = torch.where(fidt_map < 0.01, 0., fidt_map)
#
#         hybrid_map = fidt_map * inhibition_factor.squeeze(0).squeeze(0)
#
#         hybrid_map = hybrid_map / (hybrid_map.max() + 1e-7)
#         hybrid_map = torch.pow(hybrid_map, self.gamma)
#         hybrid_map = torch.max(hybrid_map, S.squeeze(0).squeeze(0))
#
#         return hybrid_map # (H, W)
#
#     def build_qs_gt(points, H, W, sigma_qs):
#         S = torch.zeros((1, 1, H, W), device=points.device)
#
#         if points.numel() > 0:
#             y = points[:, 1].long().clamp(0, H - 1)
#             x = points[:, 0].long().clamp(0, W - 1)
#             S.view(-1).put_(y * W + x, torch.ones(len(points), device=points.device), accumulate=True)
#
#         qs = gaussian_blur_torch(S, sigma_qs)
#         qs = qs / (qs.max() + 1e-6)
#
#         return qs.squeeze(0)  # (1,H,W)
#
#     def _build_qs_gt(
#             self,
#             points: torch.Tensor,
#             H: int,
#             W: int,
#             sigma_qs: float
#     ) -> torch.Tensor:
#         """
#             Build query selection ground truth map using Gaussian blur
#
#             Args:
#                 points: Point annotations tensor of shape (N, 2) where N is number of points
#                 H: Height of output map
#                 W: Width of output map
#                 sigma_qs: Sigma parameter for Gaussian blur
#
#             Returns:
#                 Gaussian blurred and normalized map of shape (1, H, W)
#         """
#         if not isinstance(points, torch.Tensor):
#             points = torch.as_tensor(points, dtype=torch.long)
#
#         device = points.device
#         S = torch.zeros((1, 1, H, W), device=device)
#
#         if points.numel() > 0:
#             if points.dim() == 1:
#                 points = points.unsqueeze(0)
#
#             y = points[:, 1].long().clamp(0, H - 1)
#             x = points[:, 0].long().clamp(0, W - 1)
#
#             S.view(-1).put_(y * W + x, torch.ones(len(points), device=device), accumulate=True)
#
#         qs = gaussian_blur_torch(S, sigma_qs)
#
#         max_val = qs.max()
#         if max_val > 0:
#             qs = qs / max_val
#
#         return qs.squeeze(0)  # (1, H, W)

