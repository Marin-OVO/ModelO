import torch

def custom_collate_fn(batch):
    images = []
    fidt_map = []
    density_map = []
    points_list = []

    for img, target_dict in batch:
        images.append(img)
        fidt_map.append(target_dict['fidt_map'])
        density_map.append(target_dict['density_map'])
        points_list.append(target_dict['points'])

    return (
        torch.stack(images, dim=0),
        {
            'fidt_map': torch.stack(fidt_map, dim=0),
            'density_map': torch.stack(density_map, dim=0),
            'points': points_list
        }
    )