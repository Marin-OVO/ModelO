import torch

def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_heatmap(model):
    for p in model.heatmap_head.parameters():
        p.requires_grad = True

def unfreeze_offset(model):
    for p in model.offset_head.parameters():
        p.requires_grad = True

def unfreeze_density(model):
    for p in model.density_head.parameters():
        p.requires_grad = True

def unfreeze_backbone(model):
    # Encoder
    for p in model.inc.parameters():
        p.requires_grad = True
    for p in model.down1.parameters():
        p.requires_grad = True
    for p in model.down2.parameters():
        p.requires_grad = True
    for p in model.down3.parameters():
        p.requires_grad = True
    for p in model.down4.parameters():
        p.requires_grad = True

    # Decoder
    for p in model.up1.parameters():
        p.requires_grad = True
    for p in model.up2.parameters():
        p.requires_grad = True
    for p in model.up3.parameters():
        p.requires_grad = True
    for p in model.up4.parameters():
        p.requires_grad = True